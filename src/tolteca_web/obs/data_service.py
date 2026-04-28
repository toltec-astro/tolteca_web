"""ObsDataService — partial-access service for TolTEC observation data.

Wraps an :class:`~.catalog.ObsCatalogBackend` and a zarr cache to provide
lazy, channel-sliced xarray Datasets for any TolTEC raw observation data kind
(sweeps, timestreams).
"""

from __future__ import annotations

import functools
from pathlib import Path

import xarray as xr

from .catalog import ObsCatalogBackend

import polars as pl

__all__ = ["ObsDataService", "DataNotFoundError"]


class DataNotFoundError(LookupError):
    """Raised when a requested zarr store is missing or not yet ingested."""


class ObsDataService:
    """Partial-access service for TolTEC raw observation data.

    ``get_obs_data`` opens the zarr store lazily and slices only the
    requested channels and samples — no full-file IO.

    Parameters
    ----------
    cache_root
        Root directory of the zarr cache
        (e.g. ``run/web_v3/cache``).
    catalog
        Any :class:`~.catalog.ObsCatalogBackend` implementation.
    zarr_base_url
        When set, zarr stores are opened via HTTP at this base URL instead
        of local file paths.  The URL must expose the same directory tree as
        ``cache_root / "zarr"``.
        Example: ``"http://localhost:8052/zarr"``.
        Requires ``fsspec`` + ``aiohttp`` to be installed.
        When ``None`` (default), local file I/O is used.
    db_url
        Optional tolteca DuckDB URL (``duckdb:///path/to/tolteca.duckdb``).
        When set, enables :meth:`get_analysis_groups` which queries analysis
        data products (OOF, Focus, Drivefit, etc.) directly from the DB.
    """

    def __init__(
        self,
        cache_root: Path,
        catalog: ObsCatalogBackend,
        zarr_base_url: str | None = None,
        db_url: str | None = None,
    ) -> None:
        self._cache_root = Path(cache_root)
        self._catalog = catalog
        self._zarr_base_url = zarr_base_url.rstrip("/") if zarr_base_url else None
        self._db_url = db_url
        # Instance-level caches — survive across Dash callbacks
        self._row_cache: dict[tuple, dict | None] = {}
        self._store_cache: dict[str, xr.Dataset] = {}

    @property
    def zarr_mode(self) -> str:
        """Return ``'http'`` or ``'local'`` indicating the active zarr access mode."""
        return "http" if self._zarr_base_url else "local"

    @property
    def cache_root(self) -> Path:
        """Return the cache root directory."""
        return self._cache_root

    # ── Catalog access ────────────────────────────────────────────────────

    def get_cal_groups(
        self,
        obsnum_range: tuple[int, int] | None = None,
        master: str | None = None,
    ) -> list[dict]:
        """Return inferred calibration groups (delegates to catalog backend).

        Returns an empty list if the backend does not support cal groups.
        """
        if hasattr(self._catalog, "query_cal_groups"):
            return self._catalog.query_cal_groups(
                obsnum_range=obsnum_range,
                master=master,
            )
        return []

    def get_analysis_groups(
        self,
        prod_type: str,
        master: str | None = None,
        obsnum_range: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Query analysis group data products from tolteca_db.

        Returns newest-first list of group dicts for the given
        ``prod_type`` (e.g. ``"dp_oof_group"``, ``"dp_focus_group"``,
        ``"dp_drivefit"``).

        Returns an empty list when no DB URL is configured or no
        matching products exist.

        Parameters
        ----------
        prod_type
            ``data_prod_type`` label.
        master
            Optional master filter (``"ics"`` or ``"tcs"``).
        obsnum_range
            Optional ``(min, max)`` obsnum filter.
        """
        if not self._db_url:
            return []
        try:
            from sqlalchemy import select
            from tolteca_db.db import create_database
            from tolteca_db.models.orm import DataProd, DataProdType
        except ImportError:
            return []

        try:
            db = create_database(self._db_url)
            rows_data: list[dict] = []
            with db.session() as session:
                stmt = (
                    select(DataProd)
                    .join(DataProdType, DataProd.data_prod_type_fk == DataProdType.pk)
                    .where(DataProdType.label == prod_type)
                    .where(DataProd.lifecycle_status == "active")
                    .order_by(DataProd.pk.desc())
                )
                dps = session.execute(stmt).scalars().all()
                for dp in dps:
                    dp_meta = dp.meta
                    name = getattr(dp_meta, "name", "") or ""
                    grp_master = getattr(dp_meta, "master", "") or ""
                    obsnum = getattr(dp_meta, "obsnum", 0) or 0
                    n_items = getattr(dp_meta, "n_items", 0) or 0
                    obs_datetime = getattr(dp_meta, "obs_datetime", None)
                    rows_data.append({
                        "name": name,
                        "master": grp_master,
                        "obsnum": obsnum,
                        "n_items": n_items,
                        "obs_datetime": str(obs_datetime) if obs_datetime else "",
                    })
            db.close()
        except Exception:
            return []

        result = []
        for row in rows_data:
            grp_master = row["master"]
            obsnum = row["obsnum"]
            if master is not None and grp_master != master:
                continue
            if obsnum_range is not None:
                lo, hi = obsnum_range
                if not (lo <= (obsnum or 0) <= hi):
                    continue
            # Parse obsnum_end from name (e.g. "tcs-152797to152808-g12-oof")
            obsnum_end = obsnum
            name = row["name"]
            if name and "to" in name:
                try:
                    part = name.split("-")[1]  # "152797to152808"
                    obsnum_end = int(part.split("to")[1])
                except (IndexError, ValueError):
                    pass
            date_str = (row["obs_datetime"] or "")[:10]
            result.append({
                "name": name,
                "master": grp_master,
                "obsnum_start": obsnum,
                "obsnum_end": obsnum_end,
                "n_items": row["n_items"],
                "date_utc": date_str,
            })
        return result

    def get_reduced_obs_groups(
        self,
        master: str | None = None,
        obsnum_range: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Return groups for targsweep observations with KIDs reduction results.

        Scans the parquet catalog for ``targsweep`` observations that have a
        completed KIDs pipeline (``kids_find/`` group present in zarr store).
        Returns newest-first list of group dicts in the same format as
        :meth:`get_analysis_groups`.

        Parameters
        ----------
        master
            Optional master filter (``"ics"`` or ``"tcs"``).
        obsnum_range
            Optional ``(min, max)`` obsnum filter.
        """
        df = self._catalog.query(
            data_kinds=["targsweep"],
            master=master,
            obsnum_range=obsnum_range,
        )
        if df.is_empty():
            return []

        # One representative row per (master, obsnum): pick any nw to check zarr
        quartets = (
            df.group_by(["master", "obsnum", "subobsnum", "scannum"])
            .agg([
                pl.col("date_utc").first(),
                pl.col("zarr_path").first(),
            ])
            .sort(["obsnum", "subobsnum", "scannum"])
        )

        result = []
        for row in quartets.iter_rows(named=True):
            zarr_rel = row["zarr_path"]
            if not zarr_rel:
                continue
            zarr_path = self._cache_root / zarr_rel
            # Fast filesystem check — avoid opening zarr
            kids_find_dir = zarr_path / "kids_find"
            if not kids_find_dir.exists():
                continue
            date_str = (row["date_utc"] or "")[:10]
            result.append({
                "master": row["master"],
                "obsnum_start": row["obsnum"],
                "obsnum_end": row["obsnum"],
                "subobsnum": row["subobsnum"],
                "scannum": row["scannum"],
                "n_items": 1,
                "date_utc": date_str,
            })

        result.sort(key=lambda r: (r["date_utc"], r["obsnum_start"]), reverse=True)
        return result

    def get_obs_catalog(
        self,
        data_kinds: list[str] | None = None,
        nw_list: list[int] | None = None,
        obsnum_range: tuple[int, int] | None = None,
        master: str | None = None,
    ) -> pl.DataFrame:
        """Return filtered catalog rows (delegates to the backend)."""
        return self._catalog.query(
            data_kinds=data_kinds,
            nw_list=nw_list,
            obsnum_range=obsnum_range,
            master=master,
        )

    def get_obs_row(
        self,
        master: str,
        obsnum: int,
        subobsnum: int,
        scannum: int,
        nw: int,
    ) -> dict | None:
        """Return the catalog row for the given quartet+nw as a dict.

        Parameters
        ----------
        master, obsnum, subobsnum, scannum, nw
            Observation identifiers.

        Returns
        -------
        dict | None
            Row as ``{column: value}`` dict, or ``None`` if not found.
        """
        cache_key = (master, obsnum, subobsnum, scannum, nw)
        if cache_key in self._row_cache:
            return self._row_cache[cache_key]
        df = self._catalog.query(master=master)
        if df.is_empty():
            self._row_cache[cache_key] = None
            return None
        row = df.filter(
            (pl.col("obsnum") == obsnum)
            & (pl.col("subobsnum") == subobsnum)
            & (pl.col("scannum") == scannum)
            & (pl.col("nw") == nw)
        )
        result = None if row.is_empty() else row.row(0, named=True)
        self._row_cache[cache_key] = result
        return result

    # ── Data access ───────────────────────────────────────────────────────

    def get_obs_data(
        self,
        master: str,
        obsnum: int,
        subobsnum: int,
        scannum: int,
        nw: int,
        chan_slice: slice | None = None,
        sample_slice: slice | None = None,
    ) -> xr.Dataset:
        """Return a lazy xarray Dataset for the requested quartet+nw.

        Only the zarr chunks for the selected channels are read from disk.
        ``chan_slice`` and ``sample_slice`` default to the full extent.

        Parameters
        ----------
        master, obsnum, subobsnum, scannum, nw
            Quartet + network index identifying the observation.
        chan_slice
            Channel range to load (default: all channels).
        sample_slice
            Sample range to load along the ``sample`` dimension
            (sweep steps for sweeps, time samples for timestreams).
            Default: all samples.

        Returns
        -------
        xr.Dataset
            Lazy Dataset with variables ``I``, ``Q``, ``tone_freq`` and
            coordinate ``lo_freq``.  Call ``.compute()`` to materialise.

        Raises
        ------
        DataNotFoundError
            If no zarr store exists for the requested quartet.
        """
        zarr_path = self._catalog.get_zarr_path(
            master, obsnum, subobsnum, scannum, nw
        )
        if zarr_path is None or not zarr_path.exists():
            raise DataNotFoundError(
                f"No zarr store for {master}/{obsnum:08d}/{subobsnum:03d}/"
                f"{scannum:04d}/{nw:02d}. Run ingest_obs.py first."
            )

        cache_key: str
        if self._zarr_base_url:
            rel = zarr_path.relative_to(self._cache_root / "zarr")
            cache_key = f"{self._zarr_base_url}/{rel}"
        else:
            cache_key = str(zarr_path)

        if cache_key not in self._store_cache:
            if self._zarr_base_url:
                ds_full = xr.open_zarr(cache_key, chunks=None)
            else:
                ds_full = xr.open_zarr(cache_key, chunks=None)
            self._store_cache[cache_key] = ds_full

        ds = self._store_cache[cache_key]
        sel: dict[str, slice] = {}
        if chan_slice is not None:
            sel["chan"] = chan_slice
        if sample_slice is not None:
            sel["sample"] = sample_slice
        if sel:
            ds = ds.isel(**sel)
        return ds

    def get_zarr_path(
        self,
        master: str,
        obsnum: int,
        subobsnum: int,
        scannum: int,
        nw: int,
    ) -> Path | None:
        """Return the local zarr path for a quartet+nw, or None if absent."""
        return self._catalog.get_zarr_path(master, obsnum, subobsnum, scannum, nw)

    def has_kids_reduction(
        self,
        master: str,
        obsnum: int,
        subobsnum: int,
        scannum: int,
        nw: int,
    ) -> bool:
        """Return True if the zarr store for the given cell has KIDs reduction results."""
        zarr_path = self._catalog.get_zarr_path(master, obsnum, subobsnum, scannum, nw)
        if zarr_path is None:
            return False
        try:
            from tolteca_kids.pipeline import has_kids_reduction as _hkr
            return _hkr(zarr_path)
        except ImportError:
            return False

    def get_kids_find(
        self,
        master: str,
        obsnum: int,
        subobsnum: int,
        scannum: int,
        nw: int,
    ):
        """Return KidsFindZarrData for the given cell, or None if not available.

        Returns ``None`` if ``tolteca_kids`` is not installed or the zarr has
        no ``kids_find`` group.
        """
        zarr_path = self._catalog.get_zarr_path(master, obsnum, subobsnum, scannum, nw)
        if zarr_path is None:
            return None
        try:
            from tolteca_kids.pipeline import read_kids_find
            return read_kids_find(zarr_path)
        except ImportError:
            return None

    def get_sweep_check(
        self,
        master: str,
        obsnum: int,
        subobsnum: int,
        scannum: int,
        nw: int,
    ):
        """Return SweepCheckZarrData for the given cell, or None if not available."""
        zarr_path = self._catalog.get_zarr_path(master, obsnum, subobsnum, scannum, nw)
        if zarr_path is None:
            return None
        try:
            from tolteca_kids.pipeline import read_sweep_check
            return read_sweep_check(zarr_path)
        except ImportError:
            return None

    @functools.lru_cache(maxsize=128)
    def get_kids_diag_fig(self, zarr_path_str: str, fig_key: str):
        """Generate and cache a single KIDs diagnostic figure.

        Figures are computed lazily per-tab so each tab loads independently.
        The pipeline data (``dt``, ``kf_ctx``, ``cfg_kf``) is shared via
        :meth:`get_kids_diag_data` which is also LRU-cached.

        Parameters
        ----------
        zarr_path_str
            Absolute path to the zarr store.
        fig_key
            One of ``peaks``, ``peak_props``, ``d21_summary``, ``s21_summary``,
            ``det_summary``, ``matched``, ``matched_ref``.
        """
        data = self.get_kids_diag_data(zarr_path_str)
        if data is None:
            return None
        dt, kf_ctx, cfg_kf = data
        try:
            from tolteca_kids.kids_plot import (
                make_d21_summary_fig,
                make_det_summary_fig,
                make_matched_fig,
                make_peak_props_fig,
                make_peaks_fig,
                make_s21_summary_fig,
            )
            ctd = kf_ctx.data
            if fig_key == "peaks":
                return make_peaks_fig(dt, kf_ctx)
            if fig_key == "peak_props":
                return make_peak_props_fig(kf_ctx, cfg_kf)
            if fig_key == "d21_summary":
                return make_d21_summary_fig(kf_ctx, cfg_kf)
            if fig_key == "s21_summary":
                return make_s21_summary_fig(kf_ctx, cfg_kf)
            if fig_key == "det_summary":
                return make_det_summary_fig(kf_ctx, cfg_kf)
            if fig_key == "matched" and ctd.matched is not ...:
                return make_matched_fig(ctd.matched, "Chan")
            if fig_key == "matched_ref" and ctd.matched_ref is not ...:
                return make_matched_fig(
                    ctd.matched_ref, cfg_kf.match_ref.capitalize()
                )
            return None
        except Exception:
            import traceback
            traceback.print_exc()
            return None

    @functools.lru_cache(maxsize=32)
    def get_kids_diag_data(self, zarr_path_str: str):
        """Run the KIDs pipeline steps and return (dt, kf_ctx, cfg_kf).

        The result is LRU-cached per zarr path so repeated calls (tab switches,
        re-renders) do not re-run the pipeline.

        Returns ``None`` if ``tolteca_kids`` is not installed or the pipeline
        fails for this store.
        """
        from pathlib import Path as _Path

        zarr_path = _Path(zarr_path_str)
        try:
            from tolteca_kids.kids_find import KidsFind
            from tolteca_kids.pipeline import (
                KidsPipelineConfig,
                _build_datatree_from_zarr,
                open_zarr_dataset,
            )
            from tolteca_kids.sweep_check import SweepCheck
        except ImportError:
            return None

        try:
            ds = open_zarr_dataset(zarr_path)
            dt = _build_datatree_from_zarr(ds)
            cfg = KidsPipelineConfig()
            SweepCheck(cfg.sweep_check)(dt)
            KidsFind(cfg.kids_find)(dt)
            kf_ctx = KidsFind.get_context(dt)
            return dt, kf_ctx, cfg.kids_find
        except Exception:
            return None

    @functools.lru_cache(maxsize=256)
    def get_channel_summary(
        self,
        master: str,
        obsnum: int,
        subobsnum: int,
        scannum: int,
        nw: int,
    ) -> pl.DataFrame:
        """Return per-channel amplitude summary (LRU-cached).

        Loads mean I/Q amplitude across all sweep steps for every channel.
        Used by the Channel Summary tab.
        """
        ds = self.get_obs_data(master, obsnum, subobsnum, scannum, nw)
        ds_c = ds[["I", "Q"]].compute()
        import numpy as np
        amp = np.sqrt(ds_c["I"].values ** 2 + ds_c["Q"].values ** 2)
        mean_amp = amp.mean(axis=1)  # (n_chans,)
        tone_freq = ds_c["tone_freq"].values if "tone_freq" in ds_c else None
        data: dict = {
            "chan": list(range(len(mean_amp))),
            "mean_amp": mean_amp.tolist(),
        }
        if tone_freq is not None:
            data["tone_freq_hz"] = tone_freq.tolist()
        return pl.DataFrame(data)
