"""Observation catalog backends for TolTEC web viewer.

Defines the :class:`ObsCatalogBackend` Protocol and the prototype
:class:`ParquetCatalogBackend` that reads from a local ``catalog.parquet``
file built by ``ingest_obs.py``.
"""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import polars as pl

__all__ = ["ObsCatalogBackend", "ParquetCatalogBackend"]

# Catalog column schema (subset used for queries)
_CATALOG_SCHEMA = {
    "master": pl.Utf8,
    "obsnum": pl.Int32,
    "subobsnum": pl.Int16,
    "scannum": pl.Int16,
    "nw": pl.Int8,
    "array_name": pl.Utf8,
    "data_kind": pl.Utf8,
    "date_utc": pl.Utf8,
    "source_name": pl.Utf8,
    "obs_goal": pl.Utf8,
    "obs_pgm": pl.Utf8,
    "n_chans": pl.Int32,
    "n_samples": pl.Int32,
    "lo_center_freq_hz": pl.Float64,
    "drive_atten_db": pl.Float32,
    "sense_atten_db": pl.Float32,
    "zarr_path": pl.Utf8,
    "nc_path": pl.Utf8,
}

# Array name by nw index
_NW_TO_ARRAY = {
    **{nw: "a1100" for nw in range(7)},    # nw 0-6  → 1.1mm array
    **{nw: "a1400" for nw in range(7, 11)}, # nw 7-10 → 1.4mm array
    **{nw: "a2000" for nw in range(11, 13)}, # nw 11-12 → 2.0mm array
}

# nw lists per array (inverse of _NW_TO_ARRAY)
_ARRAY_GROUPS: dict[str, list[int]] = {
    "a1100": list(range(7)),
    "a1400": list(range(7, 11)),
    "a2000": list(range(11, 13)),
}


@runtime_checkable
class ObsCatalogBackend(Protocol):
    """Protocol for observation catalog backends.

    Covers all quartet member files: KIDs sweep + timestream data.
    Prototype uses parquet; production uses tolteca_db DuckDB.
    """

    def query(
        self,
        data_kinds: list[str] | None = None,
        nw_list: list[int] | None = None,
        obsnum_range: tuple[int, int] | None = None,
        master: str | None = None,
    ) -> pl.DataFrame:
        """Return catalog rows matching the given filters."""
        ...

    def get_zarr_path(
        self,
        master: str,
        obsnum: int,
        subobsnum: int,
        scannum: int,
        nw: int,
    ) -> Path | None:
        """Return the zarr store path for the given quartet+nw, or None."""
        ...


class ParquetCatalogBackend:
    """Prototype catalog backend backed by a local ``catalog.parquet`` file.

    The catalog is read on first access and cached for ``ttl_seconds``
    (default 60 s); subsequent calls reuse the cached copy.

    Parameters
    ----------
    catalog_path
        Path to the ``catalog.parquet`` file.
    cache_root
        Root of the zarr cache directory.  ``zarr_path`` column values are
        relative to this directory.
    ttl_seconds
        How long to cache the catalog in memory before re-reading (seconds).
    """

    def __init__(
        self,
        catalog_path: Path,
        cache_root: Path,
        ttl_seconds: float = 60.0,
    ) -> None:
        self._catalog_path = Path(catalog_path)
        self._cache_root = Path(cache_root)
        self._ttl = ttl_seconds
        self._df: pl.DataFrame | None = None
        self._loaded_at: float = 0.0

    # ── Internal ──────────────────────────────────────────────────────────

    def _load(self) -> pl.DataFrame:
        now = time.monotonic()
        if self._df is None or (now - self._loaded_at) > self._ttl:
            if self._catalog_path.exists():
                self._df = pl.read_parquet(self._catalog_path)
            else:
                # Return an empty DataFrame with the expected schema
                self._df = pl.DataFrame({col: pl.Series(col, [], dtype=dtype)
                                         for col, dtype in _CATALOG_SCHEMA.items()})
            self._loaded_at = now
        return self._df

    # ── ObsCatalogBackend protocol ─────────────────────────────────────────

    def query(
        self,
        data_kinds: list[str] | None = None,
        nw_list: list[int] | None = None,
        obsnum_range: tuple[int, int] | None = None,
        master: str | None = None,
    ) -> pl.DataFrame:
        """Return catalog rows matching the given filters."""
        df = self._load()
        if df.is_empty():
            return df
        masks = []
        if data_kinds is not None:
            masks.append(pl.col("data_kind").is_in(data_kinds))
        if nw_list is not None:
            masks.append(pl.col("nw").is_in(nw_list))
        if obsnum_range is not None:
            lo, hi = obsnum_range
            masks.append(pl.col("obsnum").is_between(lo, hi))
        if master is not None:
            masks.append(pl.col("master") == master)
        if masks:
            expr = functools.reduce(lambda a, b: a & b, masks)
            df = df.filter(expr)
        return df

    def get_zarr_path(
        self,
        master: str,
        obsnum: int,
        subobsnum: int,
        scannum: int,
        nw: int,
    ) -> Path | None:
        """Return the zarr store path for the given quartet+nw, or None."""
        df = self._load()
        if df.is_empty():
            return None
        row = df.filter(
            (pl.col("master") == master)
            & (pl.col("obsnum") == obsnum)
            & (pl.col("subobsnum") == subobsnum)
            & (pl.col("scannum") == scannum)
            & (pl.col("nw") == nw)
        )
        if row.is_empty() or row["zarr_path"][0] is None:
            return None
        return self._cache_root / row["zarr_path"][0]

    def query_cal_groups(
        self,
        obsnum_range: tuple[int, int] | None = None,
        master: str | None = None,
    ) -> list[dict]:
        """Infer calibration groups from consecutive sweep observations.

        A calibration group starts at each VnaSweep and collects the
        following TargetSweep/Tune observations (ordered by obsnum,
        subobsnum, scannum) until the next VnaSweep.  Groups with only
        a single member (lone VNA with no following sweeps) are excluded.

        Parameters
        ----------
        obsnum_range
            Optional (min, max) obsnum filter applied before grouping.
        master
            Optional master filter (``"ics"`` or ``"tcs"``).

        Returns
        -------
        list[dict]
            Newest-first list of cal-group dicts, each containing:
            ``group_key`` (str), ``member_quartets`` (list[str]),
            ``data_kinds`` (list[str]), ``date_utc`` (str),
            ``obsnum_start`` (int), ``obsnum_end`` (int),
            ``nws_with_zarr`` (set[int]),
            ``array_counts`` (dict[str, tuple[int, int]]).
        """
        df = self.query(
            data_kinds=["vnasweep", "targsweep", "tune"],
            master=master,
            obsnum_range=obsnum_range,
        )
        if df.is_empty():
            return []

        # One representative row per quartet (first date/kind per group)
        unique_q = (
            df.group_by(["master", "obsnum", "subobsnum", "scannum"])
            .agg([
                pl.col("data_kind").first(),
                pl.col("date_utc").first(),
            ])
            .sort(["master", "obsnum", "subobsnum", "scannum"])
        )

        # Split into groups: each VnaSweep starts a new group
        raw_groups: list[list[dict]] = []
        current: list[dict] = []
        for row in unique_q.iter_rows(named=True):
            if row["data_kind"] == "vnasweep":
                if current:
                    raw_groups.append(current)
                current = [row]
            elif current:
                current.append(row)
        if current:
            raw_groups.append(current)

        # Require at least VNA + one other sweep
        raw_groups = [g for g in raw_groups if len(g) >= 2]

        result: list[dict] = []
        for g in raw_groups:
            member_quartets = [
                f"{r['master']}-{r['obsnum']}-{r['subobsnum']}-{r['scannum']}"
                for r in g
            ]
            data_kinds = sorted({r["data_kind"] for r in g})
            date_utc = g[0]["date_utc"] or ""
            if date_utc and len(date_utc) > 10:
                date_utc = date_utc[:10]

            # Zarr availability: union over all member quartets
            nws_with_zarr: set[int] = set()
            for r in g:
                member_rows = df.filter(
                    (pl.col("master") == r["master"])
                    & (pl.col("obsnum") == r["obsnum"])
                    & (pl.col("subobsnum") == r["subobsnum"])
                    & (pl.col("scannum") == r["scannum"])
                )
                for nw_val in member_rows["nw"].to_list():
                    nw_row = member_rows.filter(pl.col("nw") == nw_val)
                    zarr_rel = nw_row["zarr_path"][0] if not nw_row.is_empty() else None
                    if zarr_rel and (self._cache_root / zarr_rel).exists():
                        nws_with_zarr.add(int(nw_val))

            array_counts: dict[str, tuple[int, int]] = {
                arr: (
                    sum(1 for nw in nw_list if nw in nws_with_zarr),
                    len(nw_list),
                )
                for arr, nw_list in _ARRAY_GROUPS.items()
            }

            result.append({
                "group_key": member_quartets[0],  # keyed to the VNA quartet
                "member_quartets": member_quartets,
                "data_kinds": data_kinds,
                "date_utc": date_utc,
                "obsnum_start": g[0]["obsnum"],
                "obsnum_end": g[-1]["obsnum"],
                "master": g[0]["master"],
                "nws_with_zarr": nws_with_zarr,
                "array_counts": array_counts,
            })

        result.sort(key=lambda r: (r["date_utc"], r["obsnum_start"]), reverse=True)
        return result
