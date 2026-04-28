"""DataProdViewerPage — portal for browsing TolTEC data products.

Two tabs:
- **All Data Products**: unified time-ordered table (raw obs + cal groups +
  analysis products), with checkboxes for dp-type filtering.
- **Raw Obs**: selection-enabled AG Grid for opening quartets in SweepViewer.

Filter bar (shared):
- Date MultiSelect — multi-select of observation dates; defaults to the
  latest date present in the catalog.
- Master SegmentedControl — All / ICS / TCS.

Additional filter on Raw Obs tab:
- ObsNum range (min/max NumberInput).

Collection view
---------------
Each row in the All Data Products tab carries an ``assoc_key``
(``{master}-{obsnum}-{subobsnum}-{scannum}``), which is the quartet key of
the raw observation the product is derived from.  Clicking the **assoc.**
link in a row navigates to ``/?collection=<assoc_key>``, which re-loads the
portal showing only data products with that assoc_key (across all dates).
An ``← All`` breadcrumb clears the filter.

Architecture
------------
* Filter state lives in ``filter_store`` (single ``dcc.Store``).
* Selection state for Raw Obs lives in ``selected_store``
  (comma-separated quartet-key strings).
* Date options are populated once on page load via a ``dcc.Interval``
  with ``max_intervals=1``.
* ``dcc.Location(id="portal-url")`` tracks ``?collection=<key>`` URL state.

State format
------------
* ``filter_store.data``: dict with keys ``master``, ``dates``
  (list[str] of YYYY-MM-DD, or ``[]`` for all dates),
  ``obsnum_min``, ``obsnum_max``.
* ``selected_store.data``: comma-separated ``"{master}-{obsnum}-{subobsnum}-{scannum}"``
  strings.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlencode

import polars as pl
from dash import Input, Output, State, dcc, html, no_update
from dash_component_template import Template

import dash_ag_grid as dag
import dash_mantine_components as dmc

from tolteca_web.obs import ObsDataService

__all__ = ["DataProdViewerPage"]


# ── Constants ──────────────────────────────────────────────────────────────────

_ARRAY_GROUPS: dict[str, list[int]] = {
    "a1100": list(range(7)),
    "a1400": list(range(7, 11)),
    "a2000": list(range(11, 13)),
}
_NW_GROUPS = [list(range(7)), list(range(7, 11)), list(range(11, 13))]
_ARRAY_COLORS: dict[str, str] = {
    "a1100": "blue",
    "a1400": "teal",
    "a2000": "violet",
}
_KIND_COLORS: dict[str, str] = {
    "vnasweep": "teal",
    "targsweep": "blue",
    "tune": "violet",
    "timestream": "gray",
}
_KIND_COLOR_HEX: dict[str, str] = {
    "vnasweep": "#12b886",
    "targsweep": "#228be6",
    "tune": "#7950f2",
    "timestream": "#868e96",
}
_KIND_LABELS: dict[str, str] = {
    "vnasweep": "VNA",
    "targsweep": "Targ",
    "tune": "Tune",
    "timestream": "TS",
}
_ALL_KINDS = ["vnasweep", "targsweep", "tune"]

_DP_TYPE_LABELS: dict[str, str] = {
    "raw_obs": "Raw Obs",
    "cal_groups": "Cal Groups",
    "drivefit": "Drivefit",
    "focus": "Focus",
    "astig": "Astig",
    "oof": "OOF",
    "reduced_obs": "KIDs",
}
_DP_TYPE_COLORS: dict[str, str] = {
    "raw_obs": "blue",
    "cal_groups": "green",
    "drivefit": "orange",
    "focus": "pink",
    "astig": "grape",
    "oof": "indigo",
    "reduced_obs": "violet",
}
_ALL_DP_TYPES = list(_DP_TYPE_LABELS.keys())

# Kind label + color for analysis group rows (shown in KindChip)
_DP_ANALYSIS_KIND: dict[str, tuple[str, str]] = {
    "drivefit": ("DRV",  "#e8590c"),
    "focus":    ("FOC",  "#c2255c"),
    "astig":    ("AST",  "#9c36b5"),
    "oof":      ("OOF",  "#3b5bdb"),
}

_ANALYSIS_PROD_TYPES: dict[str, str] = {
    "drivefit": "dp_drivefit",
    "focus":    "dp_focus_group",
    "astig":    "dp_astig_group",
    "oof":      "dp_oof_group",
    "reduced_obs": "dp_reduced_obs",
}

_DEFAULT_FILTERS: dict = {
    "master": "all",
    "dates": [],      # [] = latest date (resolved at callback time); list = filter
    "obsnum_min": None,
    "obsnum_max": None,
}

# AG Grid column definitions for All Data Products tab
_ALL_DP_COLUMN_DEFS = [
    {
        "field": "uid",
        "headerName": "UID",
        "width": 215,
        "tooltipField": "uid",
        "cellStyle": {"fontFamily": "monospace", "fontSize": "12px", "fontWeight": "600"},
    },
    {"field": "dp_type", "headerName": "Type", "width": 95, "cellRenderer": "TypeChip"},
    {
        "field": "obs_datetime",
        "headerName": "obs_datetime",
        "width": 165,
        "cellStyle": {"fontSize": "12px", "color": "#868e96"},
    },
    {"field": "source_name", "headerName": "Source", "width": 120, "cellStyle": {"fontSize": "12px"}},
    {"field": "obs_goal", "headerName": "Goal", "width": 80, "cellStyle": {"fontSize": "12px", "color": "#868e96"}},
    {"field": "obs_pgm", "headerName": "Pgm", "width": 80, "cellStyle": {"fontSize": "12px", "color": "#868e96"}},
    {"field": "kind_label", "headerName": "Kind", "width": 90, "cellRenderer": "KindChip"},
    {"field": "n_items", "headerName": "N obs", "width": 65, "cellStyle": {"fontSize": "12px", "textAlign": "center"}},
    {"field": "obsnums_first", "headerName": "ObsNums", "width": 175, "cellRenderer": "ObsnumsCell"},
    # Action buttons: multiple viewer links per row
    {"field": "actions_json", "headerName": "Open", "width": 240, "sortable": False, "cellRenderer": "ActionsCell"},
    # Collection link: click to show all associated data products
    {"field": "assoc_key", "headerName": "", "width": 95, "sortable": False, "cellRenderer": "AssocLink"},
]

# AG Grid column definitions for Raw Obs tab
_RAW_OBS_COLUMN_DEFS = [
    {
        "field": "uid",
        "headerName": "UID",
        "width": 185,
        "cellStyle": {"fontFamily": "monospace", "fontSize": "12px", "fontWeight": "600"},
    },
    {"field": "obs_datetime", "headerName": "obs_datetime", "width": 165},
    {"field": "source_name", "headerName": "Source", "width": 115},
    {"field": "obs_goal", "headerName": "Goal", "width": 90},
    {"field": "obs_pgm", "headerName": "Pgm", "width": 90},
    {
        "field": "nws_json",
        "headerName": "Roaches",
        "width": 230,
        "headerTooltip": "Roach presence: 0-6 (a1100 · blue) · 7-10 (a1400 · teal) · 11-12 (a2000 · violet)",
        "sortable": False,
        "cellRenderer": "RoachRenderer",
    },
    {"field": "kinds", "headerName": "Kind", "width": 80, "cellStyle": {"fontSize": "12px"}},
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _quartet_key(master: str, obsnum: int, subobsnum: int, scannum: int) -> str:
    return f"{master}-{obsnum}-{subobsnum}-{scannum}"


def _raw_obs_uid(master: str, obsnum: int, subobsnum: int, scannum: int) -> str:
    """Canonical raw-obs uid: ``{master}-{obsnum}-{subobsnum}-{scannum}``."""
    return f"{master}-{obsnum}-{subobsnum}-{scannum}"


# Maps portal tab_val / dp_type to the short tag used in canonical DB names
_DP_TYPE_TAG: dict[str, str] = {
    "cal_groups": "cal",
    "drivefit": "drivefit",
    "focus": "focus",
    "astig": "astig",
    "oof": "oof",
    "reduced_obs": "reduced_obs",
}


def _group_uid(tag: str, obsnum_start: int, n: int) -> str:
    """Group display uid: ``{tag}_{obsnum_start}_g{n}``

    e.g. ``oof_153517_g4``, ``cal_153488_g23``, ``drivefit_152784_g7``

    Uses the short dp-type tag as prefix and first obsnum only (no verbose range).
    """
    return f"{tag}_{obsnum_start}_g{n}"


def _get_date_options(dates: list[str]) -> list[dict]:
    """Build newest-first MultiSelect option list from date strings."""
    unique = sorted({d[:10] for d in dates if d and len(d) >= 10}, reverse=True)
    return [{"label": d, "value": d} for d in unique]


def _aggregate_quartets(
    df: pl.DataFrame,
    svc: ObsDataService,
    dates: list[str] | None = None,
    check_zarr: bool = True,
) -> list[dict]:
    """Group catalog rows by quartet; aggregate NW/array availability.

    Parameters
    ----------
    df
        Raw catalog DataFrame.
    svc
        Data service (used for zarr-path existence checks).
    dates
        Optional list of YYYY-MM-DD strings to filter by; ``None``/``[]`` = all.
    check_zarr
        When ``False``, skip filesystem existence checks for zarr paths
        (faster; ``nws_with_zarr`` will equal ``nws_present``).
    """
    if df.is_empty():
        return []

    rows: list[dict] = []
    for (master, obsnum, subobsnum, scannum), group in df.group_by(
        ["master", "obsnum", "subobsnum", "scannum"],
        maintain_order=False,
    ):
        raw_dt = group["date_utc"][0] or ""
        obs_datetime = raw_dt.replace(" ", "T") if raw_dt else ""
        date_utc = obs_datetime[:10] if obs_datetime else ""

        if dates and date_utc not in dates:
            continue

        nws_present: set[int] = set()
        nws_with_zarr: set[int] = set()
        for nw_val in group["nw"].to_list():
            nws_present.add(int(nw_val))
            if check_zarr:
                g2 = group.filter(pl.col("nw") == nw_val)
                zarr_rel = g2["zarr_path"][0] if not g2.is_empty() else None
                if zarr_rel and (svc.cache_root / zarr_rel).exists():
                    nws_with_zarr.add(int(nw_val))
            else:
                nws_with_zarr.add(int(nw_val))

        data_kinds = sorted(set(group["data_kind"].to_list()))

        # Telescope metadata (same for all nws in a quartet — use first row)
        first = group.row(0, named=True)
        source_name = (first.get("source_name") or "") if "source_name" in group.columns else ""
        obs_goal = (first.get("obs_goal") or "") if "obs_goal" in group.columns else ""
        obs_pgm = (first.get("obs_pgm") or "") if "obs_pgm" in group.columns else ""

        rows.append({
            "qk": _quartet_key(master, obsnum, subobsnum, scannum),
            "uid": _raw_obs_uid(master, obsnum, subobsnum, scannum),
            "master": master,
            "obsnum": obsnum,
            "subobsnum": subobsnum,
            "scannum": scannum,
            "data_kinds": data_kinds,
            "date_utc": date_utc,
            "obs_datetime": obs_datetime,
            "nws_present": nws_present,
            "nws_with_zarr": nws_with_zarr,
            "source_name": source_name,
            "obs_goal": obs_goal,
            "obs_pgm": obs_pgm,
        })

    rows.sort(key=lambda r: (r["date_utc"], r["obsnum"]), reverse=True)
    return rows


def _filter_by_dates(items: list[dict], dates: list[str] | None) -> list[dict]:
    """Filter a list of dicts (must have ``date_utc`` key) by date."""
    if not dates:
        return items
    return [r for r in items if (r.get("date_utc") or "")[:10] in dates]


# ── Visual component helpers ────────────────────────────────────────────────────


# ── Unified "All" table builder ─────────────────────────────────────────────────


def _kind_row_data(kinds: list[str]) -> tuple[str, str, str | None]:
    """Return (kind_label, kind_color_hex, kind_tooltip) for a list of data kinds."""
    if not kinds:
        return "—", "#dee2e6", None
    if len(kinds) == 1:
        dk = kinds[0]
        return _KIND_LABELS.get(dk, dk), _KIND_COLOR_HEX.get(dk, "#868e96"), None
    label = f"{len(kinds)} kinds"
    tooltip = ", ".join(_KIND_LABELS.get(dk, dk) for dk in kinds)
    return label, "#868e96", tooltip


def _build_all_dp_row_data(
    raw_rows: list[dict],
    cal_groups: list[dict],
    analysis_groups: dict[str, list[dict]],
    dp_types_shown: list[str],
    obs_meta_lookup: dict[int, dict] | None = None,
) -> list[dict]:
    """Build AG Grid rowData for the All Data Products tab.

    Parameters
    ----------
    obs_meta_lookup
        Optional dict mapping ``obsnum`` → ``{source_name, obs_goal, obs_pgm}``.
        Used to populate telescope metadata for group rows.
    """
    _lookup = obs_meta_lookup or {}
    all_rows: list[dict] = []

    if "raw_obs" in dp_types_shown:
        for row in raw_rows:
            sweep_qs = urlencode({"quartets": row["qk"]})
            kind_label, kind_color_hex, kind_tooltip = _kind_row_data(row["data_kinds"])
            all_rows.append({
                "row_id": f"raw_obs:{row['uid']}",
                "uid": row["uid"],
                "dp_type": "raw_obs",
                "obs_datetime": row["obs_datetime"] or row["date_utc"] or "",
                "source_name": row["source_name"] or "",
                "obs_goal": row["obs_goal"] or "",
                "obs_pgm": row["obs_pgm"] or "",
                "kind_label": kind_label,
                "kind_color_hex": kind_color_hex,
                "kind_tooltip": kind_tooltip,
                "n_items": 1,
                "obsnums_first": row["obsnum"],
                "obsnums_count": 1,
                "obsnums_tooltip": None,
                "actions_json": json.dumps([
                    {"label": "Sweep →", "href": f"/sweep?{sweep_qs}", "color": "#228be6"},
                    {"label": "KIDs →", "href": f"/kids-diag?{sweep_qs}", "color": "#7048e8"},
                    {"label": "Tel →", "href": "/tel?" + urlencode({"quartet": row["qk"]}), "color": "#e67700"},
                ]),
                "assoc_key": row["qk"],
                "_sort": (row["date_utc"], row["obsnum"]),
            })

    if "cal_groups" in dp_types_shown:
        for grp in cal_groups:
            member_obsnums = sorted({
                int(qk.split("-")[1])
                for qk in grp["member_quartets"]
                if len(qk.split("-")) >= 2
            })
            qs = urlencode({"quartets": ",".join(grp["member_quartets"])})
            grp_master = grp.get("master", "ics")
            uid = _group_uid(_DP_TYPE_TAG["cal_groups"], grp["obsnum_start"], len(grp["member_quartets"]))
            kind_label, kind_color_hex, kind_tooltip = _kind_row_data(grp["data_kinds"])
            obs_count = len(member_obsnums)
            # Single-member cal groups can be associated with their raw obs
            if len(grp["member_quartets"]) == 1:
                assoc_key = grp["member_quartets"][0]
            else:
                assoc_key = None
            _cal_meta = _lookup.get(member_obsnums[0], {}) if member_obsnums else {}
            all_rows.append({
                "row_id": f"cal_groups:{uid}",
                "uid": uid,
                "dp_type": "cal_groups",
                "obs_datetime": grp["date_utc"] or "",
                "source_name": _cal_meta.get("source_name") or "",
                "obs_goal": _cal_meta.get("obs_goal") or "",
                "obs_pgm": _cal_meta.get("obs_pgm") or "",
                "kind_label": kind_label,
                "kind_color_hex": kind_color_hex,
                "kind_tooltip": kind_tooltip,
                "n_items": len(grp["member_quartets"]),
                "obsnums_first": member_obsnums[0] if member_obsnums else None,
                "obsnums_count": obs_count,
                "obsnums_tooltip": ", ".join(str(o) for o in member_obsnums) if obs_count > 1 else None,
                "actions_json": json.dumps([
                    {"label": "Sweep →", "href": f"/sweep?{qs}", "color": "#228be6"},
                ]),
                "assoc_key": assoc_key,
                "_sort": (grp["date_utc"], grp["obsnum_start"]),
            })

    for tab_val, groups in analysis_groups.items():
        if tab_val not in dp_types_shown:
            continue
        for grp in groups:
            obsnum_start = grp["obsnum_start"]
            grp_master = grp.get("master", "ics")
            uid = _group_uid(_DP_TYPE_TAG.get(tab_val, tab_val), obsnum_start, grp.get("n_items", 0))
            obsnum_end = grp.get("obsnum_end", obsnum_start)
            obsnums = list(range(obsnum_start, obsnum_end + 1)) if obsnum_end else [obsnum_start]
            obs_count = len(obsnums)
            _grp_meta = _lookup.get(obsnum_start, {})
            if tab_val == "reduced_obs":
                sub = grp.get("subobsnum", 0)
                scan = grp.get("scannum", 0)
                quartet_qs = urlencode({"quartets": f"{grp_master}-{obsnum_start}-{sub}-{scan}"})
                kind_label, kind_color_hex, kind_tooltip = "KIDS", "#7048e8", None
                actions_json = json.dumps([
                    {"label": "KIDs →", "href": f"/kids-diag?{quartet_qs}", "color": "#7048e8"},
                ])
                assoc_key = f"{grp_master}-{obsnum_start}-{sub}-{scan}"
            else:
                _ak, _ac = _DP_ANALYSIS_KIND.get(tab_val, ("—", "#dee2e6"))
                kind_label, kind_color_hex = _ak, _ac
                n_items_val = grp.get("n_items", 0)
                kind_tooltip = f"{_DP_TYPE_LABELS.get(tab_val, tab_val)} group, {n_items_val} obs"
                # Link to portal with obsnum range filter to show member raw obs
                obs_qs = urlencode({"obsnum_min": obsnum_start, "obsnum_max": obsnum_end, "master": grp_master})
                tel_qs = urlencode({"quartet": f"{grp_master}-{obsnum_start}-0-1"})
                actions_json = json.dumps([
                    {"label": "View obs →", "href": f"/?{obs_qs}", "color": "#2f9e44"},
                    {"label": "Tel →", "href": f"/tel?{tel_qs}", "color": "#e67700"},
                ])
                assoc_key = None
            all_rows.append({
                "row_id": f"{tab_val}:{uid}",
                "uid": uid,
                "dp_type": tab_val,
                "obs_datetime": grp["date_utc"] or "",
                "source_name": _grp_meta.get("source_name") or "",
                "obs_goal": _grp_meta.get("obs_goal") or "",
                "obs_pgm": _grp_meta.get("obs_pgm") or "",
                "kind_label": kind_label,
                "kind_color_hex": kind_color_hex,
                "kind_tooltip": kind_tooltip,
                "n_items": grp.get("n_items", 0),
                "obsnums_first": obsnum_start,
                "obsnums_count": obs_count,
                "obsnums_tooltip": ", ".join(str(o) for o in obsnums) if obs_count > 1 else None,
                "actions_json": actions_json,
                "assoc_key": assoc_key,
                "_sort": (grp["date_utc"], obsnum_start),
            })

    all_rows.sort(key=lambda r: r["_sort"], reverse=True)
    for r in all_rows:
        del r["_sort"]
    return all_rows


# ── Raw Obs AG Grid row builder ─────────────────────────────────────────────────


def _build_raw_obs_row_data(rows: list[dict]) -> list[dict]:
    """Convert aggregated quartet rows to AG Grid rowData (Raw Obs tab)."""
    result = []
    for row in rows:
        kinds_str = " · ".join(_KIND_LABELS.get(dk, dk) for dk in row["data_kinds"])
        result.append({
            "uid": row["uid"],
            "qk": row["qk"],
            "master": row["master"].upper(),
            "obsnum": row["obsnum"],
            "obs_datetime": row["obs_datetime"] or row["date_utc"] or "—",
            "source_name": row["source_name"] or "—",
            "obs_goal": row["obs_goal"] or "—",
            "obs_pgm": row["obs_pgm"] or "—",
            "nws_json": json.dumps({
                "present": sorted(row["nws_present"]),
                "zarr": sorted(row.get("nws_with_zarr", set())),
            }),
            "kinds": kinds_str,
        })
    return result


def _tab_label(label: str, n: int) -> list:
    """Tab label + count badge."""
    return dmc.Group(
        [
            label,
            dmc.Badge(
                str(n), size="xs",
                variant="filled" if n > 0 else "outline",
                color="blue" if n > 0 else "gray",
            ),
        ],
        gap=4, wrap="nowrap",
    )


# ── Main Template ──────────────────────────────────────────────────────────────


class DataProdViewerPage(Template):
    """Portal page for browsing and selecting TolTEC data products.

    Parameters
    ----------
    data_service
        :class:`~tolteca_web.obs.ObsDataService` providing catalog and zarr
        data access (shared with :class:`~tolteca_web.sweep.SweepViewerPage`).
    """

    def __init__(self, data_service: ObsDataService) -> None:
        super().__init__()
        self._svc = data_service

        provider = self.child[dmc.MantineProvider]()
        root = provider.child[dmc.Stack](gap=0, style={"minHeight": "100vh"})

        # ── Stores & URL ──────────────────────────────────────────────────
        self._location = root.child[dcc.Location](id="portal-url", refresh=False)
        self._filter_store = root.child[dcc.Store](data=dict(_DEFAULT_FILTERS))
        self._selected_store = root.child[dcc.Store](data="")
        # One-shot interval for date-options initialization
        self._init_interval = root.child[dcc.Interval](
            interval=200, max_intervals=1, n_intervals=0,
        )

        # ── Header ────────────────────────────────────────────────────────
        header_box = root.child[dmc.Paper](
            withBorder=True, shadow="none",
            style={"borderLeft": "none", "borderRight": "none", "borderTop": "none"},
        )
        hdr = header_box.child[dmc.Group](px="md", py="xs", justify="space-between")
        hdr.child[dmc.Title](children="TolTEC Data Products", order=4)
        hdr.child[dmc.Anchor](
            children="→ Sweep Viewer", href="/sweep", size="sm", c="blue",
        )

        container = root.child[html.Div](style={"padding": "8px 16px 0", "width": "100%"})

        # ── Shared filter bar ─────────────────────────────────────────────
        filter_paper = container.child[dmc.Paper](
            withBorder=True, p="sm", mb="sm", radius="sm",
        )
        filter_row = filter_paper.child[dmc.Group](gap="lg", align="flex-end", wrap="wrap")

        # Date multi-select (populated by init callback)
        date_col = filter_row.child[dmc.Stack](gap=2)
        date_col.child[dmc.Text]("Date", size="xs", c="dimmed", fw=600)
        self._date_select = date_col.child[dmc.MultiSelect](
            data=[],
            value=[],
            placeholder="All dates",
            searchable=True,
            clearable=True,
            w=200,
            size="xs",
        )

        # Master selector
        master_col = filter_row.child[dmc.Stack](gap=2)
        master_col.child[dmc.Text]("Master", size="xs", c="dimmed", fw=600)
        self._master_ctrl = master_col.child[dmc.SegmentedControl](
            data=[
                {"label": "All", "value": "all"},
                {"label": "ICS", "value": "ics"},
                {"label": "TCS", "value": "tcs"},
            ],
            value="all",
            size="xs",
        )

        # ── Collection breadcrumb bar (hidden when no collection active) ──
        self._collection_bar = container.child[dmc.Paper](
            withBorder=True, p="xs", mb="xs", radius="sm",
            style={"display": "none", "background": "#e7f5ff", "borderColor": "#74c0fc"},
        )
        bar_row = self._collection_bar.child[dmc.Group](gap="sm", align="center")
        self._back_btn = bar_row.child[dmc.Anchor](
            "← All", href="/", size="sm", fw=700, c="blue",
        )
        bar_row.child[dmc.Text]("·", size="sm", c="dimmed")
        self._collection_text = bar_row.child[dmc.Text](
            "", size="sm", c="dark",
        )

        # ── Tabs ──────────────────────────────────────────────────────────
        tabs = container.child[dmc.Tabs](value="all", mt="xs", style={"width": "100%"})
        tabs_list = tabs.child[dmc.TabsList](mb="xs")
        self._tab_all = tabs_list.child[dmc.TabsTab](
            children="All Data Products", value="all",
        )
        self._tab_raw = tabs_list.child[dmc.TabsTab](
            children="Raw Obs", value="raw_obs",
        )

        # ── All Data Products panel ───────────────────────────────────────
        all_panel = tabs.child[dmc.TabsPanel](value="all", style={"width": "100%"})

        # Type checkboxes
        type_row = all_panel.child[dmc.Group](gap="md", mb="xs", wrap="wrap")
        type_row.child[dmc.Text]("Show:", size="xs", c="dimmed", fw=600)
        self._dp_type_checks = type_row.child[dmc.CheckboxGroup](
            value=list(_ALL_DP_TYPES),
            children=dmc.Group(
                [
                    dmc.Checkbox(
                        label=dmc.Badge(
                            _DP_TYPE_LABELS[t],
                            color=_DP_TYPE_COLORS[t],
                            variant="light",
                            size="xs",
                        ),
                        value=t,
                        size="xs",
                    )
                    for t in _ALL_DP_TYPES
                ],
                gap="sm",
            ),
        )

        all_stats_row = all_panel.child[dmc.Group](
            justify="space-between", align="center", mb="xs",
        )
        self._all_stats_text = all_stats_row.child[dmc.Text](
            "Loading…", size="sm", c="dimmed",
        )

        self._all_grid = all_panel.child[dag.AgGrid](
            rowData=[],
            columnDefs=_ALL_DP_COLUMN_DEFS,
            getRowId="params.data.row_id",
            defaultColDef={"resizable": True, "sortable": True},
            dashGridOptions={"rowHeight": 34, "headerHeight": 36},
            style={"height": "calc(100vh - 260px)", "width": "100%"},
            className="ag-theme-alpine",
        )

        # ── Raw Obs panel ─────────────────────────────────────────────────
        raw_panel = tabs.child[dmc.TabsPanel](value="raw_obs", style={"width": "100%"})

        # ObsNum range (Raw Obs only)
        obs_filter_row = raw_panel.child[dmc.Group](gap="lg", align="flex-end", mb="xs", wrap="wrap")
        obsnum_col = obs_filter_row.child[dmc.Stack](gap=2)
        obsnum_col.child[dmc.Text]("ObsNum range", size="xs", c="dimmed", fw=600)
        obsnum_row = obsnum_col.child[dmc.Group](gap="xs", align="center")
        self._obsnum_min = obsnum_row.child[dmc.NumberInput](
            placeholder="min", w=100, size="xs", min=0,
        )
        obsnum_row.child[dmc.Text]("–", size="xs", c="dimmed")
        self._obsnum_max = obsnum_row.child[dmc.NumberInput](
            placeholder="max", w=100, size="xs", min=0,
        )

        # Action row: stats + selection badge + open link
        action_row = raw_panel.child[dmc.Group](
            justify="space-between", align="center", mb="xs",
        )
        self._stats_text = action_row.child[dmc.Text](
            "Loading…", size="sm", c="dimmed",
        )
        right_actions = action_row.child[dmc.Group](gap="sm", align="center")
        self._sel_badge = right_actions.child[dmc.Badge](
            "0 selected", color="blue", variant="light", size="md",
        )
        self._open_link = right_actions.child[dmc.Anchor](
            "Open in SweepViewer →",
            href="/sweep", size="sm", fw=600,
            style={"color": "#868e96"},
        )

        self._grid = raw_panel.child[dag.AgGrid](
            rowData=[],
            columnDefs=_RAW_OBS_COLUMN_DEFS,
            getRowId="params.data.uid",
            defaultColDef={"resizable": True, "sortable": True},
            dashGridOptions={
                "rowHeight": 34,
                "headerHeight": 36,
                "rowSelection": "multiple",
            },
            style={"height": "calc(100vh - 370px)", "width": "100%"},
            className="ag-theme-alpine",
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def setup_callbacks(self, app) -> None:
        """Register all Dash callbacks."""

        svc = self._svc

        # ── I1: one-shot interval → populate date MultiSelect ──────────────
        @app.callback(
            Output(self._date_select(), "data"),
            Output(self._date_select(), "value"),
            Input(self._init_interval(), "n_intervals"),
        )
        def _init_dates(n: int | None) -> tuple:
            """Load available dates from catalog; default to the latest."""
            df = svc.get_obs_catalog()
            if df.is_empty() or "date_utc" not in df.columns:
                return [], []
            options = _get_date_options(df["date_utc"].drop_nulls().to_list())
            latest = [options[0]["value"]] if options else []
            return options, latest

        # ── F1: filter controls → filter_store ────────────────────────────
        @app.callback(
            Output(self._filter_store(), "data"),
            Input(self._master_ctrl(), "value"),
            Input(self._date_select(), "value"),
            Input(self._obsnum_min(), "value"),
            Input(self._obsnum_max(), "value"),
        )
        def _update_filters(
            master: str,
            dates: list[str] | None,
            obsnum_min: int | None,
            obsnum_max: int | None,
        ) -> dict:
            return {
                "master": master or "all",
                "dates": dates or [],
                "obsnum_min": obsnum_min,
                "obsnum_max": obsnum_max,
            }

        # ── C1: URL search → collection breadcrumb bar (clientside) ────────
        app.clientside_callback(
            """function(search) {
                var qs = new URLSearchParams((search || '').replace(/^\\?/, ''));
                var col = qs.get('collection');
                var oMin = qs.get('obsnum_min');
                var oMax = qs.get('obsnum_max');
                if (col) {
                    return [{display: '', background: '#e7f5ff', borderColor: '#74c0fc',
                             padding: '6px 10px', borderRadius: '4px', marginBottom: '8px',
                             border: '1px solid #74c0fc'},
                            'Showing data products associated with: ' + col];
                }
                if (oMin || oMax) {
                    var label = 'Showing raw obs in obsnum range ' + (oMin || '0') + '–' + (oMax || '∞');
                    return [{display: '', background: '#ebfbee', borderColor: '#69db7c',
                             padding: '6px 10px', borderRadius: '4px', marginBottom: '8px',
                             border: '1px solid #69db7c'},
                            label];
                }
                return [{display: 'none'}, ''];
            }""",
            Output(self._collection_bar(), "style"),
            Output(self._collection_text(), "children"),
            Input(self._location(), "search"),
        )

        # ── T_all: filter_store + type-checks + URL → All grid + stats ─────
        @app.callback(
            Output(self._all_grid(), "rowData"),
            Output(self._all_stats_text(), "children"),
            Input(self._filter_store(), "data"),
            Input(self._dp_type_checks(), "value"),
            Input(self._location(), "search"),
        )
        def _update_all_table(
            filters: dict | None,
            dp_types: list[str] | None,
            search: str | None,
        ) -> tuple:
            f = filters or dict(_DEFAULT_FILTERS)
            master_val: str = f.get("master", "all")
            dates: list[str] = f.get("dates") or []
            master = master_val if master_val != "all" else None

            # Extract params from URL
            qs = parse_qs((search or "").lstrip("?"))
            collection: str | None = qs.get("collection", [None])[0]
            # URL-based obsnum range filter (used by analysis group "View obs" links)
            url_obsnum_min: int | None = int(qs["obsnum_min"][0]) if "obsnum_min" in qs else None
            url_obsnum_max: int | None = int(qs["obsnum_max"][0]) if "obsnum_max" in qs else None
            url_master: str | None = qs.get("master", [None])[0]

            if collection:
                # Collection view: ignore date filter, show all dp types
                dates = []
                shown = list(_ALL_DP_TYPES)
            elif url_obsnum_min is not None or url_obsnum_max is not None:
                # Obsnum range view: show only raw_obs in range (from analysis group nav)
                dates = []
                shown = ["raw_obs"]
                if url_master and url_master != "all":
                    master = url_master
            else:
                shown = dp_types or list(_ALL_DP_TYPES)

            obsnum_range: tuple[int, int] | None = None
            if url_obsnum_min is not None or url_obsnum_max is not None:
                lo = url_obsnum_min if url_obsnum_min is not None else 0
                hi = url_obsnum_max if url_obsnum_max is not None else 999_999
                obsnum_range = (lo, hi)

            df = svc.get_obs_catalog(master=master, obsnum_range=obsnum_range)
            raw_rows = _aggregate_quartets(df, svc, dates=dates if dates else None, check_zarr=False)

            # Build obsnum → telescope meta lookup for group rows (source/goal/pgm)
            obs_meta_lookup: dict[int, dict] = {}
            if not df.is_empty():
                _meta_cols = [c for c in ("obsnum", "source_name", "obs_goal", "obs_pgm") if c in df.columns]
                if "obsnum" in _meta_cols:
                    for _row in df.select(_meta_cols).unique("obsnum").iter_rows(named=True):
                        _onum = _row.get("obsnum")
                        if _onum is not None:
                            obs_meta_lookup[int(_onum)] = {
                                "source_name": _row.get("source_name") or "",
                                "obs_goal": _row.get("obs_goal") or "",
                                "obs_pgm": _row.get("obs_pgm") or "",
                            }

            all_cal = svc.get_cal_groups(master=master)
            cal_groups = _filter_by_dates(all_cal, dates if dates else None)

            analysis_groups: dict[str, list[dict]] = {}
            for tab_val, prod_type in _ANALYSIS_PROD_TYPES.items():
                if tab_val == "reduced_obs":
                    all_ag = svc.get_reduced_obs_groups(master=master)
                else:
                    all_ag = svc.get_analysis_groups(
                        prod_type=prod_type,
                        master=master,
                        obsnum_range=obsnum_range,
                    )
                analysis_groups[tab_val] = _filter_by_dates(
                    all_ag, dates if dates else None
                )

            row_data = _build_all_dp_row_data(
                raw_rows, cal_groups, analysis_groups, shown,
                obs_meta_lookup=obs_meta_lookup,
            )

            if collection:
                row_data = [r for r in row_data if r.get("assoc_key") == collection]
                total = len(row_data)
                stats = f"{total} data product{'s' if total != 1 else ''} for {collection}"
            elif obsnum_range:
                total = len(row_data)
                lo, hi = obsnum_range
                stats = f"{total} observation{'s' if total != 1 else ''} in obsnum {lo}–{hi}"
            else:
                total = len(row_data)
                stats = f"{total} data product{'s' if total != 1 else ''}"

            return row_data, stats

        # ── T_raw: filter_store → Raw Obs grid + stats ─────────────────────
        @app.callback(
            Output(self._grid(), "rowData"),
            Output(self._stats_text(), "children"),
            Input(self._filter_store(), "data"),
        )
        def _update_raw_table(filters: dict | None) -> tuple:
            f = filters or dict(_DEFAULT_FILTERS)
            master_val: str = f.get("master", "all")
            dates: list[str] = f.get("dates") or []
            obsnum_min: int | None = f.get("obsnum_min")
            obsnum_max: int | None = f.get("obsnum_max")

            obsnum_range: tuple[int, int] | None = None
            if obsnum_min is not None or obsnum_max is not None:
                lo = int(obsnum_min) if obsnum_min is not None else 0
                hi = int(obsnum_max) if obsnum_max is not None else 999_999
                obsnum_range = (lo, hi)

            df = svc.get_obs_catalog(
                master=master_val if master_val != "all" else None,
                obsnum_range=obsnum_range,
            )
            rows = _aggregate_quartets(df, svc, dates=dates if dates else None)
            total = len(rows)
            stats = f"{total} observation{'s' if total != 1 else ''}"
            return _build_raw_obs_row_data(rows), stats

        # ── S1: AG Grid row selection → selected_store (clientside) ────────
        app.clientside_callback(
            """function(selectedRows) {
                if (!selectedRows || !selectedRows.length) return "";
                return selectedRows.map(function(r) { return r.qk; }).sort().join(",");
            }""",
            Output(self._selected_store(), "data"),
            Input(self._grid(), "selectedRows"),
        )

        # ── A1: selected_store → badge text + open-link href/style ─────────
        app.clientside_callback(
            """function(selectedStr) {
                var parts = (selectedStr || "").split(",")
                    .map(s => s.trim()).filter(s => s.length > 0);
                var n = parts.length;
                var badge = n > 0 ? n + " selected" : "0 selected";
                if (n > 0) {
                    var qs = new URLSearchParams({quartets: parts.sort().join(",")});
                    var href = "/sweep?" + qs.toString();
                    return [badge, href, {"color": "#1c7ed6"}, "blue"];
                }
                return [badge, "/sweep", {"color": "#868e96"}, "gray"];
            }""",
            Output(self._sel_badge(), "children"),
            Output(self._open_link(), "href"),
            Output(self._open_link(), "style"),
            Output(self._sel_badge(), "color"),
            Input(self._selected_store(), "data"),
        )
