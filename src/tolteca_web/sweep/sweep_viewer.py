"""SweepViewerPage — Dash template for interactive TolTEC sweep inspection.

Displays S21 (dB) as a grid of (quartet × nw) panels.

Axes
----
* **Rows** — quartets: ``{master}-{obsnum}-{subobsnum}-{scannum}``
  (e.g. ``ics-18596-0-0``).
* **Columns** — roach/network index (nw), 0–12.
* **Cells** — full obs-file spec: ``{master}-{obsnum}-{subobsnum}-{scannum}-{nw}``
  (e.g. ``ics-18596-0-0-0``).

Interaction
-----------
1. User types quartet strings into the TagsInput; each becomes a matrix row.
2. A Plotly heatmap shows the matrix: gray = not in catalog, light-blue =
   available, solid-blue = selected/enabled.
3. Clicking a non-gray cell toggles it between enabled/disabled.
4. Enabled cells drive the S21 subplot grid below.

State (URL params)
------------------
* ``quartets`` — comma-separated quartet specs.
* ``selected`` — comma-separated full obs-file specs (selected cells).

Callbacks
---------
M1.  (quartets, selected) → matrix heatmap figure.
M2.  matrix.clickData → toggle cell in selected (state-store patch).
A.   selected → pager n_items + channel-count text.
B.   (selected, page_info, prev_hist) → subplot grid + status + hist.
C.   selected → obs-info badge row.
D.   fetch_hist → zarr stats text.
"""

from __future__ import annotations

import re
import time

import numpy as np
from plotly.subplots import make_subplots

from dash import Input, Output, State, dcc
from dash_component_template import Template
from pydantic import BaseModel
from _plotly_utils.utils import to_typed_array_spec

import dash_ag_grid as dag
import dash_mantine_components as dmc

from tolteca_web.common import Pager, UrlStateManager
from tolteca_web.obs import ObsDataService
from tolteca_web.sweep.zarr_model import ZarrSweepDataset

__all__ = ["SweepViewerPage"]


# ── State ──────────────────────────────────────────────────────────────────────


class SweepViewerState(BaseModel):
    """URL-synced page state."""

    # Default shows a multi-row matrix: ics multi-nw vnasweep + ics targsweep
    # + ics vnasweep single-nw + two tcs entries (vnasweep / targsweep)
    quartets: str = (
        "ics-18596-32-0,"
        "ics-18252-2-0,"
        "ics-18306-0-0,"
        "tcs-113515-0-1,"
        "tcs-113516-0-1"
    )
    selected: str = ""
    # Plot mode: "s21" (dB lines vs freq) or "iq" (I vs Q scatter)
    plot_mode: str = "s21"


# ── Constants ──────────────────────────────────────────────────────────────────

_CHANS_PER_PAGE = 30
_NWS = list(range(13))

_DATA_KIND_COLORS: dict[str, str] = {
    "targsweep": "blue",
    "vnasweep": "teal",
    "tune": "violet",
}

# AG Grid column defs for the nw-vs-quartet selection matrix
_NW_CELL_STYLE = {
    "function": (
        "params.value === 0 ? {backgroundColor:'#f1f3f5',cursor:'default',border:'1px solid rgba(0,0,0,0.07)',borderRadius:'2px'} :"
        "params.value === 1 ? {backgroundColor:'#ffd43b',cursor:'default',border:'1px solid rgba(0,0,0,0.07)',borderRadius:'2px'} :"
        "params.value === 2 ? {backgroundColor:'#a5d8ff',cursor:'pointer',border:'1px solid rgba(0,0,0,0.07)',borderRadius:'2px'} :"
        "{backgroundColor:'#1c7ed6',cursor:'pointer',border:'2px solid #1971c2',borderRadius:'2px'}"
    )
}
_NW_TOOLTIP = {
    "function": (
        "params.value === 0 ? 'absent' :"
        "params.value === 1 ? 'catalogued (no zarr)' :"
        "params.value === 2 ? 'zarr available' : 'selected'"
    )
}
_MATRIX_COL_DEFS: list[dict] = [
    {
        "field": "quartet",
        "headerName": "quartet",
        "flex": 1,
        "minWidth": 165,
        "pinned": "left",
        "cellStyle": {"fontFamily": "monospace", "fontSize": "12px", "cursor": "pointer"},
        "tooltipValueGetter": {
            "function": "'click to select/deselect all networks for this quartet'"
        },
        "suppressMovable": True,
        "sortable": False,
    },
    *[
        {
            "field": f"nw{nw}",
            "headerName": str(nw),
            "headerClass": "nw-col-header",
            "headerTooltip": "click to select/deselect all available in this column",
            "width": 46,
            "cellStyle": _NW_CELL_STYLE,
            "valueFormatter": {"function": "''"},
            "tooltipValueGetter": _NW_TOOLTIP,
            "sortable": False,
            "suppressMovable": True,
        }
        for nw in range(13)
    ],
]


# ── Parse helpers ──────────────────────────────────────────────────────────────

_QUARTET_RE = re.compile(r"^(ics|tcs)-(\d+)-(\d+)-(\d+)$")
_CELL_RE = re.compile(r"^(ics|tcs)-(\d+)-(\d+)-(\d+)-(\d+)$")


def _parse_quartet(s: str) -> tuple[str, int, int, int] | None:
    """Parse ``'ics-18596-0-0'`` → ``(master, obsnum, subobsnum, scannum)``."""
    m = _QUARTET_RE.match(s.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))


def _parse_cell_key(s: str) -> tuple[str, int, int, int, int] | None:
    """Parse ``'ics-18596-0-0-0'`` → ``(master, obsnum, subobsnum, scannum, nw)``."""
    m = _CELL_RE.match(s.strip())
    if not m:
        return None
    return (
        m.group(1), int(m.group(2)), int(m.group(3)),
        int(m.group(4)), int(m.group(5)),
    )


def _parse_quartets(s: str) -> list[tuple[str, int, int, int]]:
    """Parse comma-separated quartet string → deduplicated list of tuples."""
    seen: set[tuple] = set()
    result: list[tuple[str, int, int, int]] = []
    for part in re.split(r"[,\n]+", s or ""):
        p = _parse_quartet(part.strip())
        if p and p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _parse_selected(s: str) -> set[str]:
    """Parse comma-separated cell-key string → set of valid cell-key strings."""
    keys: set[str] = set()
    for part in (s or "").split(","):
        part = part.strip()
        if _CELL_RE.match(part):
            keys.add(part)
    return keys


def _quartet_key(master: str, obsnum: int, subobsnum: int, scannum: int) -> str:
    return f"{master}-{obsnum}-{subobsnum}-{scannum}"


def _cell_key(master: str, obsnum: int, subobsnum: int, scannum: int, nw: int) -> str:
    return f"{master}-{obsnum}-{subobsnum}-{scannum}-{nw}"


def _get_ref_cell(
    selected: set[str],
    svc: ObsDataService,
) -> tuple[str, int, int, int, int] | None:
    """Return the ref cell: first selected cell sorted by (date_utc, nw)."""
    candidates: list[tuple] = []
    for key in selected:
        parsed = _parse_cell_key(key)
        if not parsed:
            continue
        row = svc.get_obs_row(*parsed)
        if row is None:
            continue
        date_utc = row.get("date_utc") or ""
        nw = parsed[4]
        candidates.append((date_utc, nw, parsed))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


# ── Main template ──────────────────────────────────────────────────────────────


class SweepViewerPage(Template):
    """Interactive sweep viewer page.

    Parameters
    ----------
    data_service
        :class:`~tolteca_web.obs.ObsDataService` providing catalog and zarr
        data access.
    """

    def __init__(self, data_service: ObsDataService) -> None:
        super().__init__()
        self._svc = data_service

        provider = self.child[dmc.MantineProvider]()
        root = provider.child[dmc.Stack](gap=0, style={"minHeight": "100vh"})

        # ── Header ────────────────────────────────────────────────────────
        header_box = root.child[dmc.Paper](
            withBorder=True,
            shadow="none",
            style={"borderLeft": "none", "borderRight": "none", "borderTop": "none"},
        )
        hdr = header_box.child[dmc.Group](px="md", py="xs", justify="space-between")
        hdr.child[dmc.Title](children="TolTEC Sweep Viewer", order=4)
        _hdr_right = hdr.child[dmc.Group](gap="md", align="center")
        self._kids_link = _hdr_right.child[dmc.Anchor](
            "Kids Results →",
            href="/reduced-obs",
            size="sm",
            fw=600,
            c="violet",
            style={"display": "none"},
        )
        _hdr_right.child[dmc.Anchor](
            "← Portal", href="/", size="sm", c="dimmed",
        )

        # ── URL state manager ─────────────────────────────────────────────
        self.url_mgr = UrlStateManager(SweepViewerState)
        root.child(self.url_mgr)

        container = root.child[dmc.Container](fluid=True, px="md", pt="sm")

        # ── Quartet TagsInput ──────────────────────────────────────────────
        ctrl = container.child[dmc.Stack](gap=2, mb="sm")
        ctrl.child[dmc.Text](
            children="Quartets  (master · obsnum · subobsnum · scannum)",
            size="sm",
            fw=500,
        )
        ctrl.child[dmc.Text](
            children=(
                "e.g.  ics-18596-0-0    press Enter or Tab to add · "
                "click × to remove"
            ),
            size="xs",
            c="dimmed",
        )
        self._quartets_input = ctrl.child[dmc.TagsInput](
            placeholder="ics-18596-0-0",
            value=[],
            w=520,
            size="sm",
        )

        # ── Matrix ────────────────────────────────────────────────────────
        matrix_paper = container.child[dmc.Paper](
            withBorder=True, p=0, mb="sm", radius="sm",
            style={"overflow": "hidden"},
        )
        matrix_hdr = matrix_paper.child[dmc.Group](
            p="xs", pb=0, justify="space-between", align="center", wrap="nowrap",
        )
        _hdr_left = matrix_hdr.child[dmc.Group](gap="xs", align="center", wrap="nowrap")
        _hdr_left.child[dmc.Text](
            children="Obs-file selection matrix",
            size="xs",
            fw=600,
            c="dimmed",
        )
        self._select_all_btn = _hdr_left.child[dmc.Button](
            "Select All", size="compact-xs", variant="light", color="blue",
        )
        self._clear_btn = _hdr_left.child[dmc.Button](
            "Clear", size="compact-xs", variant="light", color="gray",
        )
        matrix_hdr.child[dmc.Text](
            children="gray = absent  ·  amber = no zarr  ·  blue = available  ·  solid = selected"
                     "  ·  click quartet = row  ·  click nw# header = column",
            size="xs",
            c="dimmed",
        )
        # AG Grid matrix — wrapped in ScrollArea for clean horizontal scroll
        _scroll = matrix_paper.child[dmc.ScrollArea](
            type="hover", offsetScrollbars=True,
        )
        self._matrix_grid = _scroll.child[dag.AgGrid](
            rowData=[],
            columnDefs=_MATRIX_COL_DEFS,
            getRowId="params.data.quartet",
            defaultColDef={"resizable": False, "sortable": False},
            dashGridOptions={
                "domLayout": "autoHeight",
                "tooltipShowDelay": 300,
                "suppressRowClickSelection": True,
                "suppressHorizontalScroll": True,
                "headerHeight": 30,
                "rowHeight": 28,
            },
            eventListeners={"columnHeaderClicked": ["onNwHeaderClicked(params, setGridProps)"]},
            style={"width": "100%", "minWidth": "763px", "maxWidth": "800px"},
            className="ag-theme-quartz tolteca-matrix",
        )

        # ── Obs info row ──────────────────────────────────────────────────
        info_paper = container.child[dmc.Paper](
            withBorder=True, px="md", py="xs", mb="xs", radius="sm"
        )
        info_inner = info_paper.child[dmc.Group](
            gap="xs", align="center", wrap="wrap"
        )
        info_inner.child[dmc.Text](
            children="Enabled:", size="xs", c="dimmed", fw=600
        )
        self._obs_info_box = info_inner.child[dmc.Group](
            gap="xs", wrap="wrap", children=[]
        )

        # ── Channel pager + plot-mode selector ────────────────────────────
        pager_row = container.child[dmc.Group](
            gap="md", align="center", mb="xs", wrap="wrap"
        )
        self._mode_ctrl = pager_row.child[dmc.SegmentedControl](
            data=[
                {"label": "S21 (dB)", "value": "s21"},
                {"label": "I–Q plane", "value": "iq"},
            ],
            value="s21",
            size="xs",
        )
        self._pager = Pager(
            per_page_options=[10, 30, 60, 120],
            default_per_page=_CHANS_PER_PAGE,
        )
        pager_row.child(self._pager)
        self._ref_badge = pager_row.child[dmc.Badge](
            children="ref: —",
            color="blue",
            variant="light",
            size="md",
        )
        self._n_chan_text = pager_row.child[dmc.Text](
            children="", size="sm", c="dimmed"
        )

        # ── Main subplot grid ─────────────────────────────────────────────
        self._graph = container.child[dcc.Graph](
            figure={},
            style={"width": "100%"},
            config={"displayModeBar": True, "scrollZoom": False},
        )

        # ── Status ────────────────────────────────────────────────────────
        self._status = container.child[dmc.Text](
            children="", size="xs", c="dimmed", mt="xs"
        )

        # ── Zarr panel ────────────────────────────────────────────────────
        zarr_paper = container.child[dmc.Paper](
            withBorder=True, p="xs", mt="sm", radius="sm"
        )
        zarr_hdr = zarr_paper.child[dmc.Group](
            justify="space-between", align="center"
        )
        zarr_hdr.child[dmc.Text](
            children="⚡ Partial zarr retrieval", size="xs", fw=600
        )
        _mode_color = "blue" if data_service.zarr_mode == "http" else "gray"
        zarr_hdr.child[dmc.Badge](
            children=f"MODE: {data_service.zarr_mode.upper()}",
            color=_mode_color,
            variant="light",
            size="xs",
        )
        self._zarr_stats = zarr_hdr.child[dmc.Text](
            children="", size="xs", c="dimmed"
        )
        self._fetch_history_store = zarr_paper.child[dcc.Store](
            data={"panels": [], "session_bytes": 0}
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def setup_callbacks(self, app) -> None:  # noqa: C901
        """Register all Dash callbacks and URL-state bindings."""

        _s = self.url_mgr.store

        # CS-1: quartets field store → TagsInput value (string → list)
        app.clientside_callback(
            """function(val) {
                if (!val || val.trim() === "") return [];
                return val.split(",").map(s => s.trim()).filter(s => s.length > 0);
            }""",
            Output(self._quartets_input(), "value"),
            Input(_s("quartets")(), "data"),
        )

        # CS-2: TagsInput value → state store (list → string).
        # Also purges selected cells whose quartet is no longer in the list.
        app.clientside_callback(
            """function(val, state) {
                if (!state) return window.dash_clientside.no_update;
                var validQ = new Set((val || []).map(s => s.trim()).filter(s => s));
                var joined = Array.from(validQ).join(",");
                var curSel = (state["selected"] || "").split(",").filter(s => s.trim());
                var newSel = curSel.filter(function(cell) {
                    var parts = cell.split("-");
                    if (parts.length < 5) return false;
                    var q = parts.slice(0, 4).join("-");
                    return validQ.has(q);
                });
                var newSelStr = newSel.join(",");
                if (joined === (state["quartets"] || "") &&
                    newSelStr === (state["selected"] || "")) {
                    return window.dash_clientside.no_update;
                }
                var s = Object.assign({}, state);
                s["quartets"] = joined;
                s["selected"] = newSelStr;
                return s;
            }""",
            Output(self.url_mgr.state_store(), "data", allow_duplicate=True),
            Input(self._quartets_input(), "value"),
            State(self.url_mgr.state_store(), "data"),
            prevent_initial_call=True,
        )

        # CS-3: plot_mode field → SegmentedControl value
        app.clientside_callback(
            """function(val) { return val || "s21"; }""",
            Output(self._mode_ctrl(), "value"),
            Input(_s("plot_mode")(), "data"),
        )

        # CS-4: SegmentedControl value → state store plot_mode field
        app.clientside_callback(
            """function(val, state) {
                if (!state) return window.dash_clientside.no_update;
                var cur = state["plot_mode"] || "s21";
                var next = val || "s21";
                if (cur === next) return window.dash_clientside.no_update;
                var s = Object.assign({}, state);
                s["plot_mode"] = next;
                return s;
            }""",
            Output(self.url_mgr.state_store(), "data", allow_duplicate=True),
            Input(self._mode_ctrl(), "value"),
            State(self.url_mgr.state_store(), "data"),
            prevent_initial_call=True,
        )

        svc = self._svc

        # ── M1: (quartets, selected) → matrix AG Grid rowData ─────────────
        @app.callback(
            Output(self._matrix_grid(), "rowData"),
            Input(_s("quartets")(), "data"),
            Input(_s("selected")(), "data"),
        )
        def _update_matrix(quartets_str: str, selected_str: str) -> list:
            quartets = _parse_quartets(quartets_str or "")
            selected = _parse_selected(selected_str or "")
            rows = []
            for master, obsnum, subobsnum, scannum in quartets:
                qk = _quartet_key(master, obsnum, subobsnum, scannum)
                row: dict = {"quartet": qk}
                for nw in _NWS:
                    key = _cell_key(master, obsnum, subobsnum, scannum, nw)
                    cat_row = svc.get_obs_row(master, obsnum, subobsnum, scannum, nw)
                    if cat_row is None:
                        state = 0
                    else:
                        zarr_rel = cat_row.get("zarr_path") or ""
                        zarr_ok = bool(zarr_rel and (svc.cache_root / zarr_rel).exists())
                        state = 3 if key in selected else (2 if zarr_ok else 1)
                    row[f"nw{nw}"] = state
                rows.append(row)
            return rows

        # ── M2: matrix cell click → toggle selected (clientside) ───────────
        # Handles two interaction modes:
        #   • click nw cell (value≥2)       → toggle individual cell
        #   • click quartet cell (data row) → select/deselect all available in row
        # Column select (nw header click) is handled by M2b via eventListeners.
        app.clientside_callback(
            """function(cellClicked, rowData, state) {
                if (!cellClicked || !state) return window.dash_clientside.no_update;
                var colId = cellClicked.colId;
                var val = cellClicked.value;
                var cur = (state["selected"] || "").split(",").filter(function(s) { return s.trim(); });
                var sel = new Set(cur);
                var keys, allSel, nwIdx, row, quartet, cellKey;

                if (colId === "quartet" && val) {
                    // Row select: click on quartet label cell
                    quartet = cellClicked.rowId;
                    if (!quartet) return window.dash_clientside.no_update;
                    row = (rowData || []).find(function(r) { return r.quartet === quartet; });
                    if (!row) return window.dash_clientside.no_update;
                    keys = [];
                    for (nwIdx = 0; nwIdx <= 12; nwIdx++) {
                        if ((row["nw" + nwIdx] || 0) >= 2) keys.push(quartet + "-" + nwIdx);
                    }
                    if (!keys.length) return window.dash_clientside.no_update;
                    allSel = keys.every(function(k) { return sel.has(k); });
                    keys.forEach(function(k) { if (allSel) sel.delete(k); else sel.add(k); });

                } else if (colId.startsWith("nw") && (val || 0) >= 2) {
                    // Individual cell toggle
                    nwIdx = parseInt(colId.slice(2));
                    if (isNaN(nwIdx)) return window.dash_clientside.no_update;
                    cellKey = cellClicked.rowId + "-" + nwIdx;
                    if (sel.has(cellKey)) { sel.delete(cellKey); } else { sel.add(cellKey); }

                } else {
                    return window.dash_clientside.no_update;
                }

                var s = Object.assign({}, state);
                s["selected"] = Array.from(sel).sort().join(",");
                return s;
            }""",
            Output(self.url_mgr.state_store(), "data", allow_duplicate=True),
            Input(self._matrix_grid(), "cellClicked"),
            State(self._matrix_grid(), "rowData"),
            State(self.url_mgr.state_store(), "data"),
            prevent_initial_call=True,
        )

        # ── M2b: nw column header click → select/deselect column ───────────
        # Triggered via eventListeners/dashAgGridFunctions.onNwHeaderClicked,
        # which writes {colId} into eventData via setGridProps.
        app.clientside_callback(
            """function(eventData, rowData, state) {
                if (!eventData || !eventData.data || !state) return window.dash_clientside.no_update;
                var colId = eventData.data.colId;
                if (!colId || !colId.startsWith("nw")) return window.dash_clientside.no_update;
                var nwIdx = parseInt(colId.slice(2));
                if (isNaN(nwIdx)) return window.dash_clientside.no_update;

                var cur = (state["selected"] || "").split(",").filter(function(s) { return s.trim(); });
                var sel = new Set(cur);
                var keys = [];
                (rowData || []).forEach(function(r) {
                    if ((r[colId] || 0) >= 2) keys.push(r.quartet + "-" + nwIdx);
                });
                if (!keys.length) return window.dash_clientside.no_update;
                var allSel = keys.every(function(k) { return sel.has(k); });
                keys.forEach(function(k) { if (allSel) sel.delete(k); else sel.add(k); });

                var s = Object.assign({}, state);
                s["selected"] = Array.from(sel).sort().join(",");
                return s;
            }""",
            Output(self.url_mgr.state_store(), "data", allow_duplicate=True),
            Input(self._matrix_grid(), "eventData"),
            State(self._matrix_grid(), "rowData"),
            State(self.url_mgr.state_store(), "data"),
            prevent_initial_call=True,
        )

        # ── M3: Select All button ───────────────────────────────────────────
        app.clientside_callback(
            """function(n, rowData, state) {
                if (!n || !state) return window.dash_clientside.no_update;
                var sel = new Set();
                (rowData || []).forEach(function(row) {
                    for (var nw = 0; nw <= 12; nw++) {
                        if ((row["nw" + nw] || 0) >= 2) sel.add(row.quartet + "-" + nw);
                    }
                });
                var s = Object.assign({}, state);
                s["selected"] = Array.from(sel).sort().join(",");
                return s;
            }""",
            Output(self.url_mgr.state_store(), "data", allow_duplicate=True),
            Input(self._select_all_btn(), "n_clicks"),
            State(self._matrix_grid(), "rowData"),
            State(self.url_mgr.state_store(), "data"),
            prevent_initial_call=True,
        )

        # ── M4: Clear button ────────────────────────────────────────────────
        app.clientside_callback(
            """function(n, state) {
                if (!n || !state) return window.dash_clientside.no_update;
                var s = Object.assign({}, state);
                s["selected"] = "";
                return s;
            }""",
            Output(self.url_mgr.state_store(), "data", allow_duplicate=True),
            Input(self._clear_btn(), "n_clicks"),
            State(self.url_mgr.state_store(), "data"),
            prevent_initial_call=True,
        )

        # ── A: selected → pager n_items from ref cell + label text ─────────
        @app.callback(
            Output(self._pager.n_items_store(), "data"),
            Output(self._n_chan_text(), "children"),
            Output(self._ref_badge(), "children"),
            Input(_s("selected")(), "data"),
        )
        def _update_n_chans(selected_str: str) -> tuple:
            selected = _parse_selected(selected_str or "")
            ref = _get_ref_cell(selected, svc)
            if ref is None:
                return 0, "", "ref: —"
            row = svc.get_obs_row(*ref)
            if row is None:
                return 0, "", "ref: —"
            n_ch = row.get("n_chans") or 0
            ref_label = f"{_quartet_key(*ref[:4])} nw{ref[4]}"
            return n_ch, f"{n_ch} channels", f"ref: {ref_label}"

        # ── B: vertical subplot figure — S21 or I-Q mode ──────────────────
        empty_hist: dict = {"panels": [], "session_bytes": 0}

        @app.callback(
            Output(self._graph(), "figure"),
            Output(self._status(), "children"),
            Output(self._fetch_history_store(), "data"),
            Input(_s("selected")(), "data"),
            Input(self._pager.page_store(), "data"),
            Input(_s("plot_mode")(), "data"),
            State(self._fetch_history_store(), "data"),
            State(_s("quartets")(), "data"),
        )
        def _update_grid(
            selected_str: str,
            page_info: dict,
            plot_mode_data: str,
            prev_hist: dict,
            quartets_str: str,
        ) -> tuple:
            empty_fig: dict = {
                "data": [],
                "layout": {"template": "plotly_white", "height": 300},
            }

            selected = _parse_selected(selected_str or "")
            cell_tuples_raw = [_parse_cell_key(k) for k in selected]
            cell_tuples_raw = [t for t in cell_tuples_raw if t is not None]

            if not cell_tuples_raw:
                return (
                    empty_fig,
                    "Enable cells in the matrix above to plot.",
                    empty_hist,
                )

            # Sort by (quartet input order, nw) — preserves user's quartet ordering
            # with networks ascending within each quartet.
            quartet_order: dict[str, int] = {
                _quartet_key(*q): i
                for i, q in enumerate(_parse_quartets(quartets_str or ""))
            }

            def _sort_key(t: tuple) -> tuple:
                return (quartet_order.get(_quartet_key(*t[:4]), 9999), t[4])

            cell_tuples = sorted(cell_tuples_raw, key=_sort_key)

            n_items = (page_info or {}).get("n_items", 0)
            if n_items > 0 and page_info:
                start = page_info.get("start", 0)
                stop = page_info.get("stop", min(_CHANS_PER_PAGE, n_items))
            else:
                start, stop = 0, _CHANS_PER_PAGE
            chan_slice = slice(start, stop)

            n_panels = len(cell_tuples)
            subplot_titles = [
                f"{_quartet_key(*t[:4])}  nw{t[4]}" for t in cell_tuples
            ]

            plot_mode = (plot_mode_data or "s21").strip().lower()
            is_iq = plot_mode == "iq"

            fig = make_subplots(
                rows=n_panels,
                cols=1,
                subplot_titles=subplot_titles,
                shared_xaxes=not is_iq,
                vertical_spacing=0.04,
            )

            new_panels: list[dict] = []
            status_parts: list[str] = []

            for row_idx, tup in enumerate(cell_tuples, 1):
                master, obsnum, subobsnum, scannum, nw_val = tup
                qk = _quartet_key(master, obsnum, subobsnum, scannum)
                t0 = time.perf_counter()
                try:
                    ds = svc.get_obs_data(
                        master, obsnum, subobsnum, scannum, nw_val,
                        chan_slice=chan_slice,
                    )
                    ds_c = ds[["I", "Q", "tone_freq"]].compute()
                except Exception as exc:
                    fig.add_annotation(
                        text=str(exc)[:80],
                        row=row_idx,
                        col=1,
                        xref="x domain",
                        yref="y domain",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font={"color": "#e03131", "size": 9},
                    )
                    status_parts.append(f"{qk} nw{nw_val}: {exc}")
                    continue

                elapsed_ms = (time.perf_counter() - t0) * 1000
                I_arr = ds_c["I"].values
                Q_arr = ds_c["Q"].values
                n_page_chans, n_samples = I_arr.shape
                bytes_fetched = n_page_chans * n_samples * 4 * 2

                if is_iq:
                    # ── I-Q plane: scattergl(x=I[i], y=Q[i]) per channel ─
                    for i in range(n_page_chans):
                        fig.add_trace(
                            dict(
                                type="scattergl",
                                x=to_typed_array_spec(I_arr[i]),
                                y=to_typed_array_spec(Q_arr[i]),
                                mode="markers",
                                name=f"ch{start + i}",
                                marker={"size": 3, "opacity": 0.7},
                                showlegend=False,
                                hovertemplate=(
                                    f"<b>ch{start + i}</b><br>"
                                    "I: %{x:.0f} adu<br>"
                                    "Q: %{y:.0f} adu"
                                    "<extra></extra>"
                                ),
                            ),
                            row=row_idx,
                            col=1,
                        )
                    # Equal-aspect range: expand the narrower axis to match
                    if n_page_chans > 0:
                        I_flat = I_arr.ravel()
                        Q_flat = Q_arr.ravel()
                        x_c = float(I_flat.mean())
                        y_c = float(Q_flat.mean())
                        half = max(
                            float(np.abs(I_flat - x_c).max()),
                            float(np.abs(Q_flat - y_c).max()),
                            1.0,
                        ) * 1.1
                        fig.update_xaxes(
                            range=[x_c - half, x_c + half],
                            row=row_idx, col=1,
                        )
                        fig.update_yaxes(
                            range=[y_c - half, y_c + half],
                            row=row_idx, col=1,
                        )
                    new_panels.append({
                        "spec": qk,
                        "nw": nw_val,
                        "start": start,
                        "stop": stop,
                        "ms": round(elapsed_ms, 1),
                        "bytes": bytes_fetched,
                    })

                else:
                    # ── S21 mode: scattergl lines vs f_tone (offset LO) ───
                    amp = np.sqrt(I_arr**2 + Q_arr**2)
                    amp_max = amp.max(axis=1, keepdims=True)
                    amp_max = np.where(amp_max == 0, 1.0, amp_max)
                    s21_db = 20.0 * np.log10(amp / amp_max)

                    acc = ZarrSweepDataset(ds_c)
                    chan_data = acc.get_chan_axis_data()
                    sweep_data = acc.get_sweep_axis_data()
                    f_tone_hz = chan_data["f_tone_hz"]
                    f_sweep_hz = sweep_data["f_sweep_hz"]
                    lo_center_hz = sweep_data["lo_center_hz"]
                    lo_center_ghz = lo_center_hz / 1e9

                    f_x_mhz = (
                        f_tone_hz[:, np.newaxis] + f_sweep_hz[np.newaxis, :]
                    ) / 1e6

                    for i in range(n_page_chans):
                        # LO offset in GHz stored as trace meta — avoids
                        # sending 500 customdata floats per channel.
                        # Hover shows MHz (x-axis) + readable channel label.
                        fig.add_trace(
                            dict(
                                type="scattergl",
                                x=to_typed_array_spec(f_x_mhz[i]),
                                y=to_typed_array_spec(s21_db[i]),
                                mode="lines",
                                name=f"ch{start + i}",
                                line={"width": 1},
                                showlegend=False,
                                meta={"lo": round(lo_center_ghz, 6)},
                                hovertemplate=(
                                    f"<b>ch{start + i}</b><br>"
                                    "fₜₒₙₑ: %{x:.3f} MHz<br>"
                                    f"LO: {lo_center_ghz:.4f} GHz"
                                    "<extra></extra>"
                                ),
                            ),
                            row=row_idx,
                            col=1,
                        )

                    new_panels.append({
                        "spec": qk,
                        "nw": nw_val,
                        "start": start,
                        "stop": stop,
                        "ms": round(elapsed_ms, 1),
                        "bytes": bytes_fetched,
                        "lo_center_hz": lo_center_hz,
                        "x_min": float(f_x_mhz.min()),
                        "x_max": float(f_x_mhz.max()),
                    })

                status_parts.append(
                    f"{qk} nw{nw_val}: ch{start}\u2013{stop - 1}"
                    f" [{elapsed_ms:.0f} ms]"
                )

            # ── Axis decoration per mode ──────────────────────────────────
            if is_iq:
                for r in range(1, n_panels + 1):
                    x_anchor = "x" if r == 1 else f"x{r}"
                    fig.update_xaxes(
                        title_text="I (adu)",
                        title_font={"size": 10},
                        title_standoff=4,
                        row=r, col=1,
                    )
                    fig.update_yaxes(
                        title_text="Q (adu)",
                        title_font={"size": 10},
                        title_standoff=2,
                        scaleanchor=x_anchor,
                        scaleratio=1,
                        row=r, col=1,
                    )
            else:
                # S21: y label on every panel, x label on bottom only
                for r in range(1, n_panels + 1):
                    fig.update_yaxes(
                        title_text="S21 (dB)",
                        title_font={"size": 10},
                        title_standoff=2,
                        row=r,
                        col=1,
                    )
                fig.update_xaxes(
                    title_text="fₜₒₙₑ (MHz)  \u2014  offset from LO center",
                    title_font={"size": 10},
                    title_standoff=4,
                    row=n_panels,
                    col=1,
                )
                # Secondary GHz axis at top + per-panel LO annotations
                loaded = [p for p in new_panels if "lo_center_hz" in p]
                if loaded:
                    all_x = [p["x_min"] for p in loaded] + [p["x_max"] for p in loaded]
                    x_pad = (max(all_x) - min(all_x)) * 0.04
                    x_range = [min(all_x) - x_pad, max(all_x) + x_pad]
                    fig.update_xaxes(range=x_range)

                    ref_lo_ghz = loaded[0]["lo_center_hz"] / 1e9
                    ref_nw = loaded[0]["nw"]
                    sec_range_ghz = [
                        ref_lo_ghz + x_range[0] / 1e3,
                        ref_lo_ghz + x_range[1] / 1e3,
                    ]
                    sec_key = f"xaxis{n_panels + 1}"
                    fig.update_layout(**{
                        sec_key: {
                            "overlaying": "x",
                            "side": "top",
                            "anchor": "y",
                            "range": sec_range_ghz,
                            "tickformat": ".4f",
                            "tickfont": {"size": 8, "color": "#868e96"},
                            "showgrid": False,
                            "title": {
                                "text": f"Freq (GHz)  [nw{ref_nw}]",
                                "font": {"size": 8, "color": "#868e96"},
                                "standoff": 2,
                            },
                            "nticks": 5,
                            "fixedrange": True,
                        }
                    })

                    for p_idx, panel in enumerate(loaded):
                        r = p_idx + 1
                        xref = "x domain" if r == 1 else f"x{r} domain"
                        yref = "y domain" if r == 1 else f"y{r} domain"
                        lo_ghz = panel["lo_center_hz"] / 1e9
                        fig.add_annotation(
                            text=f"LO: {lo_ghz:.4f} GHz",
                            xref=xref,
                            yref=yref,
                            x=1.0,
                            y=1.0,
                            xanchor="right",
                            yanchor="top",
                            showarrow=False,
                            font={"size": 8, "color": "#868e96"},
                            bgcolor="rgba(255,255,255,0.8)",
                            borderpad=2,
                        )

            fig.update_layout(
                template="plotly_white",
                height=max(350, 220 * n_panels + 80),
                margin={"l": 55, "r": 15, "t": 60, "b": 50},
                showlegend=False,
                paper_bgcolor="white",
                plot_bgcolor="white",
            )

            session_bytes = (prev_hist or {}).get("session_bytes", 0) + sum(
                p["bytes"] for p in new_panels
            )
            new_hist = {"panels": new_panels, "session_bytes": session_bytes}

            ref = _get_ref_cell(selected, svc)
            ref_label = (
                f"ref: {_quartet_key(*ref[:4])} nw{ref[4]}" if ref else ""
            )
            mode_label = "I-Q" if is_iq else "S21"
            status = (
                (f"{ref_label}  [{mode_label}]  \u00b7  " if ref_label else "")
                + (
                    "  \u00b7  ".join(status_parts)
                    if status_parts
                    else "No data loaded."
                )
            )
            return fig, status, new_hist

        # ── C: selected → obs-info badge row ──────────────────────────────
        @app.callback(
            Output(self._obs_info_box(), "children"),
            Input(_s("selected")(), "data"),
        )
        def _update_obs_info(selected_str: str) -> list:
            cells = sorted(_parse_selected(selected_str or ""))
            if not cells:
                return [dmc.Text("—", size="xs", c="dimmed")]
            items: list = []
            for key in cells:
                parsed = _parse_cell_key(key)
                if not parsed:
                    continue
                master, obsnum, subobsnum, scannum, nw = parsed
                row = svc.get_obs_row(master, obsnum, subobsnum, scannum, nw)
                if row is None:
                    items.append(
                        dmc.Badge(
                            f"{key}: not found",
                            color="red",
                            variant="light",
                            size="sm",
                        )
                    )
                    continue
                dk = row.get("data_kind") or "?"
                color = _DATA_KIND_COLORS.get(dk, "gray")
                n_ch = row.get("n_chans") or "?"
                lo_hz = row.get("lo_center_freq_hz") or 0
                items.append(
                    dmc.Badge(
                        f"{key}  [{dk}]  {n_ch} ch  lo:{lo_hz / 1e6:.1f} MHz",
                        color=color,
                        variant="light",
                        size="sm",
                    )
                )
            return items or [dmc.Text("—", size="xs", c="dimmed")]

        # ── K: selected → Kids Results link ───────────────────────────────
        @app.callback(
            Output(self._kids_link(), "style"),
            Output(self._kids_link(), "href"),
            Input(_s("selected")(), "data"),
            Input(_s("quartets")(), "data"),
        )
        def _update_kids_link(selected_str: str, quartets_str: str) -> tuple:
            """Show 'Kids Results →' when the selected cell has kids data."""
            from urllib.parse import urlencode

            cells = sorted(_parse_selected(selected_str or ""))
            if not cells:
                return {"display": "none"}, "/reduced-obs"

            # Use the first selected cell for the check
            first = cells[0]
            parsed = _parse_cell_key(first)
            if not parsed:
                return {"display": "none"}, "/reduced-obs"

            master, obsnum, subobsnum, scannum, nw = parsed
            try:
                has_kids = svc.has_kids_reduction(master, obsnum, subobsnum, scannum, nw)
            except Exception:
                has_kids = False

            if not has_kids:
                return {"display": "none"}, "/reduced-obs"

            qs = urlencode({
                "quartets": quartets_str or "",
                "selected": selected_str or "",
            })
            return {"display": "inline"}, f"/reduced-obs?{qs}"

        # ── D: zarr stats ──────────────────────────────────────────────────
        @app.callback(
            Output(self._zarr_stats(), "children"),
            Input(self._fetch_history_store(), "data"),
        )
        def _update_zarr_stats(hist: dict) -> str:
            if not hist:
                return ""
            panels = hist.get("panels", [])
            if not panels:
                return ""
            total_ms = sum(p["ms"] for p in panels)
            total_kb = sum(p["bytes"] for p in panels) / 1024
            session_kb = hist.get("session_bytes", 0) / 1024
            n = len(panels)
            return (
                f"Latest: {n} panel{'s' if n != 1 else ''}"
                f"  ·  {total_kb:.0f} KB  in  {total_ms:.0f} ms"
                f"  ·  session total: {session_kb:.0f} KB"
            )
