"""KidsDiagViewerPage — v2-style KIDs diagnostic figures viewer.

Shows the full set of figures from ``make_kids_find_figs`` in a tabbed layout:

    Peaks | Peak Props | D21 Summary | S21 Summary | Det Summary
    | Matched | Matched Ref

URL params
----------
quartets : str
    ``"master-obsnum-subobsnum-scannum"``
nw : int
    Network index (default 0).

Tab routing design
------------------
Each figure callback listens only to its own ``dcc.Store`` (trigger store).
A single router callback watches ``tabs.value``, ``search``, and ``nw``:
- It sets the active tab's store to ``"<search>|<nw>"`` only when that key
  differs from the store's current value (i.e., first visit or obs changed).
- Inactive tabs' stores are left untouched (``no_update``).

This means:
- Tab switches to an *already-loaded* tab do NOT update any store, so no
  figure callback fires and no loading spinner appears.
- Changing the observation (search/nw) updates the active tab immediately;
  inactive tabs reload lazily the next time they are activated.
"""

from __future__ import annotations

from urllib.parse import parse_qs

from dash import Input, Output, State, dcc, html, no_update
from dash_component_template import Template
import dash_mantine_components as dmc

from tolteca_web.obs import ObsDataService

__all__ = ["KidsDiagViewerPage"]

# ── Tab definitions ────────────────────────────────────────────────────────────

_TABS: list[tuple[str, str]] = [
    ("peaks",       "Peaks"),
    ("peak_props",  "Peak Props"),
    ("d21_summary", "D21 Summary"),
    ("s21_summary", "S21 Summary"),
    ("det_summary", "Det Summary"),
    ("matched",     "Matched"),
    ("matched_ref", "Matched Ref"),
]
_TAB_KEYS = [k for k, _ in _TABS]


def _parse_quartet(s: str) -> tuple[str, int, int, int] | None:
    parts = s.strip().split("-")
    if len(parts) != 4:
        return None
    try:
        return parts[0], int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return None


# ── Page ──────────────────────────────────────────────────────────────────────


class KidsDiagViewerPage(Template):
    """KIDs diagnostic figure viewer (``/kids-diag``)."""

    def __init__(self, data_service: ObsDataService) -> None:
        super().__init__()
        self._svc = data_service

        provider = self.child[dmc.MantineProvider]()
        root = provider.child[dmc.Stack](gap=0, style={"minHeight": "100vh"})

        self._location = root.child[dcc.Location](id="kids-diag-url", refresh=False)

        # ── Header ────────────────────────────────────────────────────────
        header_box = root.child[dmc.Paper](
            withBorder=True, shadow="none",
            style={"borderLeft": "none", "borderRight": "none", "borderTop": "none"},
        )
        hdr = header_box.child[dmc.Group](px="md", py="xs", justify="space-between")
        hdr.child[dmc.Title]("TolTEC KIDs Diagnostics", order=4)
        hdr_right = hdr.child[dmc.Group](gap="md")
        self._back_link = hdr_right.child[dmc.Anchor](
            "← KIDs Viewer", href="/reduced-obs", size="sm", c="dimmed",
        )

        container = root.child[dmc.Container](fluid=True, px="md", pt="sm")

        # ── Controls bar ──────────────────────────────────────────────────
        ctrl = container.child[dmc.Group](gap="md", align="flex-end", mb="sm")
        self._nw = ctrl.child[dmc.NumberInput](
            label="Network (nw)", value=0, min=0, max=15, step=1, w=140, size="xs",
        )
        self._cell_text = ctrl.child[dmc.Text](
            "", size="xs", c="dimmed", style={"flex": 1},
        )

        # ── Trigger stores (one per tab, invisible) ───────────────────────
        # Value = "<search>|<nw>" when loaded, None when never loaded.
        # The router callback updates only the active tab's store when the
        # observation key changes; figure callbacks listen to their store only.
        self._trigger_stores: dict[str, dcc.Store] = {}
        for key, _ in _TABS:
            self._trigger_stores[key] = container.child[dcc.Store](data=None)

        # ── Tabs ──────────────────────────────────────────────────────────
        self._tabs = container.child[dmc.Tabs](value="peaks")
        tabs_list = self._tabs.child[dmc.TabsList]()
        for key, label in _TABS:
            tabs_list.child[dmc.TabsTab](label, value=key)

        # One panel per tab — each contains a dcc.Loading + dcc.Graph
        self._graphs: dict[str, dcc.Graph] = {}
        for key, _label in _TABS:
            panel = self._tabs.child[dmc.TabsPanel](value=key, pt="sm")
            loading = panel.child[dcc.Loading](type="circle", style={"minHeight": 400})
            self._graphs[key] = loading.child[dcc.Graph](
                figure={},
                style={"width": "100%"},
                config={"displayModeBar": True, "scrollZoom": True},
            )

        self._status = container.child[dmc.Text](
            "Select an observation in the URL to view diagnostics.",
            size="xs", c="dimmed", mt="xs",
        )

    def setup_callbacks(self, app) -> None:  # noqa: C901
        svc = self._svc

        @app.callback(
            Output(self._back_link(), "href"),
            Input(self._location(), "search"),
        )
        def _update_back_link(search: str) -> str:
            qs = parse_qs((search or "").lstrip("?"))
            qs.pop("tab", None)
            s = ("?" + "&".join(f"{k}={v[0]}" for k, v in qs.items())) if qs else ""
            return f"/reduced-obs{s}"

        # Tab ↔ URL hash sync (hash never triggers search-based callbacks)
        @app.callback(
            Output(self._tabs(), "value"),
            Input(self._location(), "hash"),
            State(self._tabs(), "value"),
        )
        def _restore_tab(hash_val: str | None, current_tab: str | None):
            tab = (hash_val or "").lstrip("#")
            resolved = tab if tab in _TAB_KEYS else "peaks"
            return resolved if resolved != current_tab else no_update

        @app.callback(
            Output(self._location(), "hash"),
            Input(self._tabs(), "value"),
            State(self._location(), "hash"),
            prevent_initial_call=True,
        )
        def _sync_tab_to_hash(tab_value: str | None, current_hash: str | None):
            if not tab_value:
                return no_update
            if (current_hash or "").lstrip("#") == tab_value:
                return no_update
            return f"#{tab_value}"

        # ── Router: update only the active tab's trigger store ─────────────
        # Store value = "<search>|<nw>" — unchanged means already loaded,
        # so the downstream figure callback will NOT be re-triggered.
        @app.callback(
            [Output(self._trigger_stores[k](), "data") for k in _TAB_KEYS],
            Input(self._tabs(), "value"),
            Input(self._location(), "search"),
            Input(self._nw(), "value"),
            [State(self._trigger_stores[k](), "data") for k in _TAB_KEYS],
        )
        def _route_loads(active_tab, search, nw_val, *current_store_values):
            load_key = f"{search or ''}|{nw_val}"
            outputs = []
            for i, key in enumerate(_TAB_KEYS):
                if key == active_tab and current_store_values[i] != load_key:
                    outputs.append(load_key)
                else:
                    outputs.append(no_update)
            return outputs

        # ── Per-tab figure callbacks (listen to trigger store only) ────────
        def _make_tab_callback(bound_key: str, bound_graph, trigger_store):
            @app.callback(
                Output(bound_graph(), "figure"),
                Input(trigger_store(), "data"),
                State(self._location(), "search"),
                State(self._nw(), "value"),
            )
            def _update_fig(trigger_data, search: str, nw_val: int | None):
                if trigger_data is None:
                    return no_update
                return _compute_fig(search, nw_val, bound_key)

        for tab_key, graph in self._graphs.items():
            _make_tab_callback(tab_key, graph, self._trigger_stores[tab_key])

        # ── Status bar ────────────────────────────────────────────────────
        @app.callback(
            Output(self._cell_text(), "children"),
            Output(self._status(), "children"),
            Input(self._location(), "search"),
            Input(self._nw(), "value"),
        )
        def _update_status(search: str, nw_val: int | None) -> tuple[str, str]:
            qs = parse_qs((search or "").lstrip("?"))
            quartets_str = (qs.get("quartets", [""])[0] or "").strip()
            if not quartets_str:
                return "", "No observation selected."
            parsed = _parse_quartet(quartets_str)
            if not parsed:
                return "", f"Cannot parse quartet: {quartets_str!r}"
            master, obsnum, subobsnum, scannum = parsed
            nw = int(nw_val) if nw_val is not None else 0
            zarr_path = svc.get_zarr_path(master, obsnum, subobsnum, scannum, nw)
            if zarr_path is None:
                return (
                    f"{master}/{obsnum:08d}/{subobsnum}/{scannum}/nw{nw}",
                    "No zarr store found for this cell.",
                )
            cell_str = f"{master}/{obsnum:08d}/{subobsnum}/{scannum}/nw{nw}"
            data = svc.get_kids_diag_data(str(zarr_path))
            if data is None:
                return cell_str, "Pipeline failed — check logs."
            _dt, kf_ctx, _cfg = data
            ctd = kf_ctx.data
            n_det = len(ctd.detected) if ctd.detected is not None else 0
            return cell_str, f"{cell_str}  ·  {n_det} detections"

        def _compute_fig(search, nw_val, tab_key):
            qs = parse_qs((search or "").lstrip("?"))
            quartets_str = (qs.get("quartets", [""])[0] or "").strip()
            if not quartets_str:
                return {}
            parsed = _parse_quartet(quartets_str)
            if not parsed:
                return {}
            master, obsnum, subobsnum, scannum = parsed
            nw = int(nw_val) if nw_val is not None else 0
            zarr_path = svc.get_zarr_path(master, obsnum, subobsnum, scannum, nw)
            if zarr_path is None:
                return {}
            fig = svc.get_kids_diag_fig(str(zarr_path), tab_key)
            if fig is None:
                return {}
            return fig
