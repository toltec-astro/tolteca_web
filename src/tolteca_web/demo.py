"""Demo app showcasing all tolteca_web common templates.

Run:
    uv run tolteca_web run tolteca_web.demo:create_app
    # or directly:
    uv run python -m tolteca_web.demo

Then open http://localhost:8050/
"""

from __future__ import annotations

import numpy as np
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash_component_template import Template

import dash_mantine_components as dmc

from tolteca_web.common import (
    CacheMonitor,
    CollapseContent,
    DownloadButton,
    IntervalTimer,
    LiveUpdateSection,
    Pager,
    UrlStateManager,
)
from tolteca_web.common.plots.surface_plot import SurfacePlot
from pydantic import BaseModel

# ── URL state model ────────────────────────────────────────────────────────────


class _UrlDemoState(BaseModel):
    dataset: str = "A"
    limit: int = 10
    active: bool = False


# ── Section helpers ────────────────────────────────────────────────────────────


def _section(title: str) -> dmc.Stack:
    return dmc.Stack(
        gap="md",
        children=[
            dmc.Divider(label=title, labelPosition="left"),
        ],
    )


# ── Root app template ──────────────────────────────────────────────────────────


class DemoApp(Template):
    """Root template demonstrating all common widgets."""

    def __init__(self) -> None:
        super().__init__()

        # Single root: MantineProvider → Container
        container = self.child[dmc.MantineProvider]().child[dmc.Container](
            maw=1000, pt="md"
        )

        container.child[dmc.Title](children="tolteca_web common widget demo", order=2)
        container.child[dmc.Text](
            children="Showcasing all templates ported from the old dbc-based tolteca_web.",
            c="dimmed",
            mb="md",
        )

        tabs = container.child[dmc.Tabs](value="collapse")
        tab_list = tabs.child[dmc.TabsList]()
        for value, label in [
            ("collapse", "CollapseContent"),
            ("download", "DownloadButton"),
            ("timer", "IntervalTimer"),
            ("live", "LiveUpdateSection"),
            ("pager", "Pager"),
            ("cache", "CacheMonitor"),
            ("surface", "SurfacePlot"),
            ("urlstate", "UrlStateManager"),
        ]:
            tab_list.child[dmc.TabsTab](children=label, value=value)

        self._build_collapse(tabs)
        self._build_download(tabs)
        self._build_timer(tabs)
        self._build_live(tabs)
        self._build_pager(tabs)
        self._build_cache(tabs)
        self._build_surface(tabs)
        self._build_url_state(tabs)

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _build_collapse(self, tabs) -> None:
        panel = tabs.child[dmc.TabsPanel](value="collapse", pt="xs")
        panel.child[dmc.Text](
            children=(
                "CollapseContent wraps any content behind a toggle button. "
                "The open/close is driven by a clientside callback — no server round-trip."
            ),
            mb="sm",
        )

        col1 = CollapseContent("Show / Hide settings", opened=False)
        panel.child(col1)
        col1.content.child[dmc.Paper](
            p="sm",
            withBorder=True,
            children=[
                dmc.Text("This content was hidden!", size="sm"),
                dmc.TextInput(label="A setting", placeholder="value"),
            ],
        )

        col2 = CollapseContent("Another panel (starts open)", opened=True)
        panel.child(col2)
        col2.content.child[dmc.Alert](
            children="I was open from the start.",
            color="blue",
            variant="light",
        )

    def _build_download(self, tabs) -> None:
        panel = tabs.child[dmc.TabsPanel](value="download", pt="xs")
        panel.child[dmc.Text](
            children=(
                "DownloadButton combines dmc.Button + DashIconify + dcc.Download. "
                "Wire Output(btn.download(), 'data') to dcc.send_string() in your callback."
            ),
            mb="sm",
        )

        self.dl_btn = DownloadButton("Export CSV", tooltip="Click to download")
        panel.child(self.dl_btn)

        panel.child[dmc.Text](
            id="dl-status",
            children="(click button to download)",
            c="dimmed",
            size="sm",
            mt="xs",
        )

    def _build_timer(self, tabs) -> None:
        panel = tabs.child[dmc.TabsPanel](value="timer", pt="xs")
        panel.child[dmc.Text](
            children=(
                "IntervalTimer runs a dcc.Interval at 500 ms but only increments "
                "n_calls_store at the selected rate. Click the hourglass to expand controls."
            ),
            mb="sm",
        )

        self.timer = IntervalTimer(
            interval_options=[2_000, 5_000, 10_000],
            interval_option_value=2_000,
        )
        panel.child(self.timer)

        panel.child[dmc.Text](
            id="timer-tick-display",
            children="Tick count: 0",
            size="lg",
            mt="sm",
        )

    def _build_live(self, tabs) -> None:
        panel = tabs.child[dmc.TabsPanel](value="live", pt="xs")
        panel.child[dmc.Text](
            children=(
                "LiveUpdateSection composes a title, IntervalTimer, LoadingOverlay "
                "and a banner row into one reusable layout."
            ),
            mb="sm",
        )

        self.live_sec = LiveUpdateSection(
            "Sensor data",
            interval_options=[3_000, 10_000],
            interval_option_value=3_000,
        )
        panel.child(self.live_sec)

        self.live_sec.content.child[dmc.Text](
            id="live-value",
            children="Waiting for first tick...",
            size="xl",
            fw=700,
        )

    def _build_pager(self, tabs) -> None:
        panel = tabs.child[dmc.TabsPanel](value="pager", pt="xs")
        panel.child[dmc.Text](
            children=(
                "Pager wraps dmc.Pagination with a per-page selector. "
                "Write total item count to n_items_store.data; read page info from page_store.data."
            ),
            mb="sm",
        )

        self.pager = Pager(per_page_options=[5, 10, 20], default_per_page=5)
        # Pre-load 47 items
        self.pager.n_items_store  # the store is part of the tree
        panel.child(self.pager)

        panel.child[dmc.Code](id="pager-info", block=True, children="(no data)")
        # Store pre-loaded with 47 items via initial data
        panel.child[dcc.Store](id="pager-demo-n-items", data=47)

    def _build_cache(self, tabs) -> None:
        panel = tabs.child[dmc.TabsPanel](value="cache", pt="xs")
        panel.child[dmc.Text](
            children=(
                "CacheMonitor reads structured status data from its status_store "
                "and renders progress bars per active download."
            ),
            mb="sm",
        )

        self.cache = CacheMonitor()
        panel.child(self.cache)

        panel.child[dmc.Button](
            id="cache-trigger",
            children="Simulate download progress",
            variant="light",
            mt="sm",
        )
        panel.child[dcc.Store](id="cache-step", data=0)
        panel.child[dcc.Interval](id="cache-interval", interval=400, disabled=True)

    def _build_surface(self, tabs) -> None:
        panel = tabs.child[dmc.TabsPanel](value="surface", pt="xs")
        panel.child[dmc.Text](
            children=(
                "SurfacePlot renders a 2-D heatmap with a companion histogram and "
                "range slider. Call make_figure_data() server-side and store the result."
            ),
            mb="sm",
        )

        self.surface = SurfacePlot()
        panel.child(self.surface)

        panel.child[dmc.Button](
            id="surface-refresh",
            children="Refresh random data",
            variant="light",
            mt="sm",
        )

    def _build_url_state(self, tabs) -> None:
        panel = tabs.child[dmc.TabsPanel](value="urlstate", pt="xs")
        panel.child[dmc.Text](
            children=(
                "UrlStateManager syncs a Pydantic model to individual URL parameters "
                "(?dataset=C&limit=75). All callbacks are clientside — zero server "
                "round-trips. bind() auto-wires bidirectional sync."
            ),
            mb="sm",
        )
        panel.child[dmc.Text](
            children='Try navigating to: /?dataset=C&limit=75&active=true',
            c="dimmed",
            size="sm",
            mb="md",
        )

        self.url_mgr = UrlStateManager(_UrlDemoState, show_debug=True)
        panel.child(self.url_mgr)

        controls = panel.child[dmc.Group](gap="md", align="flex-end", mb="sm")
        self._url_select = controls.child[dmc.Select](
            label="Dataset",
            data=["A", "B", "C", "D"],
            value="A",
            w=140,
        )
        self._url_limit = controls.child[dmc.NumberInput](
            label="Limit",
            value=10,
            min=1,
            max=100,
            step=1,
            w=120,
        )
        self._url_active = controls.child[dmc.Switch](
            label="Active",
            checked=False,
        )

        panel.child[dmc.Text](children="Current URL:", size="sm", mt="sm")
        panel.child[dmc.Code](id="url-search-display", block=True, children="(none)")

    def setup_callbacks(self, app) -> None:
        """Register demo callbacks."""

        # ── Download ──────────────────────────────────────────────────────────
        @app.callback(
            Output(self.dl_btn.download(), "data"),
            Output("dl-status", "children"),
            Input(self.dl_btn.button(), "n_clicks"),
            prevent_initial_call=True,
        )
        def _download(n):
            return dcc.send_string(
                "col_a,col_b\n1,2\n3,4\n", filename="demo.csv"
            ), "Downloaded demo.csv"

        # ── Timer tick display ────────────────────────────────────────────────
        @app.callback(
            Output("timer-tick-display", "children"),
            Input(self.timer.n_calls_store(), "data"),
        )
        def _tick(n):
            return f"Tick count: {n}"

        # ── Live update section ───────────────────────────────────────────────
        @app.callback(
            Output("live-value", "children"),
            Input(self.live_sec.timer.n_calls_store(), "data"),
        )
        def _live_tick(n):
            return f"Random value: {np.random.default_rng(n).integers(100)}"

        # ── Pager — seed n_items from demo store ──────────────────────────────
        @app.callback(
            Output(self.pager.n_items_store(), "data"),
            Input("pager-demo-n-items", "data"),
        )
        def _seed_pager(n):
            return n

        @app.callback(
            Output("pager-info", "children"),
            Input(self.pager.page_store(), "data"),
        )
        def _pager_info(data):
            import json
            return json.dumps(data, indent=2)

        # ── Cache monitor simulation ──────────────────────────────────────────
        @app.callback(
            Output("cache-interval", "disabled"),
            Output("cache-step", "data"),
            Input("cache-trigger", "n_clicks"),
            prevent_initial_call=True,
        )
        def _start_sim(n):
            return False, 0

        @app.callback(
            Output(self.cache.status_store(), "data"),
            Output("cache-interval", "disabled", allow_duplicate=True),
            Output("cache-step", "data", allow_duplicate=True),
            Input("cache-interval", "n_intervals"),
            Input("cache-step", "data"),
            prevent_initial_call=True,
        )
        def _sim_progress(_, step):
            step = (step or 0) + 1
            prog = min(step * 10, 100)
            done = prog >= 100
            return (
                {
                    "active": [
                        {
                            "filename": "large_dataset.fits",
                            "progress": prog,
                            "speed_bps": 512_000,
                            "eta_s": max(0, (100 - prog) / 10),
                        }
                    ],
                    "stats": f"Step {step}/10",
                },
                done,
                step,
            )

        # ── Surface plot ──────────────────────────────────────────────────────
        @app.callback(
            Output(self.surface.figure_data_store(), "data"),
            Input("surface-refresh", "n_clicks"),
        )
        def _refresh_surface(n):
            rng = np.random.default_rng(n or 0)
            image = rng.standard_normal((60, 80))
            return SurfacePlot.make_figure_data(image=image, title="Random noise")

        # ── URL state ─────────────────────────────────────────────────────────
        self.url_mgr.bind("dataset", self._url_select)
        self.url_mgr.bind("limit", self._url_limit)
        self.url_mgr.bind("active", self._url_active, prop="checked")

        app.clientside_callback(
            "function(s) { return s || '(empty)'; }",
            Output("url-search-display", "children"),
            Input(self.url_mgr.location(), "search"),
        )


# ── App factory ────────────────────────────────────────────────────────────────


def create_app() -> Dash:
    """Create and return the demo Dash application."""
    dash_app = Dash(
        __name__,
        suppress_callback_exceptions=True,
    )

    root = DemoApp()
    dash_app.layout = root.layout()
    root.register_callbacks(dash_app)
    return dash_app


if __name__ == "__main__":
    a = create_app()
    print("\nStarting tolteca_web demo")
    print("Visit:  http://localhost:8050/")
    print("Press Ctrl+C to stop\n")
    a.run(host="0.0.0.0", port=8050, debug=False)
