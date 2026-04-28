"""Multi-view data product viewer — pathname routing + standalone capability.

Each viewer is a reusable Template with two methods for composition:

* ``bind_to(mgr)``           — register widget ↔ state bindings
* ``setup_reactive(app, mgr)`` — register view-specific reactive callbacks

The composite app uses **pathname routing** (``dcc.Location.pathname``) to
toggle view visibility while a single ``UrlStateManager`` keeps all query
params in sync.  Each viewer also runs as a **standalone** Dash server via
the ``StandaloneViewer`` wrapper.

URL examples::

    Composite:    /array?obsnum=98765&array=a1400
    Standalone:   http://localhost:8053/?obsnum=98765&array=a1400

Run::

    cd tolteca_web

    # Composite (all views, pathname routing)
    uv run python examples/dataprod_viewer_demo.py                   # port 8051

    # Standalone (individual view on its own server)
    uv run python examples/dataprod_viewer_demo.py summary           # port 8052
    uv run python examples/dataprod_viewer_demo.py array             # port 8053
    uv run python examples/dataprod_viewer_demo.py pointing          # port 8054
"""

from __future__ import annotations

import sys

from dash import Dash, Input, Output, State, dcc, html
from dash_component_template import Template
from pydantic import BaseModel

import dash_mantine_components as dmc

from tolteca_web.common import UrlStateManager


# ── State models ──────────────────────────────────────────────────────────────
#
# Each view declares its own model (for standalone use).  The composite
# merges them into one flat model so a single UrlStateManager can handle
# all widgets and all query params.


class SummaryState(BaseModel):
    obsnum: int = 12345


class ArrayState(BaseModel):
    obsnum: int = 12345
    array: str = "a1100"
    column: str = "x"
    show_flagged: bool = True


class PointingState(BaseModel):
    obsnum: int = 12345
    frame: str = "altaz"
    trail_len: int = 50


class CompositeState(BaseModel):
    """Union of all view states for the composite app."""

    obsnum: int = 12345
    array: str = "a1100"
    column: str = "x"
    show_flagged: bool = True
    frame: str = "altaz"
    trail_len: int = 50


# ── Reusable view templates ──────────────────────────────────────────────────
#
# Each view builds its own layout and provides:
#   bind_to(mgr)            — declare which model fields map to which widgets
#   setup_reactive(app, mgr) — reactive callbacks that read state_store
#
# The view does NOT create its own UrlStateManager — the caller provides one.


class SummaryView(Template):
    """Observation summary — read-only, with cross-view links."""

    State = SummaryState

    def __init__(self) -> None:
        super().__init__()
        root = self.child[dmc.Stack](gap="sm", p="md")
        root.child[dmc.Title](children="Observation Summary", order=4)
        self._badge = root.child[dmc.Badge](
            children="—", size="xl", variant="light", color="blue",
        )
        self._info = root.child[dmc.Code](block=True, children="")

        # Cross-view navigation links
        links = root.child[dmc.Group](gap="md", mt="md")
        links.child[dmc.Text](children="Jump to:", size="sm", fw=500)
        self._array_link = links.child[dcc.Link](
            children="Array View", href="/array",
            style={"fontWeight": "500", "color": "var(--mantine-color-blue-6)"},
        )
        self._pointing_link = links.child[dcc.Link](
            children="Pointing View", href="/pointing",
            style={"fontWeight": "500", "color": "var(--mantine-color-blue-6)"},
        )

    def bind_to(self, mgr: UrlStateManager) -> None:  # noqa: ARG002
        pass  # No editable widgets — purely reactive

    def setup_reactive(self, app, mgr: UrlStateManager) -> None:
        app.clientside_callback(
            """function(state) {
                if (!state) return [
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update
                ];
                var n = state.obsnum;
                return [
                    'obsnum ' + n,
                    'Observation ' + n + '\\nStatus: complete\\nInstrument: TolTEC\\nType: Science',
                    '/array?obsnum=' + n,
                    '/pointing?obsnum=' + n,
                ];
            }""",
            Output(self._badge(), "children"),
            Output(self._info(), "children"),
            Output(self._array_link(), "href"),
            Output(self._pointing_link(), "href"),
            Input(mgr.state_store(), "data"),
        )


class ArrayView(Template):
    """Array property viewer — its own controls for array, column, flags."""

    State = ArrayState

    def __init__(self) -> None:
        super().__init__()
        root = self.child[dmc.Stack](gap="md", p="md")
        root.child[dmc.Title](children="Array Properties", order=4)

        controls = root.child[dmc.Group](gap="md", align="flex-end")
        self.array_select = controls.child[dmc.Select](
            label="Array", data=["a1100", "a1400", "a2000"],
            value="a1100", w=140,
        )
        self.column_select = controls.child[dmc.Select](
            label="Column", data=["x", "y", "amp", "fwhm", "s2n"],
            value="x", w=140,
        )
        self.flagged_switch = controls.child[dmc.Switch](
            label="Show flagged", checked=True,
        )

        self._plot = root.child[dmc.Paper](
            p="lg", withBorder=True, radius="sm",
        ).child[dmc.Code](block=True, children="(array plot)")

    def bind_to(self, mgr: UrlStateManager) -> None:
        mgr.bind("array", self.array_select)
        mgr.bind("column", self.column_select)
        mgr.bind("show_flagged", self.flagged_switch, prop="checked")

    def setup_reactive(self, app, mgr: UrlStateManager) -> None:
        app.clientside_callback(
            """function(state) {
                if (!state) return window.dash_clientside.no_update;
                var obs = state.obsnum, arr = state.array,
                    col = state.column, flagged = state.show_flagged;
                var lines = [
                    'Array Property Map',
                    '═'.repeat(30),
                    'obsnum:       ' + obs,
                    'array:        ' + arr,
                    'column:       ' + col,
                    'show_flagged: ' + flagged,
                    '═'.repeat(30),
                    'Detectors: ' + (arr==='a1100' ? '7938' : '3458'),
                    'Valid: ' + (flagged ? 'all' : 'unflagged only'),
                ];
                return lines.join('\\n');
            }""",
            Output(self._plot(), "children"),
            Input(mgr.state_store(), "data"),
        )


class PointingView(Template):
    """Telescope pointing viewer — coordinate frame and trail controls."""

    State = PointingState

    def __init__(self) -> None:
        super().__init__()
        root = self.child[dmc.Stack](gap="md", p="md")
        root.child[dmc.Title](children="Telescope Pointing", order=4)

        controls = root.child[dmc.Group](gap="md", align="flex-end")
        self.frame_select = controls.child[dmc.Select](
            label="Coord frame", data=["altaz", "icrs", "galactic"],
            value="altaz", w=160,
        )
        self.trail_input = controls.child[dmc.NumberInput](
            label="Trail length", value=50, min=10, max=500, step=10, w=140,
        )

        self._plot = root.child[dmc.Paper](
            p="lg", withBorder=True, radius="sm",
        ).child[dmc.Code](block=True, children="(pointing plot)")

    def bind_to(self, mgr: UrlStateManager) -> None:
        mgr.bind("frame", self.frame_select)
        mgr.bind("trail_len", self.trail_input)

    def setup_reactive(self, app, mgr: UrlStateManager) -> None:
        app.clientside_callback(
            """function(state) {
                if (!state) return window.dash_clientside.no_update;
                var obs = state.obsnum, frame = state.frame,
                    trail = state.trail_len;
                var labels = {altaz: 'Az / El', icrs: 'RA / Dec', galactic: 'l / b'};
                var lines = [
                    'Telescope Pointing Track',
                    '═'.repeat(30),
                    'obsnum:    ' + obs,
                    'frame:     ' + frame + ' (' + (labels[frame] || frame) + ')',
                    'trail_len: ' + trail + ' samples',
                    '═'.repeat(30),
                    'Pattern: Lissajous',
                    'Duration: ' + Math.round(trail * 0.2) + 's',
                ];
                return lines.join('\\n');
            }""",
            Output(self._plot(), "children"),
            Input(mgr.state_store(), "data"),
        )


# ── Standalone wrapper ────────────────────────────────────────────────────────
#
# Wraps any view Template with its own UrlStateManager so it can run as
# an independent Dash server.  The view's State model drives the URL params.


class StandaloneViewer(Template):
    """Generic standalone wrapper: UrlStateManager + obsnum control + view."""

    def __init__(self, view_cls: type[Template], *, title: str | None = None) -> None:
        super().__init__()
        self._view_cls = view_cls

        root = self.child[dmc.MantineProvider]()
        container = root.child[dmc.Container](maw=800, pt="lg")

        container.child[dmc.Title](
            children=title or view_cls.__name__, order=3, mb="xs",
        )
        container.child[dmc.Text](
            children="Running standalone — all state from URL query params.",
            c="dimmed", size="sm", mb="md",
        )

        self.mgr = UrlStateManager(view_cls.State, show_debug=True)
        container.child(self.mgr)

        # Shared obsnum control
        shared = container.child[dmc.Group](gap="md", align="flex-end", mb="md")
        self._obsnum = shared.child[dmc.NumberInput](
            label="Observation Number", value=12345, min=1, w=200,
        )

        self.view = view_cls()
        container.child(self.view)

    def setup_callbacks(self, app) -> None:
        self.mgr.bind("obsnum", self._obsnum)
        self.view.bind_to(self.mgr)
        self.view.setup_reactive(app, self.mgr)


# ── Composite app (pathname routing) ─────────────────────────────────────────
#
# All three views are always in the DOM; a clientside callback reads
# ``dcc.Location.pathname`` to toggle which one is visible.
# One UrlStateManager with a flat CompositeState handles all query params.
# Nav links are ``dcc.Link`` with dynamically generated hrefs that carry
# the current search string, so navigating between pages preserves state.


class DataProdViewer(Template):
    """Multi-view composite with pathname routing and URL query state.

    Routes::

        /             → Summary (default)
        /summary      → Summary
        /array        → Array
        /pointing     → Pointing

    Query params sync bidirectionally with all widgets regardless of
    which page is active.
    """

    def __init__(self) -> None:
        super().__init__()

        root = self.child[dmc.MantineProvider]()
        container = root.child[dmc.Container](maw=800, pt="lg")

        # ── Header ──
        container.child[dmc.Title](
            children="Data Product Viewer", order=3, mb="xs",
        )
        container.child[dmc.Text](
            children=(
                "Pathname routing — each view is a page. "
                "Every view also runs standalone on its own port."
            ),
            c="dimmed", size="sm", mb="md",
        )

        # ── State manager (flat model, one for the whole composite) ──
        self.mgr = UrlStateManager(CompositeState, show_debug=True)
        container.child(self.mgr)

        # ── Shared controls ──
        shared = container.child[dmc.Group](gap="md", align="flex-end", mb="md")
        self._obsnum = shared.child[dmc.NumberInput](
            label="Observation Number", value=12345, min=1, w=200,
        )

        # ── Nav bar: dcc.Link nodes with dynamic hrefs ──
        nav = container.child[dmc.Paper](
            withBorder=True, radius="sm", p="xs", mb="sm",
        )
        nav_group = nav.child[dmc.Group](gap="xs")

        self._nav_links: dict[str, object] = {}
        self._nav_labels: dict[str, object] = {}
        for page_id, label, href in [
            ("summary", "Summary", "/"),
            ("array", "Array", "/array"),
            ("pointing", "Pointing", "/pointing"),
        ]:
            link = nav_group.child[dcc.Link](href=href, style={"textDecoration": "none"})
            text = link.child[dmc.Badge](
                children=label, variant="light", size="lg",
                style={"cursor": "pointer"},
            )
            self._nav_links[page_id] = link
            self._nav_labels[page_id] = text

        # ── URL display bar ──
        url_row = container.child[dmc.Group](gap="xs", mb="xs")
        url_row.child[dmc.Text](children="URL:", size="sm", fw=500)
        self._url_display = url_row.child[dmc.Code](
            children="", style={"flex": "1"},
        )

        # ── Standalone link panel ──
        sp = container.child[dmc.Paper](
            withBorder=True, radius="sm", p="sm", mb="md", bg="gray.0",
        )
        sp.child[dmc.Text](
            children="Standalone server URLs (same view, own port):",
            size="xs", fw=500, mb="xs",
        )
        self._standalone_links = sp.child[dmc.Code](block=True, children="")

        # ── View containers (always mounted, visibility toggled) ──
        self._divs: dict[str, object] = {}

        self.summary = SummaryView()
        self._divs["summary"] = container.child[html.Div]()
        self._divs["summary"].child(self.summary)

        self.array = ArrayView()
        self._divs["array"] = container.child[html.Div](style={"display": "none"})
        self._divs["array"].child(self.array)

        self.pointing = PointingView()
        self._divs["pointing"] = container.child[html.Div](style={"display": "none"})
        self._divs["pointing"].child(self.pointing)

    def setup_callbacks(self, app) -> None:
        mgr = self.mgr

        # ── Shared binding ──
        mgr.bind("obsnum", self._obsnum)

        # ── Per-view bindings + reactive ──
        for view in [self.summary, self.array, self.pointing]:
            view.bind_to(mgr)
            view.setup_reactive(app, mgr)

        # ── Pathname routing: toggle view visibility ──
        app.clientside_callback(
            """function(pathname) {
                var p = (pathname || '/').replace(/^\\//, '');
                if (!p) p = 'summary';
                return [
                    p === 'summary'  ? {} : {display: 'none'},
                    p === 'array'    ? {} : {display: 'none'},
                    p === 'pointing' ? {} : {display: 'none'},
                ];
            }""",
            Output(self._divs["summary"](), "style"),
            Output(self._divs["array"](), "style"),
            Output(self._divs["pointing"](), "style"),
            Input(mgr.location(), "pathname"),
        )

        # ── Nav link hrefs: carry current search params ──
        app.clientside_callback(
            """function(search) {
                var s = search || '';
                return ['/' + s, '/array' + s, '/pointing' + s];
            }""",
            Output(self._nav_links["summary"](), "href"),
            Output(self._nav_links["array"](), "href"),
            Output(self._nav_links["pointing"](), "href"),
            Input(mgr.location(), "search"),
        )

        # ── Active nav highlighting ──
        app.clientside_callback(
            """function(pathname) {
                var p = (pathname || '/').replace(/^\\//, '');
                if (!p) p = 'summary';
                return [
                    p === 'summary'  ? 'filled' : 'light',
                    p === 'array'    ? 'filled' : 'light',
                    p === 'pointing' ? 'filled' : 'light',
                ];
            }""",
            Output(self._nav_labels["summary"](), "variant"),
            Output(self._nav_labels["array"](), "variant"),
            Output(self._nav_labels["pointing"](), "variant"),
            Input(mgr.location(), "pathname"),
        )

        # ── URL display ──
        app.clientside_callback(
            """function(pathname, search) {
                return (pathname || '') + (search || '') || '(none)';
            }""",
            Output(self._url_display(), "children"),
            Input(mgr.location(), "pathname"),
            Input(mgr.location(), "search"),
        )

        # ── Standalone links panel ──
        app.clientside_callback(
            """function(state) {
                if (!state) return window.dash_clientside.no_update;
                var n = state.obsnum;
                var lines = [
                    'Summary:  http://localhost:8052/?obsnum=' + n,
                    'Array:    http://localhost:8053/?obsnum=' + n
                              + '&array=' + state.array
                              + '&column=' + state.column,
                    'Pointing: http://localhost:8054/?obsnum=' + n
                              + '&frame=' + state.frame
                              + '&trail_len=' + state.trail_len,
                ];
                return lines.join('\\n');
            }""",
            Output(self._standalone_links(), "children"),
            Input(mgr.state_store(), "data"),
        )


# ── Entry points ──────────────────────────────────────────────────────────────


def create_composite_app() -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True)
    root = DataProdViewer()
    app.layout = root.layout()
    root.register_callbacks(app)
    return app


def create_standalone_app(
    view_cls: type[Template], title: str,
) -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True)
    root = StandaloneViewer(view_cls, title=title)
    app.layout = root.layout()
    root.register_callbacks(app)
    return app


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "composite"

    factories = {
        "composite": (lambda: create_composite_app(), 8051),
        "summary": (lambda: create_standalone_app(SummaryView, "Summary Viewer"), 8052),
        "array": (lambda: create_standalone_app(ArrayView, "Array Viewer"), 8053),
        "pointing": (lambda: create_standalone_app(PointingView, "Pointing Viewer"), 8054),
    }

    if mode not in factories:
        print(f"Usage: python {sys.argv[0]} [{' | '.join(factories)}]")
        sys.exit(1)

    factory, port = factories[mode]
    a = factory()

    print(f"\nData Product Viewer — {mode} mode")
    print(f"Visit: http://localhost:{port}/")
    if mode == "composite":
        print(f"Try:   http://localhost:{port}/array?obsnum=98765&array=a1400")
    print("Press Ctrl+C to stop\n")

    a.run(host="0.0.0.0", port=port, debug=False)
