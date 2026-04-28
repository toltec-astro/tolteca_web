"""Standalone URL state manager demo.

Run::

    cd tolteca_web
    uv run python examples/url_state_demo.py

Then open http://localhost:8060/ and try navigating to::

    /?dataset=C&limit=75&active=true
"""

from __future__ import annotations

from dash import Dash, Input, Output, State, dcc
from dash_component_template import Template
from pydantic import BaseModel

import dash_mantine_components as dmc

from tolteca_web.common import UrlStateManager


# ── State model ────────────────────────────────────────────────────────────────


class PageState(BaseModel):
    dataset: str = "A"
    limit: int = 10
    active: bool = False


# ── App template ───────────────────────────────────────────────────────────────


class UrlStateDemo(Template):
    """Minimal demo of UrlStateManager with three bound widgets."""

    def __init__(self) -> None:
        super().__init__()

        root = self.child[dmc.MantineProvider]().child[dmc.Container](
            maw=600, pt="lg"
        )
        root.child[dmc.Title](children="UrlStateManager Demo", order=3, mb="sm")
        root.child[dmc.Text](
            children="Change a widget → URL updates. Navigate to a URL → widgets update.",
            c="dimmed",
            size="sm",
            mb="md",
        )

        self.url_mgr = UrlStateManager(PageState, show_debug=True)
        root.child(self.url_mgr)

        controls = root.child[dmc.Group](gap="md", align="flex-end", mb="sm")
        self._select = controls.child[dmc.Select](
            label="Dataset",
            data=["A", "B", "C", "D"],
            value="A",
            w=140,
        )
        self._limit = controls.child[dmc.NumberInput](
            label="Limit",
            value=10,
            min=1,
            max=100,
            step=1,
            w=120,
        )
        self._active = controls.child[dmc.Switch](
            label="Active",
            checked=False,
        )

        root.child[dmc.Text](children="Current URL search:", size="sm", mt="md")
        self._url_display = root.child[dmc.Code](block=True, children="(empty)")

        # ── External source: button that bumps limit by 10 ────────────
        ext_group = root.child[dmc.Group](gap="md", align="center", mt="md")
        ext_group.child[dmc.Text](
            children="External source test:", size="sm", fw=500,
        )
        self._bump_btn = ext_group.child[dmc.Button](
            children="Bump limit +10",
            variant="outline",
            size="sm",
        )

    def setup_callbacks(self, app) -> None:
        self.url_mgr.bind("dataset", self._select)
        self.url_mgr.bind("limit", self._limit)
        self.url_mgr.bind("active", self._active, prop="checked")

        app.clientside_callback(
            "function(s) { return s || '(empty)'; }",
            Output(self._url_display(), "children"),
            Input(self.url_mgr.location(), "search"),
        )

        # External source: button patches state_store directly
        app.clientside_callback(
            """function(n, state) {
                if (!n || !state) return window.dash_clientside.no_update;
                var s = Object.assign({}, state);
                s.limit = Math.min((s.limit || 0) + 10, 100);
                return s;
            }""",
            Output(self.url_mgr.state_store(), "data", allow_duplicate=True),
            Input(self._bump_btn(), "n_clicks"),
            State(self.url_mgr.state_store(), "data"),
            prevent_initial_call=True,
        )


# ── Entry point ────────────────────────────────────────────────────────────────


def create_app() -> Dash:
    """Create the standalone demo app."""
    app = Dash(__name__, suppress_callback_exceptions=True)
    root = UrlStateDemo()
    app.layout = root.layout()
    root.register_callbacks(app)
    return app


if __name__ == "__main__":
    a = create_app()
    print("\nURL State Manager Demo")
    print("Visit:  http://localhost:8051/")
    print("Try:    http://localhost:8051/?dataset=C&limit=75&active=true")
    print("Press Ctrl+C to stop\n")
    a.run(host="0.0.0.0", port=8051, debug=False)
