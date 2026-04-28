"""Collapsible content panel template."""

from __future__ import annotations

from dash import Input, Output, State, html
from dash_component_template import Template

import dash_mantine_components as dmc

__all__ = ["CollapseContent"]


class CollapseContent(Template):
    """A toggle button + collapsible content panel.

    Clicking the button opens/closes the contained content.
    Uses a clientside callback for zero server round-trips.

    Parameters
    ----------
    button_text : str or Dash component
        Content for the toggle button.
    button_props : dict, optional
        Extra props passed to `dmc.Button`.
    opened : bool, optional
        Initial open state, by default False.

    Examples
    --------
    >>> from dash import html
    >>> panel = CollapseContent("Settings")
    >>> panel.content.child[html.P](children="Inner content")
    """

    def __init__(
        self,
        button_text: str | object = "Toggle",
        button_props: dict | None = None,
        *,
        opened: bool = False,
    ) -> None:
        super().__init__()
        self.button_text = button_text
        _button_props = {
            "variant": "subtle",
            "size": "compact-sm",
        }
        if button_props:
            _button_props.update(button_props)

        self.button = self.child[dmc.Button](children=button_text, **_button_props)
        self.content = self.child[dmc.Collapse](opened=opened)

    def setup_callbacks(self, app) -> None:
        """Wire toggle button to collapse via clientside callback."""
        app.clientside_callback(
            """
            function(n, is_open) {
                if (n) return !is_open;
                return is_open;
            }
            """,
            Output(self.content(), "opened"),
            Input(self.button(), "n_clicks"),
            State(self.content(), "opened"),
        )
