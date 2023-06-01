#!/usr/bin/env python

import dash_bootstrap_components as dbc
from dash import ClientsideFunction, Input, Output, State, html
from dash_component_template import ComponentTemplate
from tollan.utils.general import rupdate

__all__ = [
    "CollapseContent",
]


class CollapseContent(ComponentTemplate):
    class Meta:
        component_cls = html.Div

    def __init__(self, button_text, button_props=None, content=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.button_text = button_text
        _button_props = {
            "className": "me-2 my-0 px-2 shadow-none",
            "color": "link",
            "style": {
                "border-bottom-width": "0px",
            },
        }
        rupdate(_button_props, button_props or dict())
        self._button = self.child(dbc.Button, self.button_text, **_button_props)
        self._content = content or self.child(dbc.Collapse)

    def setup_layout(self, app):
        super().setup_layout(app)

        app.clientside_callback(
            ClientsideFunction(
                namespace="ui",
                function_name="toggleWithClick",
            ),
            output=Output(self._content.id, "is_open"),
            inputs=[
                Input(self._button.id, "n_clicks"),
                State(self._content.id, "is_open"),
            ],
        )

    @property
    def content(self):
        """The content component."""
        return self._content

    @property
    def button(self):
        """The button component."""
        return self._button
