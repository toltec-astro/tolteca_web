#!/usr/bin/env python

import dash_bootstrap_components as dbc
from dash import ClientsideFunction, Input, Output, State, dcc, html
from dash_component_template import ComponentTemplate
from tollan.utils.general import rupdate

from .utils import fa

__all__ = [
    "DownloadButton",
]


class DownloadButton(ComponentTemplate):
    class Meta:
        component_cls = html.Div

    def __init__(
        self,
        button_text,
        button_props=None,
        tooltip=None,
        download=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.button_text = button_text
        _button_props = {"size": "sm"}
        rupdate(_button_props, button_props or dict())
        self._button = self.child(
            dbc.Button,
            [fa("fa fa-download", className="me-2"), self.button_text],
            **_button_props,
        )
        self._download = download or self.child(dcc.Download)
        self._tooltip = tooltip

    def setup_layout(self, app):
        if self._tooltip is not None:
            self.child(dbc.Tooltip, self._tooltip, target=self._button.id)
        super().setup_layout(app)

    @property
    def download(self):
        """The download component."""
        return self._download

    @property
    def button(self):
        """The button component."""
        return self._button
