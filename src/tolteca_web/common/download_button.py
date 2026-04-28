"""Download button template."""

from __future__ import annotations

from dash import dcc, html
from dash_component_template import Template
from dash_iconify import DashIconify

import dash_mantine_components as dmc

__all__ = ["DownloadButton"]


class DownloadButton(Template):
    """A button pre-wired to a `dcc.Download` component.

    Renders a `dmc.Button` with a download icon. The caller connects
    `button.n_clicks` → `dcc.send_*` → `download.data`.

    Parameters
    ----------
    button_text : str
        Label shown on the button.
    button_props : dict, optional
        Extra props for `dmc.Button`.
    tooltip : str, optional
        Tooltip text shown on hover.

    Attributes
    ----------
    button : dmc.Button node
    download : dcc.Download node

    Examples
    --------
    >>> btn = DownloadButton("Export CSV")
    >>> # In a callback: return dcc.send_data_frame(df.to_csv, "data.csv")
    >>> # Output(btn.download(), "data")
    """

    def __init__(
        self,
        button_text: str = "Download",
        button_props: dict | None = None,
        tooltip: str | None = None,
    ) -> None:
        super().__init__()
        _button_props: dict = {"size": "sm", "variant": "light"}
        if button_props:
            _button_props.update(button_props)

        # Always wrap in Tooltip; disable it when no text is given so the hover
        # element is still in the tree but invisible.
        tip = self.child[dmc.Tooltip](
            label=tooltip or "",
            disabled=tooltip is None,
            withinPortal=True,
        )
        self.button = tip.child[dmc.Button](
            children=button_text,
            leftSection=DashIconify(icon="mdi:download", width=16),
            **_button_props,
        )
        self.download = self.child[dcc.Download]()
