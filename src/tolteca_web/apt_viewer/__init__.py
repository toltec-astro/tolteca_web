import os

import dash_bootstrap_components as dbc
from dash import html
from dash_component_template import ComponentTemplate, NullComponent
from tollan.utils.log import logger

from ..common import LabeledDropdown


class AptViewer(ComponentTemplate):
    class Meta:
        component_cls = dbc.Container

    def __init__(
        self, title_text="APT Viewer", subtitle_text="(test version)", **kwargs
    ):
        kwargs.setdefault("fluid", True)
        super().__init__(**kwargs)
        self._title_text = title_text
        self._subtitle_text = subtitle_text
        self.fluid = True

    def setup_layout(self, app):
        container = self
        header, body = container.grid(2, 1)
        # Header
        title_container = header.child(
            html.Div, className="d-flex align-items-baseline"
        )
        title_container.child(html.H2(self._title_text, className="my-2"))
        if self._subtitle_text is not None:
            title_container.child(
                html.P(self._subtitle_text, className="text-secondary mx-2")
            )
        controls_panel, views_panel = body.grid(2, 1)
        # pull down to select apt file
        apt_select = controls_panel.child(
            LabeledDropdown(
                label_text="APT file",
                # className='w-auto',
                size="sm",
                placeholder="Select a APT file ...",
            )
        ).dropdown


DASHA_SITE = {
    "dasha": {
        "template": AptViewer,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", False),
    }
}
