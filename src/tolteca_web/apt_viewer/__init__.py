import os

import dash_bootstrap_components as dbc
from dash import html
from dash_component_template import ComponentTemplate, NullComponent
from tollan.utils.log import logger
from dash import dcc, Input, Output
import numpy as np

from ..common import LabeledDropdown
from ..common.plots.surface_plot import SurfacePlot


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
                className="mb-2",
            )
        ).dropdown
        surface_plot = views_panel.child(SurfacePlot())
        btn = controls_panel.child(dbc.Button, "Click Me")

        surface_plot_anim = views_panel.child(SurfacePlot())
        btn_anim = controls_panel.child(dbc.Button, "Click Me Animation")

        super().setup_layout(app)

        @app.callback(
            output=surface_plot.component_output,
            inputs=[Input(btn.id, "n_clicks"), surface_plot.component_inputs],
        )
        def make_random_image(n_clicks, sp_inputs):
            image_data = np.random.normal(1, 1.0, (100, 100))
            return surface_plot.make_figure_data(
                image_data,
                title="Random data",
                **sp_inputs,
            )

        @app.callback(
            output=surface_plot_anim.component_output,
            inputs=[Input(btn_anim.id, "n_clicks"), surface_plot_anim.component_inputs],
        )
        def make_random_anim(n_clicks, sp_inputs):
            x, y, z, t = [], [], [], []
            for ii in range(10):
                xx, yy, zz = np.random.normal(1, 1.0, (3, ii * 10))
                x.append(xx)
                y.append(yy)
                z.append(zz)
                t.append(np.full(xx.shape, ii))
            return surface_plot.make_figure_data(
                {
                    "x": np.hstack(x),
                    "y": np.hstack(y),
                    "z": np.hstack(z),
                    "t": np.hstack(t),
                },
                title="Random anim",
                animation_frame="t",
                **sp_inputs,
            )


DASHA_SITE = {
    "dasha": {
        "template": AptViewer,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", False),
    }
}
