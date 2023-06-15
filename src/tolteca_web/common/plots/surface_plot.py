from dash_component_template import ComponentTemplate
from dash import html, dcc, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
from tollan.utils.log import logger


class SurfacePlot(ComponentTemplate):
    """A component template to generate 2d surface plot."""

    _range_slider_defaults = (-99, 99)

    class Meta:  # noqa: D106
        component_cls = html.Div

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        container = self
        self._store = container.child(dcc.Store)
        hist_container, range_container, graph_container = container.colgrid(3, 1)
        hist_container.width = 10
        hist_container.parent.className = "justify-content-center"
        range_container.width = 10
        range_container.parent.className = "justify-content-center"
        self._hist_graph = hist_container.child(dcc.Loading, type="circle").child(
            dcc.Graph,
            config={
                "displayModeBar": False,
            },
        )
        self._range_slider = range_container.child(
            dcc.RangeSlider,
            min=self._range_slider_defaults[0],
            max=self._range_slider_defaults[1],
            value=self._range_slider_defaults,
            allowCross=False,
            tickformat=".1f",
        )
        self._graph = graph_container.child(dcc.Graph)

    def setup_layout(self, app):
        """Set up layout."""
        hist_graph = self._hist_graph
        range_slider = self._range_slider
        graph = self._graph
        super().setup_layout(app)

        app.clientside_callback(
            """
            function (data) {
                if (data === null) {
                    return Array(2).fill(window.dash_clientside.no_update)
                }
                return [data.hist_fig, data.fig]
            }
            """,
            [
                Output(hist_graph.id, "figure"),
                Output(graph.id, "figure"),
            ],
            [
                Input(self._store.id, "data"),
            ],
            prevent_initial_call=True,
        )
        app.clientside_callback(
            """
            function (data) {
                if (data === null) {
                    return Array(2).fill(window.dash_clientside.no_update)
                }
                return [data.vmin_min, data.vmax_max]
            }
            """,
            [
                Output(range_slider.id, "min"),
                Output(range_slider.id, "max"),
            ],
            [
                Input(self._store.id, "data"),
            ],
            prevent_initial_call=True,
        )

    def make_figure_data(  # noqa: PLR0913
            self,
            data,
            hist_data=None,
            title=None,
            size_max=1000000,
            x_label=None,
            y_label=None,
            value_range=None,
            xaxis_range=None,
            yaxis_range=None,
            axis_font=None,
            n_bins=50,
            animation_frame=None,
            image_height=400,
            image_width=None,
            marker_size=None,
            **kwargs,
    ):
        """Generate figure data.

        Data can be one of the two forms:
        1. 2-d array. This is treated as 2d image data.
        2. DataFrame or dict of arrays. This is treated as scatter data. It expects
          arrays of keys x, y, z, and more others that keyword arguments may
          refer to.
        """

        def is_scatter_data():
            return isinstance(data, (dict, pd.DataFrame))

        if hist_data is None:
            # create hist_data from data
            hist_data = data["z"] if is_scatter_data() else data.flatten()
        vmed = np.nanmedian(hist_data)
        vnmin = np.nanmin(hist_data)
        vnmax = np.nanmax(hist_data)
        # vmin_min, vmax_max = sorted([vnmin, vnmax])
        vmin_min, vmax_max = sorted([0.1 * vmed, 2.5 * vmed])
        if tuple(value_range) == self._range_slider_defaults:
            vmin, vmax = None, None
        else:
            vmin, vmax = value_range
        vmin = vmin or vmin_min
        vmax = vmax or vmax_max
        logger.debug(
            f"{vmin_min=} {vmin=} {vmax=} {vmax_max=}",
        )
        # Generate the histogram figure
        hfig = go.Figure()
        mask = ~np.isnan(hist_data)
        h, bins = np.histogram(hist_data[mask], bins=n_bins, range=[vmin, vmax])
        bincenters = 0.5 * (bins[1:] + bins[:-1])
        hfig = go.Figure()
        hfig.add_trace(go.Bar(x=bincenters, y=h, name=""))
        hfig.update_layout(plot_bgcolor="white")
        hfig.update_yaxes(visible=False, showticklabels=False)
        hfig.update_layout(
            height=60,
            margin=go.layout.Margin(
                l=10,
                r=10,
                b=5,
                t=40,
            ),
            title={
                "text": title or "Unnamed Plot",
                "x": 0.5,
                "font": {"size": 11},
            },
            font=axis_font,
        )

        if is_scatter_data():
            # scatter data
            imfig = px.scatter(
                data_frame=data,
                x="x",
                y="y",
                color="z",
                range_color=[vmin, vmax],
                animation_frame=animation_frame,
                **kwargs,
            )
            imfig.update_traces(marker=dict(size=marker_size))
        else:
            # image data
            bs = data.size > size_max

            imfig = px.imshow(
                data,
                zmin=vmin,
                zmax=vmax,
                binary_string=bs,
                origin="lower",
                **kwargs,
            )
        imfig.update_layout(
            uirevision=True,
            showlegend=False,
            autosize=True,
            plot_bgcolor="white",
            font=axis_font,
        )

        imfig.update_coloraxes(colorbar_thickness=5)
        imfig.update_xaxes(title=x_label or "x")
        imfig.update_yaxes(title=y_label or "y")
        if(xaxis_range is not None):
            imfig.update_xaxes(range=xaxis_range)
        if(yaxis_range is not None):
            imfig.update_yaxes(range=yaxis_range)
        imfig.update_layout(
            height=image_height,
            width=image_width,
            margin=go.layout.Margin(
                l=10,
                r=10,
                b=20,
                t=0,
            ),
        )

        return {
            "hist_fig": hfig,
            "fig": imfig,
            "vmin_min": vmin_min,
            "vmax_max": vmax_max,
        }

    @property
    def figure_data(self):
        """The figure data store."""
        return self._store

    @property
    def component_inputs(self):
        """The controls inputs."""
        return {
            "value_range": Input(self._range_slider.id, "value"),
        }

    @property
    def component_output(self):
        """The figure data store used for output."""
        return Output(self._store.id, "data")
