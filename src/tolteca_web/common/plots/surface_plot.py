"""Surface plot template."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc
from dash_component_template import Template

import dash_mantine_components as dmc

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["SurfacePlot"]


class SurfacePlot(Template):
    """2-D image/scatter visualization with histogram and range slider.

    The caller builds figure data server-side via :meth:`make_figure_data`
    and stores it in `figure_data_store.data`. Two clientside callbacks then
    update the main graph and histogram without an extra server round-trip.

    Attributes
    ----------
    figure_data_store : dcc.Store node — write figure data here

    Examples
    --------
    >>> sp = SurfacePlot()
    >>> data = sp.make_figure_data(image=my_2d_array)
    >>> # In a callback: return data  → Output(sp.figure_data_store(), "data")
    """

    def __init__(self) -> None:
        super().__init__()
        self.figure_data_store = self.child[dcc.Store](data={})

        outer = self.child[dmc.Stack](gap="xs")

        top_row = outer.child[dmc.Group](gap="xs", align="flex-start", wrap="nowrap")

        # Main image graph
        self.graph = top_row.child[dcc.Graph](
            style={"flex": "1 1 auto"},
            config={"scrollZoom": True},
        )

        # Histogram panel
        self.hist_graph = top_row.child[dcc.Graph](
            style={"width": "120px", "flex": "0 0 auto"},
            config={"staticPlot": True},
        )

        # Range slider for vmin/vmax
        self.value_range = outer.child[dcc.RangeSlider](
            min=0,
            max=1,
            step=0.01,
            value=[0, 1],
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
        )

    def setup_callbacks(self, app) -> None:
        """Wire figure_data_store → graphs + range slider via clientside."""
        app.clientside_callback(
            """
            function(data) {
                if (!data || !data.fig) return [window.dash_clientside.no_update, window.dash_clientside.no_update];
                return [data.hist_fig || {}, data.fig];
            }
            """,
            Output(self.hist_graph(), "figure"),
            Output(self.graph(), "figure", allow_duplicate=True),
            Input(self.figure_data_store(), "data"),
            prevent_initial_call="initial_duplicate",
        )

        app.clientside_callback(
            """
            function(data) {
                if (!data) return [window.dash_clientside.no_update, window.dash_clientside.no_update, window.dash_clientside.no_update];
                var vmin = data.vmin_min !== undefined ? data.vmin_min : 0;
                var vmax = data.vmax_max !== undefined ? data.vmax_max : 1;
                return [vmin, vmax, [vmin, vmax]];
            }
            """,
            Output(self.value_range(), "min"),
            Output(self.value_range(), "max"),
            Output(self.value_range(), "value"),
            Input(self.figure_data_store(), "data"),
        )

        # Update heatmap zmin/zmax when slider moves
        app.clientside_callback(
            """
            function(range_value, figure) {
                if (!figure || !figure.data || !figure.data.length) return window.dash_clientside.no_update;
                if (!range_value || range_value.length < 2) return window.dash_clientside.no_update;
                var vmin = range_value[0];
                var vmax = range_value[1];
                var fig = JSON.parse(JSON.stringify(figure));
                var trace = fig.data[0];
                if (trace.type === 'heatmap') {
                    trace.zmin = vmin;
                    trace.zmax = vmax;
                } else if (trace.marker) {
                    trace.marker.cmin = vmin;
                    trace.marker.cmax = vmax;
                }
                return fig;
            }
            """,
            Output(self.graph(), "figure", allow_duplicate=True),
            Input(self.value_range(), "value"),
            State(self.graph(), "figure"),
            prevent_initial_call=True,
        )

    @staticmethod
    def make_figure_data(
        image: np.ndarray | None = None,
        scatter_data: dict | None = None,
        colorscale: str = "Viridis",
        axis_labels: dict | None = None,
        title: str = "",
    ) -> dict[str, Any]:
        """Build serializable figure data for `figure_data_store`.

        Pass exactly one of *image* or *scatter_data*.

        Parameters
        ----------
        image : np.ndarray, optional
            2-D numpy array for heatmap display.
        scatter_data : dict, optional
            Dict with ``"x"``, ``"y"``, ``"z"`` arrays for scatter.
        colorscale : str, optional
            Plotly colorscale name. Default ``"Viridis"``.
        axis_labels : dict, optional
            Keys ``"x"`` and/or ``"y"`` for axis titles.
        title : str, optional
            Figure title.

        Returns
        -------
        dict
            Serializable dict to store in ``figure_data_store.data``.
        """
        axis_labels = axis_labels or {}

        if image is not None:
            z = image
        elif scatter_data is not None:
            z = np.asarray(scatter_data["z"])
        else:
            z = np.zeros((10, 10))

        vmin = float(np.nanmin(z))
        vmax = float(np.nanmax(z))

        # Main figure
        if image is not None:
            trace = go.Heatmap(
                z=z,
                colorscale=colorscale,
                zmin=vmin,
                zmax=vmax,
                showscale=False,
            )
        else:
            assert scatter_data is not None
            trace = go.Scatter(
                x=scatter_data["x"],
                y=scatter_data["y"],
                mode="markers",
                marker={
                    "color": z,
                    "colorscale": colorscale,
                    "cmin": vmin,
                    "cmax": vmax,
                    "showscale": False,
                },
            )

        fig = go.Figure(data=[trace])
        fig.update_layout(
            title=title,
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis_title=axis_labels.get("x", ""),
            yaxis_title=axis_labels.get("y", ""),
            uirevision="fixed",
        )

        # Histogram figure
        flat = z.ravel()
        flat = flat[np.isfinite(flat)]
        hist_fig = go.Figure(
            data=[
                go.Histogram(
                    x=flat,
                    nbinsx=50,
                    marker_color="steelblue",
                )
            ]
        )
        hist_fig.update_layout(
            margin={"l": 20, "r": 5, "t": 5, "b": 30},
            xaxis_title="",
            yaxis_title="",
            showlegend=False,
            bargap=0.02,
        )

        return {
            "fig": fig.to_dict(),
            "hist_fig": hist_fig.to_dict(),
            "vmin_min": vmin,
            "vmax_max": vmax,
        }
