from ..toltec_dp_utils.ToltecObsStats import ToltecObsStats
from dash_component_template import ComponentTemplate
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from tollan.utils.log import logger
from astropy.nddata import Cutout2D
from ..common import LabeledInput
from ..common.plots.surface_plot import SurfacePlot
from astropy import units as u
import plotly.graph_objs as go
import plotly.express as px
import dash_daq as daq
from dash import html
from glob import glob
from dash import dcc
from dash import ctx
import pandas as pd
import numpy as np
import functools
import time
import os


class ToltecObsStatsViewer(ComponentTemplate):
    class Meta:
        component_cls = dbc.Container

    def __init__(
        self,
        title_text="Toltec Observation Statistics Viewer",
        subtitle_text="(test version)",
        **kwargs,
    ):
        kwargs.setdefault("fluid", True)
        super().__init__(**kwargs)
        self._title_text = title_text
        self._subtitle_text = subtitle_text
        self.fluid = True

    def setup_layout(self, app):
        container = self
        header, body = container.grid(2, 1)

        # Again ... cheating
        self.to = None

        # Header
        title_container = header.child(
            html.Div, className="d-flex align-items-baseline"
        )
        title_container.child(html.H2(self._title_text, className="my-2"))
        if self._subtitle_text is not None:
            title_container.child(
                html.P(self._subtitle_text, className="text-secondary mx-2")
            )

        # Hard code the input path for testing.
        dPath = "/Users/wilson/Desktop/tmp/macs0717/pointing/102384/"
        g = glob("{}*stats.nc".format(dPath))
        gp = glob("/Users/wilson/Desktop/tmp/102518/redu*/*stats.nc")
        g = g + gp
        paths = [os.path.dirname(gg) for gg in g]
        paths.append("./")
        pathOptions = [{"label": p, "value": p} for p in paths]

        # pull down to select obs stats file
        controls_panel, views_panel, bigBox = body.grid(3, 1)
        controlBox = controls_panel.child(dbc.Row).child(dbc.Col, width=10)
        settingsRow = controlBox.child(dbc.Row)
        statsCol = settingsRow.child(dbc.Col, width=6)
        statsTitle = statsCol.child(dbc.Row).child(
            html.H5, "Stats File Path", className="mb-2"
        )
        statsList = statsCol.child(dbc.Row).child(
            dcc.Dropdown,
            options=pathOptions,
            placeholder="Select Path",
            value="",
            searchable=False,
            style=dict(width="100%", verticalAlign="middle"),
        )

        # Stat Choice controls
        settingsRow.child(dbc.Col, width=1)
        mcCol = settingsRow.child(dbc.Col, width=3)
        mcTitle = mcCol.child(dbc.Row).child(
            html.H5, "Statistic Choice", className="mb-2"
        )
        statChoiceCol = mcCol.child(dbc.Row)
        statChoice = statChoiceCol.child(
            dcc.Dropdown,
            value="",
            searchable=False,
            style=dict(width="100%", verticalAlign="middle"),
        )
        mainControls = {
            "stat": statChoice,
        }

        # Put in a break
        bigBox.child(dbc.Row).child(html.Br)

        # The three images side by side
        a11Col, a14Col, a20Col = bigBox.colgrid(1, 3)
        a11_plot = a11Col.child(SurfacePlot())
        a14_plot = a14Col.child(SurfacePlot())
        a20_plot = a20Col.child(SurfacePlot())

        threeStat = {
            "a1100": a11_plot,
            "a1400": a14_plot,
            "a2000": a20_plot,
        }

        # Another break
        bigBox.child(dbc.Row).child(html.Br)

        # The array view
        a11ViewCol, a14ViewCol, a20ViewCol = bigBox.colgrid(1, 3)
        a11_view = a11ViewCol.child(SurfacePlot())
        a14_view = a14ViewCol.child(SurfacePlot())
        a20_view = a20ViewCol.child(SurfacePlot())

        arrayView = {
            "a1100": a11_view,
            "a1400": a14_view,
            "a2000": a20_view,
        }
        super().setup_layout(app)

        self._registerCallbacks(
            app,
            statsList,
            threeStat,
            mainControls,
            arrayView,
        )

    def _registerCallbacks(
        self,
        app,
        statsList,
        threeStat,
        mainControls,
        arrayView,
    ):
        # ---------------------------
        # StatsList dropdown
        # ---------------------------
        @app.callback(
            [
                Output(mainControls["stat"].id, "options"),
                Output(mainControls["stat"].id, "value"),
            ],
            [
                Input(statsList.id, "value"),
            ],
            prevent_initial_call=True,
        )
        def primaryDropdown(path):
            if (path == "") | (path is None):
                raise PreventUpdate
            self.to = ToltecObsStats(path=path)
            keys = [k for k in self.to.keys if ("apt_" not in k) & ("obsnum" not in k)]
            options = []
            for k in keys:
                options.append({"label": k, "value": k})
            return [options, None]

        # ---------------------------
        # Big three panel fig
        # ---------------------------
        @app.callback(
            [
                threeStat["a1100"].component_output,
                threeStat["a1400"].component_output,
                threeStat["a2000"].component_output,
            ],
            [
                Input(mainControls["stat"].id, "value"),
                threeStat["a1100"].component_inputs,
                threeStat["a1400"].component_inputs,
                threeStat["a2000"].component_inputs,
            ],
            prevent_initial_call=True,
        )
        def threePanelFig(stat, *sp_inputs_list):
            if self.to is None:
                raise PreventUpdate
            if (stat is None) or (stat == ""):
                raise PreventUpdate
            outputs = []
            for sp_inputs, array in zip(sp_inputs_list, ["a1100", "a1400", "a2000"]):
                image_data, _ = getImage(self.to, stat, array=array)
                plotTitle = "{0:}: {1:}".format(array, stat)
                outputs.append(
                    threeStat[array].make_figure_data(
                        image_data,
                        title=plotTitle,
                        aspect="auto",
                        **sp_inputs,
                    )
                )
            return outputs

        @app.callback(
            [
                arrayView["a1100"].component_output,
                arrayView["a1400"].component_output,
                arrayView["a2000"].component_output,
            ],
            [
                Input(mainControls["stat"].id, "value"),
                arrayView["a1100"].component_inputs,
                arrayView["a1400"].component_inputs,
                arrayView["a2000"].component_inputs,
            ],
            prevent_initial_call=True,
        )
        def threePanelFig2(stat, *sp_inputs_list):
            if self.to is None:
                raise PreventUpdate
            if (stat is None) or (stat == ""):
                raise PreventUpdate
            outputs = []
            for sp_inputs, array in zip(sp_inputs_list, ["a1100", "a1400", "a2000"]):
                to = self.to
                image_data, _ = getImage(to, stat, array=array)
                # detector positions
                xt, xtu = getChunk(to, "apt_x_t", chunk=None, array=array)
                yt, ytu = getChunk(to, "apt_y_t", chunk=None, array=array)

                # Set up pandas dataframe with the data
                bigList = []
                for i in range(to.nChunks):
                    for j in range(len(xt)):
                        bigList.append([i, xt[j], yt[j], image_data[i, j]])
                data = pd.DataFrame(bigList, columns=["Chunk", "x", "y", "z"])
                plotTitle = "{0:}: {1:}".format(array, stat)
                outputs.append(
                    threeStat[array].make_figure_data(
                        data,
                        x_label = "x_t [arcsec]",
                        y_label = "y_t [arcsec]",
                        xaxis_range = [-150, 150],
                        yaxis_range = [-150, 150],
                        title=plotTitle,
                        animation_frame="Chunk",
                        **sp_inputs,
                    )
                )
            return outputs


@functools.lru_cache()
def getImage(to, name, array="a1100"):
    image = to.getStat(name, array=array, unflagged=True)
    units = to.getStatUnits(name)
    return image, units


@functools.lru_cache()
def getChunk(to, name, chunk, array="a1100"):
    stat = to.getStat(name, array=array, chunk=chunk, unflagged=True)
    units = to.getStatUnits(name)
    return stat, units


# common figure axis definitions
def getXYAxisLayouts():
    xaxis = dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor="black",
        linewidth=2,
        ticks="outside",
        tickfont=dict(
            family="Arial",
            size=12,
            color="rgb(82, 82, 82)",
        ),
        title={
            "text": "",
            "font": {"size": 12, "color": "black"}
        },
    )

    yaxis = dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor="black",
        linewidth=2,
        ticks="outside",
        tickfont=dict(
            family="Arial",
            size=12,
            color="rgb(82, 82, 82)",
        ),
        title={
            "text": "",
            "font": {"size": 12, "color": "black"}
        },
    )
    return xaxis, yaxis


DASHA_SITE = {
    "dasha": {
        "template": ToltecObsStatsViewer,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", True),
    }
}
