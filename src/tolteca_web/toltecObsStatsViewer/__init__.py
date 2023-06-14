import sys

# sys.path.append("/Users/wilson/GitHub/toltec-data-product-utilities/toltec_dp_utils/")
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
            self.to.printKeys()
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
                        title=plotTitle,
                        animation_frame="Chunk",
                        **sp_inputs,
                    )
                )
            return outputs

        # @app.callback(
        #     [
        #         Output(threeStat["a1100"]["histPlot"].id, "figure"),
        #         Output(threeStat["a1100"]["imagePlot"].id, "figure"),
        #         Output(threeStat["a1100"]["histRange"].id, "min"),
        #         Output(threeStat["a1100"]["histRange"].id, "max"),
        #         Output(threeStat["a1400"]["histPlot"].id, "figure"),
        #         Output(threeStat["a1400"]["imagePlot"].id, "figure"),
        #         Output(threeStat["a1400"]["histRange"].id, "min"),
        #         Output(threeStat["a1400"]["histRange"].id, "max"),
        #         Output(threeStat["a2000"]["histPlot"].id, "figure"),
        #         Output(threeStat["a2000"]["imagePlot"].id, "figure"),
        #         Output(threeStat["a2000"]["histRange"].id, "min"),
        #         Output(threeStat["a2000"]["histRange"].id, "max"),
        #     ],
        #     [
        #         Input(mainControls["stat"].id, "value"),
        #         Input(threeStat["a1100"]["histRange"].id, "value"),
        #         Input(threeStat["a1400"]["histRange"].id, "value"),
        #         Input(threeStat["a2000"]["histRange"].id, "value"),
        #     ],
        #     prevent_initial_call=True,
        # )
        # def threePanelFig(stat, a11Range, a14Range, a20Range):
        #     if self.to is None:
        #         raise PreventUpdate
        #     if (stat is None) | (stat == ""):
        #         raise PreventUpdate

        #     a1100 = getFancyFig(self.to, stat, "a1100", a11Range)
        #     a1400 = getFancyFig(self.to, stat, "a1400", a14Range)
        #     a2000 = getFancyFig(self.to, stat, "a2000", a20Range)

        #     return a1100 + a1400 + a2000

        # ---------------------------
        # Array Views
        # ---------------------------
        # @app.callback(
        #     [
        #         Output(arrayView["a1100"]["histPlot"].id, "figure"),
        #         Output(arrayView["a1100"]["imagePlot"].id, "figure"),
        #         Output(arrayView["a1100"]["histRange"].id, "min"),
        #         Output(arrayView["a1100"]["histRange"].id, "max"),
        #         Output(arrayView["a1400"]["histPlot"].id, "figure"),
        #         Output(arrayView["a1400"]["imagePlot"].id, "figure"),
        #         Output(arrayView["a1400"]["histRange"].id, "min"),
        #         Output(arrayView["a1400"]["histRange"].id, "max"),
        #         Output(arrayView["a2000"]["histPlot"].id, "figure"),
        #         Output(arrayView["a2000"]["imagePlot"].id, "figure"),
        #         Output(arrayView["a2000"]["histRange"].id, "min"),
        #         Output(arrayView["a2000"]["histRange"].id, "max"),
        #     ],
        #     [
        #         Input(mainControls["stat"].id, "value"),
        #         Input(arrayView["a1100"]["histRange"].id, "value"),
        #         Input(arrayView["a1400"]["histRange"].id, "value"),
        #         Input(arrayView["a2000"]["histRange"].id, "value"),
        #     ],
        #     prevent_initial_call=True,
        # )
        # def threePanelFig(stat, a11Range, a14Range, a20Range):
        #     if self.to is None:
        #         raise PreventUpdate
        #     if (stat is None) | (stat == ""):
        #         raise PreventUpdate

        #     a1100 = getAVAnimation(self.to, stat, "a1100", a11Range)
        #     a1400 = getAVAnimation(self.to, stat, "a1400", a14Range)
        #     a2000 = getAVAnimation(self.to, stat, "a2000", a20Range)

        #     return a1100 + a1400 + a2000


# Returns the containers and controls for a fancy figure
# def getFancyFigContainer(box):
#     # The top row has the histogram and controls
#     histRow = box.child(dbc.Row)
#     foo = histRow.child(dbc.Col, width=1)
#     histPlot = (
#         histRow.child(dbc.Col, width=10)
#         .child(dcc.Loading, type="circle")
#         .child(dcc.Graph)
#     )
#     rangeRow = box.child(dbc.Row)
#     bar = rangeRow.child(dbc.Col, width=1)
#     histRange = rangeRow.child(dbc.Col, width=10).child(
#         dcc.RangeSlider,
#         min=-99,
#         max=99,
#         value=[-99, 99],
#         allowCross=False,
#         tickformat=".2f",
#     )
#     imagePlot = box.child(dbc.Row).child(dcc.Graph)
#     ffContainer = {"histPlot": histPlot, "histRange": histRange, "imagePlot": imagePlot}
#     return ffContainer


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


# def getFancyFig(to, name, array, ranges):
#     # fetch the image and units
#     image, units = getImage(to, name, array=array)
#     signal = image.flatten()
#     w = np.where(signal != 0.0)[0]
#     signal = signal[w]
#     s = image.shape

#     # Deal with ranges
#     statmin = 0.1 * np.median(signal)
#     statmax = 2.5 * np.median(signal)
#     if ranges[0] == -99:
#         ranges[0] = min(statmin, statmax)
#     if ranges[1] == 99:
#         ranges[1] = max(statmin, statmax)

#     # Generate the histogram figure
#     h, bins = np.histogram(signal, bins=50, range=ranges)
#     bincenters = np.array([(bins[i] + bins[i + 1]) / 2.0 for i in range(len(bins) - 1)])
#     hfig = go.Figure()
#     hfig.add_trace(go.Bar(x=bincenters, y=h, name=""))
#     hfig.update_layout(plot_bgcolor="white")
#     hfig.update_yaxes(visible=False, showticklabels=False)
#     margin = go.layout.Margin(
#         l=10,  # left margin
#         r=10,  # right margin
#         b=5,  # bottom margin
#         t=40,  # top margin
#     )
#     plotTitle = "{0:}: {1:}".format(array, name)
#     hfig.update_layout(
#         margin=margin,
#         height=60,
#         title={
#             "text": plotTitle,
#             "x": 0.5,
#         },
#     )

#     # Generate the image figure
#     if (image.shape[0] > 2000) | (image.shape[1] > 2000):
#         bs = True
#     else:
#         bs = False

#     if bs:
#         imfig = px.imshow(
#             image, zmin=ranges[0], zmax=ranges[1], binary_string=bs, origin="lower"
#         )
#     else:
#         imfig = go.Figure()
#         imfig.add_trace(
#             go.Heatmap(
#                 z=image,
#                 zmin=ranges[0],
#                 zmax=ranges[1],
#             )
#         )

#     imfig.update_layout(
#         uirevision=True, showlegend=False, autosize=True, plot_bgcolor="white"
#     )
#     imfig.update_xaxes(title="Detector Index")
#     imfig.update_yaxes(title="Data Chunk Index")
#     margin = go.layout.Margin(
#         l=10,  # left margin
#         r=10,  # right margin
#         b=20,  # bottom margin
#         t=0,  # top margin
#     )
#     imfig.update_layout(margin=margin, height=400)
#     return [hfig, imfig, int(statmin), int(statmax)]


# def getAVAnimation(to, name, array, ranges, chunk=None):
#     # fetch the image and units
#     image, units = getImage(to, name, array=array)
#     signal = image.flatten()
#     w = np.where(signal != 0.0)[0]
#     signal = signal[w]
#     s = image.shape

#     # detector positions
#     xt, xtu = getChunk(to, "apt_x_t", chunk=None, array=array)
#     yt, ytu = getChunk(to, "apt_y_t", chunk=None, array=array)

#     # Deal with ranges
#     statmin = 0.1 * np.median(signal)
#     statmax = 2.5 * np.median(signal)
#     if ranges[0] == -99:
#         ranges[0] = min(statmin, statmax)
#     if ranges[1] == 99:
#         ranges[1] = max(statmin, statmax)

#     # Generate the histogram figure
#     h, bins = np.histogram(signal, bins=50, range=ranges)
#     bincenters = np.array([(bins[i] + bins[i + 1]) / 2.0 for i in range(len(bins) - 1)])
#     hfig = go.Figure()
#     hfig.add_trace(go.Bar(x=bincenters, y=h, name=""))
#     hfig.update_layout(plot_bgcolor="white")
#     hfig.update_yaxes(visible=False, showticklabels=False)
#     margin = go.layout.Margin(
#         l=10,  # left margin
#         r=10,  # right margin
#         b=5,  # bottom margin
#         t=40,  # top margin
#     )
#     plotTitle = "{0:}: {1:}".format(array, name)
#     hfig.update_layout(
#         margin=margin,
#         height=60,
#         title={
#             "text": plotTitle,
#             "x": 0.5,
#         },
#     )

#     # Set up pandas dataframe with the data
#     bigList = []
#     for i in range(to.nChunks):
#         for j in range(len(xt)):
#             bigList.append([i, xt[j], yt[j], image[i, j]])
#     df = pd.DataFrame(bigList, columns=["Chunk", "x_t", "y_t", "value"])

#     imfig = px.scatter(
#         df, x="x_t", y="y_t", color="value", range_color=ranges, animation_frame="Chunk"
#     )
#     imfig.update_layout(
#         uirevision=True, showlegend=False, autosize=True, plot_bgcolor="white"
#     )
#     imfig.update_xaxes(title="x_t [arcsec]")
#     imfig.update_yaxes(title="y_t [arcsec]")
#     margin = go.layout.Margin(
#         l=10,  # left margin
#         r=10,  # right margin
#         b=20,  # bottom margin
#         t=0,  # top margin
#     )
#     imfig.update_layout(margin=margin, height=400)
#     imfig.update_xaxes(range=[-130, 130])
#     imfig.update_yaxes(range=[-130, 130])
#     return [hfig, imfig, int(statmin), int(statmax)]


# common figure axis definitions
def getXYAxisLayouts():
    xaxis = dict(
        titlefont=dict(size=20),
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor="black",
        linewidth=4,
        ticks="outside",
        tickfont=dict(
            family="Arial",
            size=18,
            color="rgb(82, 82, 82)",
        ),
    )

    yaxis = dict(
        titlefont=dict(size=20),
        scaleanchor="x",
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor="black",
        linewidth=4,
        ticks="outside",
        tickfont=dict(
            family="Arial",
            size=18,
            color="rgb(82, 82, 82)",
        ),
    )
    return xaxis, yaxis


DASHA_SITE = {
    "dasha": {
        "template": ToltecObsStatsViewer,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", True),
    }
}
