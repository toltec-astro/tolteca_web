"""
To Do:
 - Spinner's aren't working when images are loading.
 - Amplitude slider and histogram for beam image isn't working.
 - Amplitude labels for individual array plots need significant digits truncated
 - Code is weirdly spread between this file, ToltecAPTDiagnostics, and ToltecBeammapFits
 - Move all image production from callback routines to stand-alone external functions
   to improve portability and cross-use.
"""

from ..toltec_dp_utils.ToltecAptDiagnostics import ToltecAptDiagnostics
from ..toltec_dp_utils.ToltecAptDiagnostics import histogramNetworksPlotly
from ..toltec_dp_utils.ToltecBeammapFits import ToltecBeammapFits
from dash_component_template import ComponentTemplate
from dash.dependencies import Input, Output, State
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
from dash import dash_table
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
from ..base import ViewerBase


class ToltecAptViewer(ViewerBase):
    class Meta:
        component_cls = dbc.Container

    def __init__(
        self,
        title_text="Toltec APT Viewer",
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

        # cheat
        self.aptd = None
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
        dPath = "/Users/wilson/Desktop/tmp/102518/redu07/"
        g = glob("{}apt*.ecsv".format(dPath))
        apts = g
        aptOptions = [{"label": p, "value": p} for p in apts]

        # pull down to select obs stats file
        controls_panel, cuts_panel, bigBox = body.grid(3, 1)
        aptSelectBox = controls_panel.child(dbc.Row).child(dbc.Col, width=12)
        aptSelectRow = aptSelectBox.child(dbc.Row)
        aptSelectCol = aptSelectRow.child(dbc.Col, width=6)
        aptTitle = aptSelectCol.child(dbc.Row).child(
            html.H5, "APT Choice", className="mb-2"
        )
        aptList = aptSelectCol.child(dbc.Row).child(
            dcc.Dropdown,
            options=aptOptions,
            placeholder="Select APT",
            value="",
            searchable=False,
            style=dict(width="100%", verticalAlign="middle"),
        )
        self.state_manager.register("aptList", aptList, ("options", "value"))

        colSelectCol = aptSelectRow.child(dbc.Col, width=4)
        colTitle = colSelectCol.child(dbc.Row).child(
            html.H5, "APT Value to Plot", className="mb-2"
        )
        colList = colSelectCol.child(dbc.Row).child(
            dcc.Dropdown,
            placeholder="Select APT Column",
            value="sens",
            searchable=False,
            style=dict(width="100%", verticalAlign="middle"),
        )

        newFileStore = aptSelectRow.child(dbc.Col, width=1).child(dcc.Store)

        # Define the cuts
        cuts_panel.child(dbc.Row).child(html.Br)
        cutsRow = cuts_panel.child(dbc.Row)
        cutsRow.child(dbc.Col, width=1).child(html.Div, "APT Cuts:")
        unflaggedCol = cutsRow.child(dbc.Col, width=1)
        unflaggedSel = unflaggedCol.child(
            daq.ToggleSwitch, size=30, value=True, label=["all", "unflagged"]
        )
        cutsRow.child(dbc.Col, width=1).child(html.Div, "")
        s2nCol = cutsRow.child(dbc.Col, width=1)
        s2nSel = s2nCol.child(
            daq.ToggleSwitch, size=30, value=False, label=["all", "s2n>10"]
        )
        cutsRow.child(dbc.Col, width=2).child(html.Div, "")
        bmFileAvailable = cutsRow.child(dbc.Col, width=2).child(
            html.Div, "Beammap File: None"
        )
        bmCutoutCol = cutsRow.child(dbc.Col, width=2)
        bmCutoutSel = bmCutoutCol.child(
            daq.ToggleSwitch,
            size=30,
            value=False,
            label=["full beammap", "source cutout"],
        )
        bmRawCol = cutsRow.child(dbc.Col, width=2)
        bmRawSel = bmRawCol.child(
            daq.ToggleSwitch, size=30, value=False, label=["raw map", "convolved"]
        )
        beammap = {
            "files available": bmFileAvailable,
            "cutout": bmCutoutSel,
            "convolved": bmRawSel,
        }

        foo = cutsRow.child(dbc.Col, width=1).child(html.Div, "")
        cuts = {
            "unflagged": unflaggedSel,
            "s2n10": s2nSel,
        }

        # Put in a break
        bigBox.child(dbc.Row).child(html.Br)
        bigBox.child(dbc.Row).child(html.Hr)

        # The three images side by side
        infoBox = bigBox.child(dbc.Row)
        figureBox = infoBox.child(dbc.Col, width=7)
        a11Col, a14Col, a20Col = figureBox.colgrid(1, 3)
        a11_plot = a11Col.child(SurfacePlot())
        a14_plot = a14Col.child(SurfacePlot())
        a20_plot = a20Col.child(SurfacePlot())
        a11_hist = a11Col.child(dbc.Row).child(dcc.Graph)
        a14_hist = a14Col.child(dbc.Row).child(dcc.Graph)
        a20_hist = a20Col.child(dbc.Row).child(dcc.Graph)
        threeStat = {
            "a1100": {"plot": a11_plot, "hist": a11_hist},
            "a1400": {"plot": a14_plot, "hist": a14_hist},
            "a2000": {"plot": a20_plot, "hist": a20_hist},
        }

        # A summary table and beam map image
        tableBox = infoBox.child(dbc.Col, width=5)
        beamRow, tableRow = tableBox.grid(2, 1)
        table = tableRow.child(dcc.Graph)
        beam_plot = beamRow.child(SurfacePlot())
        beammap["plot"] = beam_plot

        # Some potential useful diagnostic plots
        fwhmxCol, fwhmyCol = bigBox.colgrid(1, 2)
        fwhmxPlot = fwhmxCol.child(dbc.Row).child(dcc.Graph)
        fwhmyPlot = fwhmyCol.child(dbc.Row).child(dcc.Graph)
        sensCol, _ = bigBox.colgrid(1, 2)
        sensPlot = sensCol.child(dbc.Row).child(dcc.Graph)
        staticPlots = {
            "fwhmx": fwhmxPlot,
            "fwhmy": fwhmyPlot,
            "sens": sensPlot,
        }

        super().setup_layout(app)

        self._registerCallbacks(
            app,
            aptList,
            colList,
            cuts,
            threeStat,
            staticPlots,
            table,
            newFileStore,
            beammap,
        )

    def _registerCallbacks(
        self,
        app,
        aptList,
        colList,
        cuts,
        threeStat,
        staticPlots,
        table,
        newFileStore,
        beammap,
    ):

        # ---------------------------
        # colList dropdown
        # ---------------------------
        @app.callback(
            [
                Output(colList.id, "options"),
                Output(beammap["files available"].id, "children"),
                Output(newFileStore.id, "data"),
            ],
            [
                Input(aptList.id, "value"),
                Input(newFileStore.id, "data"),
            ],
            # prevent_initial_call=True,
        )
        def columnListDropdown(aptFile, newFile):
            if (aptFile == "") | (aptFile is None):
                raise PreventUpdate
            if (newFile == "") | (newFile is None):
                newFile = 1
            self.aptd = ToltecAptDiagnostics(aptFile=aptFile)
            newFile += 1
            options = []
            for k in self.aptd.getPlottableKeys():
                options.append({"label": k, "value": k})

            # Collect beammap files if available
            g = glob(os.path.dirname(aptFile) + "/toltec*.fits")
            if len(g) > 0:
                bmFileAvailable = "Beammap Files Available"
                self.beammapFiles = g
            else:
                bmFileAvailable = "No Beammap Files Available"
                self.beammapFiles = None
            return [options, bmFileAvailable, newFile]

        # ---------------------------
        # aptList dropdown
        # ---------------------------
        @app.callback(
            [
                threeStat["a1100"]["plot"].component_output,
                threeStat["a1400"]["plot"].component_output,
                threeStat["a2000"]["plot"].component_output,
                Output(threeStat["a1100"]["hist"].id, "figure"),
                Output(threeStat["a1400"]["hist"].id, "figure"),
                Output(threeStat["a2000"]["hist"].id, "figure"),
            ],
            [
                Input(newFileStore.id, "data"),
                Input(colList.id, "value"),
                Input(cuts["unflagged"].id, "value"),
                Input(cuts["s2n10"].id, "value"),
                threeStat["a1100"]["plot"].component_inputs,
                threeStat["a1400"]["plot"].component_inputs,
                threeStat["a2000"]["plot"].component_inputs,
            ],
        )
        def primaryDropdown(newFile, aptCol, unflagged, s2n10, *sp_inputs_list):
            if (newFile == "") | (newFile is None) | (aptCol == "") | (aptCol is None):
                aptd = None
                empty = True
            else:
                aptd = self.aptd
                empty = False
            outputs = makeThreeStatPlots(
                threeStat,
                aptCol,
                unflagged,
                s2n10,
                *sp_inputs_list,
                empty=empty,
                aptd=aptd,
            )
            histFigs = makeThreeStatHists(
                aptCol,
                unflagged,
                s2n10,
                empty=empty,
                aptd=aptd,
            )
            return outputs + histFigs

        # ---------------------------
        # static plots
        # ---------------------------
        @app.callback(
            [
                Output(staticPlots["fwhmx"].id, "figure"),
                Output(staticPlots["fwhmy"].id, "figure"),
                Output(staticPlots["sens"].id, "figure"),
            ],
            [
                Input(newFileStore.id, "data"),
                Input(cuts["unflagged"].id, "value"),
                Input(cuts["s2n10"].id, "value"),
            ],
        )
        def makeStaticPlots(newFile, unflagged, s2n10):
            if (newFile == "") | (newFile is None):
                return makeEmptyFigs(3)

            # apply apt cuts here
            cutapt = self.aptd.apt.copy()
            if unflagged:
                cutapt = self.aptd.cullByColumn(
                    "flag", apt=cutapt, valueMax=0, verbose=False
                )
            if s2n10:
                cutapt = self.aptd.cullByColumn(
                    "sig2noise", apt=cutapt, valueMin=10.0, verbose=False
                )
            fwhmFigs = self.aptd.plotFWHMvsPos(apt=cutapt, returnPlotlyFig=True)
            sensFig = self.aptd.plotSensvsFreq(apt=cutapt, returnPlotlyFig=True)
            xaxis, yaxis = getXYAxisLayouts()
            figList = fwhmFigs + [sensFig]
            for fig in figList:
                fig.update_layout(height=200, xaxis=xaxis, yaxis=yaxis)
            return figList

        # ---------------------------
        # Summary Table
        # ---------------------------
        @app.callback(
            [Output(table.id, "figure")],
            [
                Input(newFileStore.id, "data"),
                Input(cuts["unflagged"].id, "value"),
                Input(cuts["s2n10"].id, "value"),
            ],
        )
        def makeTable(newFile, unflagged, s2n10):
            if (newFile == "") | (newFile is None):
                return getTableFig(unflagged, s2n10, empty=True)
            return getTableFig(unflagged, s2n10, aptd=self.aptd)

        # ---------------------------
        # Beammap Controls
        # ---------------------------
        @app.callback(
            [
                Output(beammap["cutout"].id, "disabled"),
                Output(beammap["convolved"].id, "disabled"),
            ],
            [
                Input(beammap["files available"].id, "children"),
            ],
            prevent_initial_call=False,
        )
        def bmControls(bmAvailable):
            if bmAvailable != "Beammap Files Available":
                return [True, True]
            return [False, False]

        # ---------------------------
        # Clickable Plots
        # ---------------------------
        @app.callback(
            [
                beammap["plot"].component_output,
            ],
            [
                Input(threeStat["a1100"]["plot"].graph.id, "clickData"),
                Input(threeStat["a1400"]["plot"].graph.id, "clickData"),
                Input(threeStat["a2000"]["plot"].graph.id, "clickData"),
                State(beammap["files available"].id, "children"),
                State(beammap["cutout"].id, "value"),
                State(beammap["convolved"].id, "value"),
                beammap["plot"].component_inputs,
            ],
        )
        def makeBeammapPlot(
            a11Data, a14Data, a20Data, bmAvailable, cutout, convolved, *sp_inputs_list
        ):
            # No sense doing anything if no beammap files
            if bmAvailable != "Beammap Files Available":
                return makeEmptySurfacePlot(beammap["plot"], *sp_inputs_list)

            # Determine which plot was clicked
            if "surfaceplot0" in ctx.triggered[0]["prop_id"]:
                array = "a1100"
                arrayNum = 0
            elif "surfaceplot1" in ctx.triggered[0]["prop_id"]:
                array = "a1400"
                arrayNum = 1
            elif "surfaceplot2" in ctx.triggered[0]["prop_id"]:
                array = "a2000"
                arrayNum = 2
            else:
                print("Uh oh, bug in array selection for clickData")
                raise PreventUpdate

            # Make sure we have the file for the array selected
            file = [i for i in self.beammapFiles if array in i]
            if len(file) < 1:
                print("No file for that beammap, try another.")
                raise PreventUpdate

            # get the apt
            apt = self.aptd.apt

            # which detector? Must match x,y location since cuts screw up order
            clickIndex = ctx.triggered[0]["value"]["points"][0]["pointNumber"]
            clickX = ctx.triggered[0]["value"]["points"][0]["x"]
            clickY = ctx.triggered[0]["value"]["points"][0]["y"]
            w = np.where(
                (apt["x_t"] == clickX)
                & (apt["y_t"] == clickY)
                & (apt["array"] == arrayNum)
            )[0]
            if len(w) < 1:
                print("Somehow we're not finding the clicked detector.")
                raise PreventUpdate
            detID = w[0]

            # now we can fetch the beammap if it's available
            path = os.path.dirname(file[0])
            bm = getBeammap(path, array)
            image = bm.getPlotlyImage(detID, cutout=cutout, convolved=convolved) * 1.0e7
            outputs = []
            outputs.append(
                beammap["plot"].make_figure_data(
                    image,
                    title="Beammap: Det {}".format(detID),
                    x_label=None,
                    y_label=None,
                    xaxis_range=None,
                    yaxis_range=None,
                    axis_font={
                        "size": 8,
                    },
                    image_height=400,
                    image_width=None,
                    marker_size=3,
                    **sp_inputs_list[0],
                )
            )
            return outputs


def makeThreeStatHists(
    aptCol,
    unflagged,
    s2n10,
    empty=False,
    aptd=None,
):
    if not empty:
        # apply apt cuts here
        cutapt = aptd.apt.copy()
        if unflagged:
            cutapt = aptd.cullByColumn("flag", apt=cutapt, valueMax=0, verbose=False)
        if s2n10:
            cutapt = aptd.cullByColumn(
                "sig2noise", apt=cutapt, valueMin=10.0, verbose=False
            )
        # build the histograms
        histFigs = []
        for i in range(3):
            fig = histogramNetworksPlotly(cutapt, i, aptCol)
            histFigs.append(fig)
    else:
        histFigs = makeEmptyFigs(3)
    for fig in histFigs:
        fig.update_layout(
            width=275,
            height=275,
            font={
                "size": 8,
            },
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=8),
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.90,
            ),
            margin=go.layout.Margin(
                l=10,
                r=10,
                b=5,
                t=40,
            ),
        )
    return histFigs


def makeThreeStatPlots(
    threeStat, aptCol, unflagged, s2n10, *sp_inputs_list, aptd=None, empty=False
):
    if not empty:
        # apply apt cuts here
        cutapt = aptd.apt.copy()
        if unflagged:
            cutapt = aptd.cullByColumn("flag", apt=cutapt, valueMax=0, verbose=False)
        if s2n10:
            cutapt = aptd.cullByColumn(
                "sig2noise", apt=cutapt, valueMin=10.0, verbose=False
            )

        # build the plots
        outputs = []
        for sp_inputs, array in zip(sp_inputs_list, ["a1100", "a1400", "a2000"]):
            apt = aptd.getArrayApt(array, apt=cutapt)
            bigList = []
            for i in range(len(apt)):
                bigList.append([apt["x_t"][i], apt["y_t"][i], apt[aptCol][i]])
            nDets = len(bigList)
            data = pd.DataFrame(bigList, columns=["x", "y", "z"])
            units = apt.meta[aptCol][0].replace("units: ", "")
            label = "{0:} ({1:} dets)<br>{2:} [{3:}]".format(
                array, nDets, aptCol, units
            )
            if array == "a1100":
                ylabel = "y_t [arcsec]"
            else:
                ylabel = " "
            outputs.append(
                threeStat[array]["plot"].make_figure_data(
                    data,
                    title=label,
                    x_label="x_t [arcsec]",
                    y_label=ylabel,
                    xaxis_range=[-150, 150],
                    yaxis_range=[-150, 150],
                    axis_font={
                        "size": 8,
                    },
                    image_height=200,
                    image_width=250,
                    marker_size=3,
                    **sp_inputs,
                )
            )
    else:
        outputs = []
        for sp_inputs, array in zip(sp_inputs_list, ["a1100", "a1400", "a2000"]):
            outputs.append(
                threeStat[array]["plot"].make_figure_data(
                    np.zeros((200, 200)),
                    title="",
                    axis_font={
                        "size": 8,
                    },
                    image_height=200,
                    image_width=250,
                    marker_size=3,
                    **sp_inputs,
                )
            )
    return outputs


def getTableFig(unflagged, s2n10, empty=False, aptd=None):
    cells = [["<b>Total</b>", "<b>Not cut</b>", "<b>Med sens [mJyrts]</b>"]]
    if not empty:
        # apply apt cuts here
        apt = aptd.apt
        cutapt = apt.copy()
        if unflagged:
            cutapt = aptd.cullByColumn("flag", apt=cutapt, valueMax=0, verbose=False)
        if s2n10:
            cutapt = aptd.cullByColumn(
                "sig2noise", apt=cutapt, valueMin=10.0, verbose=False
            )

        # table values
        for i, array in enumerate(["a1100", "a1400", "a2000"]):
            a = cutapt[cutapt["array"] == i]
            cells.append(
                [len(apt[apt["array"] == i]), len(a), int(np.median(a["sens"]))]
            )
    else:
        for i, array in enumerate(["a1100", "a1400", "a2000"]):
            cells.append(["-", "-", "-"])

    # build the table
    fig = go.Figure()
    fig.add_trace(
        go.Table(
            header=dict(
                values=["", "<b>a1100</b>", "<b>a1400</b>", "<b>a2000</b>"],
                line_color="darkslategray",
                fill_color="orange",
                align="center",
            ),
            cells=dict(
                values=cells,
                line_color="darkslategray",
                fill_color="white",
                align="center",
            ),
            columnwidth=[0.4, 0.2, 0.2, 0.2],
        )
    )
    fig.update_layout(
        autosize=False,
        height=200,
        margin=go.layout.Margin(
            l=20,
            r=20,
            b=5,
            t=5,
            pad=0,
        ),
    )
    return [fig]


def makeEmptySurfacePlot(sp, *sp_inputs_list):
    outputs = []
    outputs.append(
        sp.make_figure_data(
            np.zeros((200, 200)),
            title="",
            x_label=None,
            y_label=None,
            xaxis_range=None,
            yaxis_range=None,
            axis_font={
                "size": 8,
            },
            image_height=200,
            image_width=None,
            marker_size=3,
            **sp_inputs_list[0],
        ),
    )
    return outputs


def makeEmptyFigs(nfigs):
    figs = []
    xaxis, yaxis = getXYAxisLayouts()
    for i in range(nfigs):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
            )
        )
        fig.update_layout(
            xaxis=xaxis,
            yaxis=yaxis,
        )
        figs.append(fig)
    return figs


@functools.lru_cache()
def getBeammap(path, array):
    return ToltecBeammapFits(path=path + "/", array=array)


# common figure axis definitions
def getXYAxisLayouts():
    xaxis = dict(
        titlefont=dict(size=8),
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor="black",
        linewidth=2,
        ticks="outside",
        tickfont=dict(
            family="Arial",
            size=8,
            color="rgb(82, 82, 82)",
        ),
    )

    yaxis = dict(
        titlefont=dict(size=8),
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor="black",
        linewidth=2,
        ticks="outside",
        tickfont=dict(
            family="Arial",
            size=8,
            color="rgb(82, 82, 82)",
        ),
    )
    return xaxis, yaxis


DASHA_SITE = {
    "dasha": {
        "template": ToltecAptViewer,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", True),
    }
}
