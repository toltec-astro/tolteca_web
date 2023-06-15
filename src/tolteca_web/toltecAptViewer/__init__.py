from ..toltec_dp_utils.ToltecAptDiagnostics import ToltecAptDiagnostics
from ..toltec_dp_utils.ToltecAptDiagnostics import histogramNetworksPlotly
from ..toltec_dp_utils.ToltecBeammapFits import ToltecBeammapFits
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


class ToltecAptViewer(ComponentTemplate):
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

        # Define the cuts
        cuts_panel.child(dbc.Row).child(html.Br)
        cutsRow = cuts_panel.child(dbc.Row)
        cutsRow.child(dbc.Col, width=1).child(html.Div, "APT Cuts:")
        unflaggedCol = cutsRow.child(dbc.Col, width=1)
        unflaggedSel = unflaggedCol.child(daq.ToggleSwitch, size=30, value=True,
                                          label=["all", "unflagged"])
        cutsRow.child(dbc.Col, width=1).child(html.Div, "")
        s2nCol = cutsRow.child(dbc.Col, width=1)
        s2nSel = s2nCol.child(daq.ToggleSwitch, size=30, value=False,
                                          label=["all", "s2n>10"])
        cutsRow.child(dbc.Col, width=2).child(html.Div, "")
        bmfile = cutsRow.child(dbc.Col, width=3).child(html.Div, "Beammap File: None")
        cuts = {
            'unflagged': unflaggedSel,
            's2n10': s2nSel,
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

        # A summary table
        tableBox = infoBox.child(dbc.Col, width=5)
        table = tableBox.child(dbc.Row).child(dcc.Graph)

        # Put in a break
        #bigBox.child(dbc.Row).child(html.Hr)

        # Some potential useful diagnostic plots
        fwhmxCol, fwhmyCol = bigBox.colgrid(1, 2)
        fwhmxPlot = fwhmxCol.child(dbc.Row).child(dcc.Graph)
        fwhmyPlot = fwhmyCol.child(dbc.Row).child(dcc.Graph)
        sensCol, _ = bigBox.colgrid(1, 2)
        sensPlot = sensCol.child(dbc.Row).child(dcc.Graph)
        staticPlots = {'fwhmx': fwhmxPlot,
                       'fwhmy': fwhmyPlot,
                       'sens': sensPlot,}

        
        super().setup_layout(app)

        self._registerCallbacks(
            app,
            aptList,
            colList,
            cuts,
            threeStat,
            staticPlots,
            table,
            bmfile
        )
        return

    
    def _registerCallbacks(
            self,
            app,
            aptList,
            colList,
            cuts,
            threeStat,
            staticPlots,
            table,
            bmfile,
    ):

        # ---------------------------
        # colList dropdown
        # ---------------------------
        @app.callback(
            [
                Output(colList.id, "options"),
                Output(bmfile.id, "children"),
            ],
            [
                Input(aptList.id, "value"),
            ],
            #prevent_initial_call=True,
        )
        def columnListDropdown(aptFile):
            if (aptFile == "") | (aptFile is None):
                raise PreventUpdate
            self.aptd = ToltecAptDiagnostics(aptFile=aptFile)
            options = []
            for k in self.aptd.getPlottableKeys():
                options.append({"label": k, "value": k})

            # Collect beammap files if available
            g = glob(os.path.dirname(aptFile)+'/toltec*.fits')
            if(len(g) > 0):
                bmFileAvailable = "Beammap Files Available"
                self.beammapFiles = g
            else:
                bmFileAvailable = "No Beammap Files Available"
                self.beammapFiles = None
            return [options, bmFileAvailable]

        
        # ---------------------------
        # aptList dropdown
        # ---------------------------
        @app.callback(
            [
                threeStat['a1100']['plot'].component_output,
                threeStat['a1400']['plot'].component_output,
                threeStat['a2000']['plot'].component_output,
                Output(threeStat['a1100']['hist'].id, "figure"),
                Output(threeStat['a1400']['hist'].id, "figure"),
                Output(threeStat['a2000']['hist'].id, "figure"),
            ],
            [
                Input(aptList.id, "value"),
                Input(colList.id, "value"),
                Input(cuts['unflagged'].id, "value"),
                Input(cuts['s2n10'].id, "value"),
                threeStat['a1100']['plot'].component_inputs,
                threeStat['a1400']['plot'].component_inputs,
                threeStat['a2000']['plot'].component_inputs,
            ],
            prevent_initial_call=True,
        )
        def primaryDropdown(aptFile, aptCol, unflagged, s2n10, *sp_inputs_list):
            if (aptFile == "") | (aptFile is None):
                raise PreventUpdate
            if (aptCol == "") | (aptCol is None):
                raise PreventUpdate
            while (self.aptd is None):
                time.sleep(0.1)
            
            # apply apt cuts here
            cutapt = self.aptd.apt.copy()
            if(unflagged):
                cutapt = self.aptd.cullByColumn('flag', apt=cutapt, valueMax=0, verbose=False)
            if(s2n10):
                cutapt = self.aptd.cullByColumn('sig2noise', apt=cutapt, valueMin=10.,
                                           verbose=False)

            # build the plots
            outputs = []
            for sp_inputs, array in zip(sp_inputs_list, ['a1100', 'a1400', 'a2000']):
                apt = self.aptd.getArrayApt(array, apt=cutapt)
                bigList = []
                for i in range(len(apt)):
                    bigList.append([apt['x_t'][i], apt['y_t'][i], apt[aptCol][i]])
                nDets = len(bigList)
                data = pd.DataFrame(bigList, columns=["x", "y", "z"])
                units = apt.meta[aptCol][0].replace('units: ', '')
                label = "{0:} ({1:} dets)<br>{2:} [{3:}]".format(array, nDets, aptCol, units)
                if(array == 'a1100'):
                    ylabel = "y_t [arcsec]"
                else:
                    ylabel = ' '
                outputs.append(
                    threeStat[array]['plot'].make_figure_data(
                        data,
                        title = label,
                        x_label = "x_t [arcsec]",
                        y_label = ylabel,
                        xaxis_range = [-150, 150],
                        yaxis_range = [-150, 150],
                        axis_font={'size': 8,},
                        image_height=200,
                        image_width=250,
                        marker_size=3,
                        **sp_inputs,
                        )
                    )
            
            # build the histograms
            histFigs = []
            for i in range(3):
                fig = histogramNetworksPlotly(cutapt, i, aptCol)
                fig.update_layout(
                    width=275,
                    height=275,
                    font={'size': 8,},
                    legend=dict(
                        bgcolor='rgba(0,0,0,0)',
                        font=dict(size= 8),
                        yanchor="top", y=0.99,
                        xanchor="right", x=0.90,),
                    margin=go.layout.Margin(
                        l=10,
                        r=10,
                        b=5,
                        t=40,
                    ),)
                histFigs.append(fig)
            
            return outputs + histFigs

        
        # ---------------------------
        # static plots
        # ---------------------------
        @app.callback(
            [
                Output(staticPlots['fwhmx'].id, "figure"),
                Output(staticPlots['fwhmy'].id, "figure"),
                Output(staticPlots['sens'].id, "figure"),
            ],
            [
                Input(aptList.id, "value"),
                Input(cuts['unflagged'].id, "value"),
                Input(cuts['s2n10'].id, "value"),
            ],
            prevent_initial_call=True,
        )
        def makeStaticPlots(aptFile, unflagged, s2n10):
            if (aptFile == "") | (aptFile is None):
                raise PreventUpdate
            while (self.aptd is None):
                time.sleep(0.1)

            # apply apt cuts here
            cutapt = self.aptd.apt.copy()
            if(unflagged):
                cutapt = self.aptd.cullByColumn('flag', apt=cutapt, valueMax=0, verbose=False)
            if(s2n10):
                cutapt = self.aptd.cullByColumn('sig2noise', apt=cutapt, valueMin=10.,
                                           verbose=False)
            fwhmFigs = self.aptd.plotFWHMvsPos(apt=cutapt, returnPlotlyFig=True)
            sensFig = self.aptd.plotSensvsFreq(apt=cutapt, returnPlotlyFig=True)
            xaxis, yaxis = getXYAxisLayouts()
            figList = fwhmFigs+[sensFig]
            for fig in figList:
                fig.update_layout(height=200, xaxis=xaxis, yaxis=yaxis)
            return figList


        # ---------------------------
        # Summary Table
        # ---------------------------
        @app.callback(
            [
                Output(table.id, "figure")
            ],
            [
                Input(aptList.id, "value"),
                Input(cuts['unflagged'].id, "value"),
                Input(cuts['s2n10'].id, "value"),
            ],
            prevent_initial_call=True,
        )
        def makeTable(aptFile, unflagged, s2n10):
            if (aptFile == "") | (aptFile is None):
                raise PreventUpdate
            while (self.aptd is None):
                time.sleep(0.1)

            # apply apt cuts here
            apt = self.aptd.apt
            cutapt = apt.copy()
            if(unflagged):
                cutapt = self.aptd.cullByColumn('flag', apt=cutapt, valueMax=0, verbose=False)
            if(s2n10):
                cutapt = self.aptd.cullByColumn('sig2noise', apt=cutapt, valueMin=10.,
                                           verbose=False)

            # table values
            cells = [["<b>Total</b>", "<b>Not cut</b>", "<b>Med sens [mJyrts]</b>"]]
            for i, array in enumerate(['a1100', 'a1400', 'a2000']):
                a = cutapt[cutapt['array'] == i]
                cells.append([len(apt[apt['array'] == i]), len(a), int(np.median(a['sens']))])
            # build the table
            fig = go.Figure()
            fig.add_trace(
                go.Table(
                    header=dict(values=['', '<b>a1100</b>', '<b>a1400</b>', '<b>a2000</b>'],
                                line_color='darkslategray',
                                fill_color='orange',
                                align='center'),
                    cells=dict(values=cells,
                               line_color='darkslategray',
                               fill_color='white',
                               align='center',
                               ),
                    columnwidth=[0.4, 0.2, 0.2, 0.2],
                ))
            fig.update_layout(
                autosize=False,
                margin=go.layout.Margin(
                    l=20,
                    r=20,
                    b=40,
                    t=0,
                    pad=0,
                ))
            
            
            return [fig]

        


# common figure axis definitions
def getXYAxisLayouts():
    xaxis = dict(
        titlefont=dict(size=10),
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
        titlefont=dict(size=10),
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
