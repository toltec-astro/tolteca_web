"""
To Do:
 - Add lots of comments.
 - There is some cleanup that must occur (e.g., buttons are too big)
 - Add obsnums and more defining data for the observation set (date, elevation, etc.)
"""

from ..toltec_dp_utils.ToltecSignalFits import ToltecSignalFits
from dash_component_template import ComponentTemplate
from ..common.plots.surface_plot import SurfacePlot
from dash.dependencies import Input, Output, State
from astropy.modeling.models import Gaussian1D
from astropy.modeling import models, fitting
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from tollan.utils.log import logger
from astropy.nddata import Cutout2D
from ..common import LabeledInput
from astropy.table import Table
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


class ToltecAstigViewer(ComponentTemplate):
    class Meta:
        component_cls = dbc.Container

    def __init__(
        self,
        title_text="Toltec Astig Viewer",
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
        dPath = "/Users/wilson/Desktop/tmp/astig_obs/"
        g = glob("{}a*".format(dPath))
        g.sort()
        focusPaths = g
        focusOptions = [{"label": p, "value": p} for p in focusPaths]

        # pull down to select obs stats file
        pulldownPanel, controls, bigBox = body.grid(3, 1)
        focusSelectBox = pulldownPanel.child(dbc.Row).child(dbc.Col, width=12)
        focusSelectRow = focusSelectBox.child(dbc.Row)
        focusSelectCol = focusSelectRow.child(dbc.Col, width=6)
        focusTitle = focusSelectCol.child(dbc.Row).child(
            html.H5, "FOCUS Choice", className="mb-2"
        )
        focusList = focusSelectCol.child(dbc.Row).child(
            dcc.Dropdown,
            options=focusOptions,
            placeholder="Select Focus Directory",
            value="",
            searchable=False,
            style=dict(width="100%", verticalAlign="middle"),
        )
        dataStore = focusSelectRow.child(dcc.Store)
        
        # The various switches and controls
        controls.child(dbc.Row).child(html.Br)
        controlsRow = controls.child(dbc.Row)
        controlsRow.child(dbc.Col, width=1)
        cutoutSwitch = controlsRow.child(dbc.Col, width=1).child(
            daq.ToggleSwitch, size=30, value=True, label=["full", "cutout"])
        controlsRow.child(dbc.Col, width=1)
        rawSwitch = controlsRow.child(dbc.Col, width=1).child(
            daq.ToggleSwitch, size=30, value=False, label=["raw", "filtered"])
        controlsRow.child(dbc.Col, width=1)
        fitterSwitch = controlsRow.child(dbc.Col, width=1).child(
            daq.ToggleSwitch, size=30, value=False, label=["Ceres", "Astropy"])
        controlsRow.child(dbc.Col, width=1)
        what2Plot = controlsRow.child(dbc.Col, width=1).child(
            dbc.RadioItems,
            class_name="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Amp", "value": 'amp'},
                {"label": "FWHM", "value": 'fwhm'},
                {"label": "Offset", "value": 'offset'},
                ],
            value='amp', size='sm')
        controls = {
            "cutout": cutoutSwitch,
            "mapType": rawSwitch,
            "fitter": fitterSwitch,
            "what2plot": what2Plot,}
        
        # Put in a break
        #bigBox.child(dbc.Row).child(html.Br)
        bigBox.child(dbc.Row).child(html.Hr)

        # The fit to all three bands of focus data
        fitBoxRow = bigBox.child(dbc.Row)
        fitBox = fitBoxRow.child(dbc.Col, width=12)
        fitPlot = fitBox.child(dbc.Row).child(dcc.Loading, type="circle").child(dcc.Graph)

        # Three columns of figures with the focus imaging
        imagingRow = bigBox.child(dbc.Row).child(dbc.Col, width=12).child(dbc.Row)
        a11Col = imagingRow.child(dbc.Col, width=4)
        a11Plot = a11Col.child(dbc.Row).child(dcc.Loading, type="circle").child(dcc.Graph)
        a14Col = imagingRow.child(dbc.Col, width=4)
        a14Plot = a14Col.child(dbc.Row).child(dcc.Loading, type="circle").child(dcc.Graph)
        a20Col = imagingRow.child(dbc.Col, width=4)
        a20Plot = a20Col.child(dbc.Row).child(dcc.Loading, type="circle").child(dcc.Graph)
        images = {
            "a11Plot": a11Plot,
            "a14Plot": a14Plot,
            "a20Plot": a20Plot,
            }
        
        super().setup_layout(app)

        self._registerCallbacks(
            app,
            focusList,
            fitPlot,
            images,
            controls,
            dataStore,
        )
        return

    
    def _registerCallbacks(
            self,
            app,
            focusList,
            fitPlot,
            images,
            controls,
            dataStore
    ):

        # ---------------------------
        # focus select dropdown
        # ---------------------------
        @app.callback(
            [
                Output(dataStore.id, "data"),
            ],
            [
                Input(focusList.id, "value"),
                Input(controls['cutout'].id, "value"),
                Input(controls['mapType'].id, "value"),
                Input(controls['fitter'].id, "value"),
                Input(dataStore.id, "data"),
            ],
            prevent_initial_call=False,
        )
        def focusListDropdown(path, cutout, mapType, fitter, dataCount):
            if (path == "") | (path is None):
                return [None]
            data = fetchFocusData(path, mapType, fitter, cutout)
            return [data]


        # ---------------------------
        # make the fit plot
        # ---------------------------
        @app.callback(
            [
                Output(fitPlot.id, "figure"),
            ],
            [
                Input(dataStore.id, "data"),
                Input(controls['what2plot'].id, "value"),
            ],
            prevent_initial_call=True,
        )
        def makePlots(data, what2plot):
            if (data == "") | (data is None):
                return makeEmptyFigs(1)
            fitPlot = getFitPlot(data, what2plot)
            return [fitPlot]


        # ---------------------------
        # make the plots
        # ---------------------------
        @app.callback(
            [
                Output(images['a11Plot'].id, "figure"),
                Output(images['a14Plot'].id, "figure"),
                Output(images['a20Plot'].id, "figure"),
            ],
            [
                Input(dataStore.id, "data"),
            ],
            prevent_initial_call=True,
        )
        def makePlots(data):
            if (data == "") | (data is None):
                return makeEmptyFigs(3)
            a11Plots, a14Plots, a20Plots = getPlots(data)
            return [a11Plots, a14Plots, a20Plots]

        

def fetchFocusData(path, mapType, fitter, cutout):
    # determine the obsnums that make up the focus set
    g = glob(path+'/10*/raw/')
    g.sort()
    nObs = len(g)
    obsnums = [i.split('/')[-2] for i in g]

    # cutout
    if(cutout):
        size = (90, 90)
    else:
        size = (320, 320)
    
    a1100 = []
    a1400 = []
    a2000 = []
    # make a run through the files
    for d, array in zip([a1100, a1400, a2000], ['a1100', 'a1400', 'a2000']):
        for path in g:
            p = path
            if(mapType):
                p = path.replace('raw', 'filtered')
                # fitter=True
            print("Working on: {}".format(p))
            tsf = ToltecSignalFits(path=p, array=array)
            tsf.setWeightCut(0.2)
            if(fitter):
                fit, image, wcs, Xs, Ys = tsf.fitGaussian(
                    'signal_I', 
                    size=size,
                    plotCutout=False,
                    plotFull=False,
                    plotConvolved=False,
                    returnImage=True,
                    verbose=False,)
                x = Xs[:, 0]
                y = Ys[0, :]
                d.append(dict(
                    obsnum = tsf.obsnum,
                    m2z = tsf.headers[0]['HEADER.M2.ZREQ'],
                    astig = tsf.headers[0]['HEADER.M1.ZERNIKEC'],
                    amp = fit[0].amplitude.value,
                    amp_err = np.sqrt(fit[1][0, 0]),
                    x_t = fit[0].x_mean.value,
                    x_t_err = np.sqrt(fit[1][1, 1]),
                    y_t = fit[0].y_mean.value,
                    y_t_err = np.sqrt(fit[1][2, 2]),
                    x_fwhm = fit[0].x_stddev.value*2.355,
                    x_fwhm_err = np.sqrt(fit[1][3, 3])*2.355,
                    y_fwhm = fit[0].y_stddev.value*2.355,
                    y_fwhm_err = np.sqrt(fit[1][4, 4])*2.355,
                    image = image,
                    x = x.value,
                    y = y.value))                    
            else:
                cutout = tsf.plotCutout('signal_I', noPlot=True, size=size)
                image = cutout.data
                wcs = cutout.wcs
                s = image.shape
                x = wcs.pixel_to_world(np.arange(0, s[0]), 0)[0]
                y = wcs.pixel_to_world(0, np.arange(0, s[1]))[1]
                d.append(dict(
                    obsnum = tsf.obsnum,
                    m2z = tsf.headers[0]['HEADER.M2.ZREQ'],
                    astig = tsf.headers[0]['HEADER.M1.ZERNIKEC'],
                    amp = tsf.headers[1]['POINTING.AMP'],
                    amp_err = tsf.headers[1]['POINTING.AMP_ERR'],
                    x_t = tsf.headers[1]['POINTING.X_T'],
                    x_t_err = tsf.headers[1]['POINTING.X_T_ERR'],
                    y_t = tsf.headers[1]['POINTING.Y_T'],
                    y_t_err = tsf.headers[1]['POINTING.Y_T_ERR'],
                    x_fwhm = tsf.headers[1]['POINTING.A_FWHM'],
                    x_fwhm_err = tsf.headers[1]['POINTING.A_FWHM_ERR'],
                    y_fwhm = tsf.headers[1]['POINTING.B_FWHM'],
                    y_fwhm_err = tsf.headers[1]['POINTING.B_FWHM_ERR'],
                    image = image,
                    x = x.value,
                    y = y.value))
    # convert the lists of dicts to pandas dataframes
    a1100 = pd.DataFrame(a1100).sort_values('astig')
    a1400 = pd.DataFrame(a1400).sort_values('astig')
    a2000 = pd.DataFrame(a2000).sort_values('astig')
    data = {
        'a1100': a1100.to_json(),
        'a1400': a1400.to_json(),
        'a2000': a2000.to_json(),
        }
    return data


def getFitPlot(data, what2plot):
    a1100 = pd.read_json(data['a1100'])
    a1400 = pd.read_json(data['a1400'])
    a2000 = pd.read_json(data['a2000'])

    # do the fit
    xb, xf, model = fitData(a1100, a1400, a2000, gaussian=True)
        
    # Make the fit plot
    xaxis, yaxis = getXYAxisLayouts()
    fitFig = go.Figure()
    maxAmp = 0.
    colors = ['blue', 'green', 'red']
    for i, a, array in zip(np.arange(3), [a1100, a1400, a2000], ['a1100', 'a1400', 'a2000']):
        if(what2plot == 'amp'):
            if a['amp'].values.max() > maxAmp:
                maxAmp = a['amp'].max() * 1.1
            fitFig.add_trace(
                go.Scatter(x=a['astig'], y=a['amp'],
                           error_y = dict(
                               type='data',
                               array=a['amp_err'],
                               visible=True,
                           ),
                           name=array,
                           mode='lines+markers',
                           marker=dict(color=colors[i], size=2),)
            )
            fitFig.add_trace(
                go.Scatter(x=xf[i], y=model[i], mode='markers',
                           marker=dict(color=colors[i], size=5),
                           name=array+' fit'))
            fitFig.update_yaxes(range=[0, maxAmp], title="Fitted Amplitude [mJy/Beam]")
        elif(what2plot == 'fwhm'):
            fitFig.add_trace(
                go.Scatter(x=a['astig'], y=a['x_fwhm'],
                           error_y = dict(
                               type='data',
                               array=a['x_fwhm_err'],
                               visible=True,
                           ),
                           mode='lines+markers',
                           marker=dict(color=colors[i], size=2),
                           name=array+' x fwhm',),
            )
            fitFig.add_trace(
                go.Scatter(x=a['astig'], y=a['y_fwhm'],
                           error_y = dict(
                               type='data',
                               array=a['y_fwhm_err'],
                               visible=True,
                           ),
                           mode='lines+markers',
                           marker=dict(color=colors[i], size=2),
                           line=dict(color=colors[i], dash='dot'),
                           name=array+' y fwhm',),
            )
            fitFig.update_xaxes(title="ASTIG")
            fitFig.update_yaxes(range=[5, 22], title="Fitted FWHM [arcsec]")
        else:
            fitFig.add_trace(
                go.Scatter(x=a['astig'], y=a['x_t'],
                           error_y = dict(
                               type='data',
                               array=a['x_t_err'],
                               visible=True,
                           ),
                           mode='lines+markers',
                           marker=dict(color=colors[i], size=2),
                           line=dict(color=colors[i]),
                           name=array+' x-offset',),
            )
            fitFig.add_trace(
                go.Scatter(x=a['astig'], y=a['y_t'],
                           error_y = dict(
                               type='data',
                               array=a['y_t_err'],
                               visible=True,
                           ),
                           mode='lines+markers',
                           marker=dict(color=colors[i], size=2),
                           line=dict(color=colors[i], dash='dot'),
                           name=array+' y-offset',),
            )
            fitFig.update_yaxes(title="Centroid Offset [arcsec]")
        fitFig.update_xaxes(title="ASTIG [mm]")
        fitFig.add_vline(x=xb[i], line_dash="dash", line_color=colors[i])

    if(what2plot == 'amp'):
        fitFig.add_annotation(
            x=a1100['astig'].values.min()+0.5,
            y=maxAmp,
            text="M2Z [mm]: {}".format(a['m2z'].values[0]),
            showarrow=False,
            font={'size': 12},
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=0.8)

    fitFig.update_layout(
        xaxis=xaxis, yaxis=yaxis,
        uirevision=True,
        showlegend=True,
        autosize=True,
        plot_bgcolor="white",
    )
    return fitFig


def getPlots(data):
    a1100 = pd.read_json(data['a1100'])
    a1400 = pd.read_json(data['a1400'])
    a2000 = pd.read_json(data['a2000'])
    a11Fig = makeImages(a1100)
    a14Fig = makeImages(a1400)
    a20Fig = makeImages(a2000)
    return a11Fig, a14Fig, a20Fig


def makeImages(df):
    df = df.reset_index()
    nObs = len(df)
    titles = ["ASTIG = {}mm".format(i) for i in df['astig']]
    fig = make_subplots(rows=nObs, cols=1, shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing = 0.01, subplot_titles=titles)
    for i, row in df.iterrows():
        ulabel = "[arcsec]"
        fig.add_trace(
            go.Heatmap(
                x=row['x'],
                y=row['y'],
                z=row['image'],
                showscale=False,
            ),
            row=i+1, col=1)
        ry = np.array(row['y'])
        rx = np.array(row['x'])
        fig.update_yaxes(range=[ry.min(), ry.max()], scaleanchor="x", row=i+1, col=1)
        fig.update_xaxes(range=[rx.min(), rx.max()], row=i+1, col=1)
        
    fig.update_layout(
        uirevision=True, showlegend=False, autosize=False, plot_bgcolor="white"
    )
    
    fig.update_layout(
        height=410*nObs,
        width=400,
        margin = go.layout.Margin(
            l=10,
            r=10,
            b=10,
            t=30,
        )
    )
    return fig



def makeEmptyFigs(nfigs):
    figs = []
    xaxis, yaxis = getXYAxisLayouts()
    for i in range(nfigs):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],))
        fig.update_layout(
            xaxis=xaxis,
            yaxis=yaxis,)
        figs.append(fig)
    return figs


# fit either a parabola or a gaussian
def fitData(a1100, a1400, a2000, gaussian=False):
    fits = []
    model = []
    xf = []
    xb = []
    data = [a1100, a1400, a2000]
    for i in range(3):
        a = data[i]
        amp = a['amp'].values
        astig = a['astig'].values
        amp_err = a['amp_err'].values
        if(gaussian):
            g_init = models.Gaussian1D(amplitude=amp.max(), mean=astig[np.argmax(amp)], stddev=100.)
            fit_g = fitting.LevMarLSQFitter()
            g = fit_g(g_init, astig, amp)
            xf.append(np.linspace(astig.min(), astig.max(), 100))
            model.append(g(xf[-1]))
            xb.append(g.mean.value)
        else:
            w = np.where(amp >= 0.5*amp.max())[0]
            fits.append(np.polyfit(astig[w], amp[w], 2, w=(1./amp_err[w])**2))
            xf.append(np.linspace(astig[w].min(), astig[w].max(), 100))
            model.append(fits[-1][0]*xf[-1]**2 + fits[-1][1]*xf[-1] + fits[-1][2])
            xb.append(-fits[-1][1]/(2.*fits[-1][0]))
    return xb, xf, model



# common figure axis definitions
def getXYAxisLayouts():
    xaxis = dict(
        titlefont=dict(size=12),
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor="black",
        linewidth=2,
        ticks="outside",
        tickfont=dict(
            family="Arial",
            size=10,
            color="rgb(82, 82, 82)",
        ),
    )

    yaxis = dict(
        titlefont=dict(size=12),
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor="black",
        linewidth=2,
        ticks="outside",
        tickfont=dict(
            family="Arial",
            size=10,
            color="rgb(82, 82, 82)",
        ),
    )
    return xaxis, yaxis


DASHA_SITE = {
    "dasha": {
        "template": ToltecAstigViewer,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", True),
    }
}
