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


class ToltecFocusViewer(ComponentTemplate):
    class Meta:
        component_cls = dbc.Container

    def __init__(
        self,
        title_text="Toltec Focus Viewer",
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
        dPath = "/Users/wilson/Desktop/tmp/focus_obs/"
        g = glob("{}f*".format(dPath))
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
        modelSwitch = controlsRow.child(dbc.Col, width=1).child(
            daq.ToggleSwitch, size=30, value=True, label=["Parabola", "Gaussian"])
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
            "model": modelSwitch,
            "what2plot": what2Plot,}
        
        # Put in a break
        #bigBox.child(dbc.Row).child(html.Br)
        bigBox.child(dbc.Row).child(html.Hr)

        # The fit to all three bands of focus data
        fitBoxRow = bigBox.child(dbc.Row)
        fitBox = fitBoxRow.child(dbc.Col, width=12)
        fitPlot = fitBox.child(dbc.Row).child(dcc.Loading, type="circle").child(dcc.Graph)

        # Three columns of figures with the focus imaging
        imageDiv = bigBox.child(dbc.Row)
        images = {
            'div': imageDiv,
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
                Input(controls['model'].id, "value"),
                Input(controls['what2plot'].id, "value"),
            ],
            prevent_initial_call=True,
        )
        def makePlots(data, gaussianFit, what2plot):
            if (data == "") | (data is None):
                return makeEmptyFigs(1)
            fitPlot = getFitPlot(data, gaussianFit, what2plot)
            return [fitPlot]


        # ---------------------------
        # make the plots
        # ---------------------------
        @app.callback(
            [
                Output(images['div'].id, "children"),
            ],
            [
                Input(dataStore.id, "data"),
            ],
            prevent_initial_call=True,
        )
        def makePlots(data):
            if (data == "") | (data is None):
                return [None]
            div = makeImageDiv(data)
            return [html.Div(div)]



def fetchFocusData(path, mapType, fitter, cutout):
    # determine the obsnums that make up the focus set
    g = glob(path+'/reduced/redu01/10*/raw/')
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
    a1100 = pd.DataFrame(a1100).sort_values('m2z')
    a1400 = pd.DataFrame(a1400).sort_values('m2z')
    a2000 = pd.DataFrame(a2000).sort_values('m2z')
    data = {
        'a1100': a1100.to_json(),
        'a1400': a1400.to_json(),
        'a2000': a2000.to_json(),
        }
    return data


def getFitPlot(data, gaussianFit, what2plot):
    a1100 = pd.read_json(data['a1100'])
    a1400 = pd.read_json(data['a1400'])
    a2000 = pd.read_json(data['a2000'])

    # do the fit
    xb, xf, model = fitData(a1100, a1400, a2000, gaussian=gaussianFit)
    
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
                go.Scatter(x=a['m2z'], y=a['amp'],
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
                go.Scatter(x=a['m2z'], y=a['x_fwhm'],
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
                go.Scatter(x=a['m2z'], y=a['y_fwhm'],
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
            fitFig.update_xaxes(title="M2Z")
            fitFig.update_yaxes(range=[5, 22], title="Fitted FWHM [arcsec]")
        else:
            fitFig.add_trace(
                go.Scatter(x=a['m2z'], y=a['x_t'],
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
                go.Scatter(x=a['m2z'], y=a['y_t'],
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
        fitFig.update_xaxes(title="M2Z [mm]")
        fitFig.add_vline(x=xb[i], line_dash="dash", line_color=colors[i])

    if(what2plot == 'amp'):
        fitFig.add_annotation(
            x=a1100['m2z'].values.min()+0.5,
            y=maxAmp,
            text="Astig Coef: {}".format(a['astig'].values[0]),
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


def makeImageDiv(data):
    # The figures are generated below.
    a11Fig, a14Fig, a20Fig = getPlots(data)

    # Adjust the margin-bottom to reduce the vertical space between images
    image_style = {'margin-bottom': '10px'}  # Smaller bottom margin

    # Create a column for each array with reduced vertical space
    col1 = [dcc.Graph(figure=a11Fig, style=image_style)]
    col2 = [dcc.Graph(figure=a14Fig, style=image_style)]
    col3 = [dcc.Graph(figure=a20Fig, style=image_style)]

    # Combine columns into a single row with center alignment
    row = dbc.Row(
        [
            dbc.Col(col1, md=4),
            dbc.Col(col2, md=4),
            dbc.Col(col3, md=4)
        ],
        justify="center"  # Center the row
    )

    # Wrap the row in a Div and set the overflow to auto
    div = html.Div(row, style={'overflowY': 'auto', 'height': 'calc(100vh - 150px)'})

    # To center the Div horizontally on the page, you might wrap it in another Div with display flex
    centered_div = html.Div(
        div,
        style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'height': '100%'
        }
    )

    return centered_div
'''

def makeImageDiv(data):
    # The figures are generated below.
    a11Fig, a14Fig, a20Fig = getPlots(data)

    # Create the column titles
    col_titles = dbc.Row(
        [
            dbc.Col(html.H3('a1100', className='text-center font-weight-bold'), width=3),
            dbc.Col(html.H3('a1400', className='text-center font-weight-bold'), width=3),
            dbc.Col(html.H3('a2000', className='text-center font-weight-bold'), width=3),
        ],
        justify="center", 
    )

    # Adjust the style to reduce the vertical space between images
    image_style = {'margin-bottom': '5px'}

    # Create a column for each array with reduced vertical space
    col1 = [dcc.Graph(figure=a11Fig, style=image_style)]
    col2 = [dcc.Graph(figure=a14Fig, style=image_style)]
    col3 = [dcc.Graph(figure=a20Fig, style=image_style)]

    # Combine columns into a single row with center alignment for the graphs
    row = dbc.Row(
        [
            dbc.Col(col1, width=3, style={'padding': 0}),  # Remove padding for tighter fit
            dbc.Col(col2, width=3, style={'padding': 0}),
            dbc.Col(col3, width=3, style={'padding': 0}),
        ],
        justify="center",  
    )

    # Combine the title row and the graph row into a single column layout
    column_layout = html.Div([col_titles, row])

    # Wrap the combined layout in a Div and set the overflow to auto
    div = html.Div(column_layout, style={'overflowY': 'auto', 'height': 'calc(100vh - 150px)'})

    return div
'''

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
    titles = ["M2Z = {}mm".format(i) for i in df['m2z']]
    
    # Assume a square aspect ratio for each subplot
    aspect_ratio = 1
    
    # Create the figure with a single column and multiple rows
    fig = make_subplots(rows=nObs, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    for i, row in df.iterrows():
        fig.add_trace(
            go.Heatmap(
                x=row['x'],
                y=row['y'],
                z=row['image'],
                showscale=False
            ),
            row=i+1, col=1
        )
        # Update axes to maintain aspect ratio
        x_range = max(row['x']) - min(row['x'])
        y_range = max(row['y']) - min(row['y'])
        fig.update_xaxes(range=[min(row['x']), min(row['x']) + x_range], row=i+1, col=1)
        fig.update_yaxes(range=[min(row['y']), min(row['y']) + y_range], row=i+1, col=1)
        fig.update_yaxes(title_text=titles[i], row=i+1, col=1)

    # Update the layout for each figure
    fig.update_layout(
        autosize=True,
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=10, r=10, b=10, t=30 + nObs * 10)  # Adjust top margin for titles
    )
    
    # Adjust the height of the figure dynamically based on the number of subplots
    fig.update_layout(height=300 * nObs)

    return fig



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
        m2z = a['m2z'].values
        amp_err = a['amp_err'].values
        if(gaussian):
            g_init = models.Gaussian1D(amplitude=amp.max(), mean=m2z[np.argmax(amp)], stddev=1.,
                                       bounds={"mean": (-5, 5)})
            fit_g = fitting.LevMarLSQFitter()
            g = fit_g(g_init, m2z, amp)
            xf.append(np.linspace(m2z.min(), m2z.max(), 100))
            model.append(g(xf[-1]))
            xb.append(g.mean.value)
        else:
            w = np.where(amp >= 0.5*amp.max())[0]
            fits.append(np.polyfit(m2z[w], amp[w], 2, w=(1./amp_err[w])**2))
            xf.append(np.linspace(m2z[w].min(), m2z[w].max(), 100))
            model.append(fits[-1][0]*xf[-1]**2 + fits[-1][1]*xf[-1] + fits[-1][2])
            xb.append(-fits[-1][1]/(2.*fits[-1][0]))
    return xb, xf, model


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
        "template": ToltecFocusViewer,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", True),
    }
}
