import sys
sys.path.append("/Users/wilson/GitHub/toltec-data-product-utilities/toltec_dp_utils/")
from ToltecSignalFits import ToltecSignalFits

from dash_component_template import ComponentTemplate, NullComponent
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from tollan.utils.log import logger
from astropy.nddata import Cutout2D
import dash_core_components as dcc
from ..common import LabeledInput
from astropy import units as u
import plotly.graph_objs as go
import plotly.express as px
import dash_daq as daq
from dash import html
from glob import glob
import numpy as np
import functools
import os


class ToltecSignalFitsViewer(ComponentTemplate):
    class Meta:
        component_cls = dbc.Container

    def __init__(
            self,
            title_text="Toltec Signal Fits Viewer",
            subtitle_text="(test version)",
            **kwargs,
    ):
        kwargs.setdefault("fluid", True)
        super().__init__(**kwargs)
        self._title_text = title_text
        self._subtitle_text = subtitle_text
        self.fluid = True
        # this is technically cheating
        self.tsf = None

        
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
        dPath = '/Users/wilson/Desktop/tmp/macs0717/pointing/102384/'
        g = glob('{}redu*/'.format(dPath))
        gp = glob('/Users/wilson/Desktop/tmp/cosmos/redu*/')
        g = g+gp
        pathOptions = [{'label': gg, 'value': gg} for gg in g]

        # pull down to select signal fits path
        controls_panel, views_panel, bigBox = body.grid(3, 1)
        controlBox = controls_panel.child(dbc.Row).child(dbc.Col, width=5)
        settingsRow = controlBox.child(dbc.Row)
        fitsPath_select = settingsRow.child(dbc.Col).child(
            dcc.Dropdown,
            options=pathOptions,
            placeholder="Select Path",
            value='',
            searchable=False,
            style=dict(
                width='100%',
                verticalAlign="middle"
            ))

        # Put in a break
        bigBox.child(dbc.Row).child(html.Br)

        # Array, weight cut, edge trimming controls
        mainControlsTitlesRow = bigBox.child(dbc.Row)
        arrayTitle = mainControlsTitlesRow.child(dbc.Col, width=3).child(html.H5,
                                                                         "Array Choice",
                                                                         className='mb-2')
        wtcutTitle = mainControlsTitlesRow.child(dbc.Col, width=3).child(html.H5,
                                                                         "Weight Cut",
                                                                         className='mb-2')
        tzTitle = mainControlsTitlesRow.child(dbc.Col, width=2).child(html.H5,
                                                                      "Trim Map Zeros",
                                                                      className='mb-2')
        mainControlsRow = bigBox.child(dbc.Row)
        arrayCol = mainControlsRow.child(dbc.Col, width=3)
        arrayChoice = arrayCol.child(
            dcc.RadioItems,
            options=[{'label': 'a1100', 'value': 'a1100', 'disabled': True},
                     {'label': 'a1400', 'value': 'a1400', 'disabled': True},
                     {'label': 'a2000', 'value': 'a2000', 'disabled': True},],
            inline=True,
            inputStyle={"margin-right": "5px", "margin-left": "20px"})
        weightCol = mainControlsRow.child(dbc.Col, width=3)
        weightCut = getWeightCutControl(weightCol)
        trimEdgeCol = mainControlsRow.child(dbc.Col, width=1)
        trimEdgeSel = trimEdgeCol.child(
            daq.ToggleSwitch,
            default=True)
        mainControls = {'array': arrayChoice,
                        'weightCut': weightCut,
                        'trimEdgeSelector': trimEdgeSel}
        
        # Put in a break
        bigBox.child(dbc.Row).child(html.Br)

        # The maps
        imageRow = bigBox.child(dbc.Row)
        signalCol = imageRow.child(dbc.Col, width=4)
        signalff = getFancyFigContainer(signalCol)
        weightCol = imageRow.child(dbc.Col, width=4)
        weightff = getFancyFigContainer(weightCol)
        s2nCol = imageRow.child(dbc.Col, width=4)
        s2nff = getFancyFigContainer(s2nCol)
        fancyFigs = {'signal': signalff,
                     'weight': weightff,
                     's2n': s2nff,}

        
        # Another break
        bigBox.child(dbc.Row).child(html.Br)
        
        # Map Histogram
        imageOptions = [{'label': a, 'value': a} for a in ['signal', 'weight', 'sig2noise']]
        histBox = bigBox.child(dbc.Row)
        histPlotPanel = histBox.child(dbc.Row)
        histControlPanel = histBox.child(dbc.Row)
        histControl = histControlPanel.child(
            dcc.RadioItems,
            options=imageOptions,
            inline=True,
            value='signal',
        )
        histPlot = histPlotPanel.child(dcc.Loading, type='circle').child(dcc.Graph)
        pixelHistogram = {'controls': histControl,
                          'plot': histPlot}



        self._registerCallbacks(app, fitsPath_select, pixelHistogram, fancyFigs,
                                mainControls)
        return


    def _registerCallbacks(self, app, fitsPath_select, pixelHistogram, fancyFigs,
                           mainControls):
        print("Registering Callbacks")

        # ---------------------------
        # FITS Path dropdown
        # ---------------------------
        @app.callback(
            [
                Output(pixelHistogram['plot'].id, "figure"),
                Output(fancyFigs['signal']['histRange'].id, 'min'),
                Output(fancyFigs['signal']['histRange'].id, 'max'),
                Output(fancyFigs['weight']['histRange'].id, 'min'),
                Output(fancyFigs['weight']['histRange'].id, 'max'),
                Output(fancyFigs['s2n']['histRange'].id, 'min'),
                Output(fancyFigs['s2n']['histRange'].id, 'max'),
                Output(mainControls['array'].id, 'options'),
                Output(mainControls['array'].id, 'value'),
            ],
            [
                Input(fitsPath_select.id, 'value'),
                Input(pixelHistogram['controls'].id, 'value'),
                Input(mainControls['weightCut'].id, 'value'),
                Input(mainControls['array'].id, 'options'),
                Input(mainControls['array'].id, 'value'),
            ],
            prevent_initial_call=True
        )
        def primaryDropdown(path, histImage, weightCut, arrayOpt, array):
            if((path == '') | (path is None)):
                raise PreventUpdate

            # check which array files exist at this path
            arrayOptions = getArrayOptions(path, arrayOpt)
            availableArrays = []
            for a in arrayOptions:
                if a['disabled'] == False:
                    availableArrays.append(a['value'])
            if(len(availableArrays) == 0):
                raise ValueError(
                    "No FITS files available for use at this path.")
            if(array is None):
                array = availableArrays[0]

            tsf = ToltecSignalFits(path=path, array=array)
            tsf.setWeightCut(weightCut)
            histImage = histImage + '_I'
            histFig = tsf.plotMapHistogram(histImage, returnPlotlyPlot=True)
            signal = tsf.getMap('signal_I')
            weight = tsf.getMap('weight_I')
            s2n = tsf.getMap('sig2noise_I')
            return [histFig,
                    np.ceil(signal.min()), np.ceil(signal.max()),
                    0, np.ceil(weight.max()), -5, np.ceil(s2n.max()),
                    arrayOptions, array]


        # ---------------------------
        # Signal Fancy Fig
        # ---------------------------
        @app.callback(
            [
                Output(fancyFigs['signal']['histPlot'].id, "figure"),
                Output(fancyFigs['signal']['imagePlot'].id, "figure"),
            ],
            [
                Input(fitsPath_select.id, 'value'),
                Input(fancyFigs['signal']['histRange'].id, 'value'),
                Input(mainControls['weightCut'].id, 'value'),
                Input(mainControls['array'].id, 'value'),
                Input(mainControls['trimEdgeSelector'].id, 'value'),
            ],
            prevent_initial_call=True
        )
        def histRangeChange(path, hrange, weightCut, array, trimEdge):
            if((path == '') | (path is None) | (array is None)):
                raise PreventUpdate
            hfig, imfig = getFancyFig('signal_I', path, hrange, weightCut, array, trimEdge)
            return [hfig, imfig]


        # ---------------------------
        # Weight Fancy Fig
        # ---------------------------
        @app.callback(
            [
                Output(fancyFigs['weight']['histPlot'].id, "figure"),
                Output(fancyFigs['weight']['imagePlot'].id, "figure"),
            ],
            [
                Input(fitsPath_select.id, 'value'),
                Input(fancyFigs['weight']['histRange'].id, 'value'),
                Input(mainControls['weightCut'].id, 'value'),
                Input(mainControls['array'].id, 'value'),
                Input(mainControls['trimEdgeSelector'].id, 'value'),
            ],
            prevent_initial_call=True
        )
        def histRangeChange(path, hrange, weightCut, array, trimEdge):
            if((path == '') | (path is None) | (array is None)):
                raise PreventUpdate
            if(hrange[0] == -99):
                hrange[0] = 0
            hfig, imfig = getFancyFig('weight_I', path, hrange, weightCut, array, trimEdge)
            return [hfig, imfig]


        # ---------------------------
        # S/N Fancy Fig
        # ---------------------------
        @app.callback(
            [
                Output(fancyFigs['s2n']['histPlot'].id, "figure"),
                Output(fancyFigs['s2n']['imagePlot'].id, "figure"),
            ],
            [
                Input(fitsPath_select.id, 'value'),
                Input(fancyFigs['s2n']['histRange'].id, 'value'),
                Input(mainControls['weightCut'].id, 'value'),
                Input(mainControls['array'].id, 'value'),
                Input(mainControls['trimEdgeSelector'].id, 'value'),
            ],
            prevent_initial_call=True
        )
        def histRangeChange(path, hrange, weightCut, array, trimEdge):
            if((path == '') | (path is None) | (array is None)):
                raise PreventUpdate
            if(hrange[0] == -99):
                hrange[0] = -5
            hfig, imfig = getFancyFig('sig2noise_I', path, hrange, weightCut, array, trimEdge)
            return [hfig, imfig]

        

# Returns the control for the weight cut
def getWeightCutControl(box):
    weightRange = box.child(
        dcc.Slider,
        min=0.0,
        max=1.0,
        step=0.1,
        value = 0.2,
        tickformat=".2f",
    )
    return weightRange


# Returns the containers and controls for a fancy figure
def getFancyFigContainer(box):
    # The top row has the histogram and controls
    histRow = box.child(dbc.Row)
    foo = histRow.child(dbc.Col, width=1)
    histPlot = histRow.child(dbc.Col, width=10).child(
            dcc.Loading, type='circle').child(dcc.Graph)
    rangeRow = box.child(dbc.Row)
    bar = rangeRow.child(dbc.Col, width=1)
    histRange = rangeRow.child(dbc.Col, width=10).child(
        dcc.RangeSlider,
        min=-99,
        max=99,
        value = [-99, 99],
        allowCross=False,
        tickformat=".2f",
    )
    imagePlot = box.child(dbc.Row).child(dcc.Graph)
    ffContainer = {'histPlot': histPlot,
                   'histRange': histRange,
                   'imagePlot': imagePlot}
    return ffContainer


@functools.lru_cache()
def getImage(name, path, weightCut, array, trimEdge=False):
    tsf = ToltecSignalFits(path=path, array=array)
    tsf.setWeightCut(weightCut)
    image = tsf.getMap(name)
    wcs = tsf.getMapWCS(name)

    if(trimEdge):
        # generate a cutout to trim out the zeros on the border
        nz = np.nonzero(image)
        xsize = nz[0].max()-nz[0].min()
        ysize = nz[1].max()-nz[1].min()
        xpos = int((nz[0].min() + nz[0].max())/2.)
        ypos = int((nz[1].min() + nz[1].max())/2.)
        signal = Cutout2D(image, (ypos, xpos), (ysize, xsize), wcs=wcs)
        image = signal.data
        wcs = signal.wcs
    return image, wcs


def getFancyFig(name, path, ranges, weightCut, array, trimEdge):

    # fetch the image and coordinates
    image, wcs = getImage(name, path, weightCut, array, trimEdge=trimEdge)
    s = image.shape
    x = wcs.pixel_to_world(np.arange(0, s[0]), 0)
    y = wcs.pixel_to_world(0, np.arange(0, s[1]))
    if(wcs.wcs.ctype[0] == 'AZOFFSET'):
        x = x[0]
        y = y[1]
        if max(image.shape) < 1000:
            units = u.arcsec
            ulabel = "[arcsec]"
        elif max(image.shape) < 3600:
            units = u.arcmin
            ulabel = "[arcmin]"
        else:
            units = u.deg
            ulabel = "[deg.]"
        x = x.to(units).value
        y = y.to(units).value
    elif(wcs.wcs.ctype[0] == 'RA---TAN'):
        ulabel = "[deg.]"
        x = x.ra.value
        y = y.dec.value
        
    # Flatten and get rid of the zeros
    signal = image.flatten()
    signal = signal[np.where(signal != 0.)[0]]

    # Deal with ranges
    if(ranges[0] == -99):
        ranges[0] = signal.min()
    if(ranges[1] == 99):
        ranges[1] = signal.max()

    # Generate the histogram figure
    h, bins = np.histogram(signal, bins=50, range=ranges)
    bincenters = np.array([(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)])
    hfig = go.Figure()
    hfig.add_trace(go.Bar(x=bincenters, y=h, name=''))
    hfig.update_layout(plot_bgcolor='white')
    hfig.update_yaxes(visible=False, showticklabels=False)
    margin=go.layout.Margin(
        l=10, #left margin
        r=10, #right margin
        b=5, #bottom margin
        t=40, #top margin
    )
    plotTitle = name
    hfig.update_layout(margin=margin, height=60,
                       title={'text': plotTitle,
                              'x': 0.5,},
                       )

    
    # Generate the image figure
    if((image.shape[0]>2000) | (image.shape[1]>2000)):
        bs=True
    else:
        bs=False

    if(bs):
        imfig = px.imshow(image, x=x, y=y, zmin=ranges[0], zmax=ranges[1],
                          binary_string=bs, origin='lower')
    else:
        imfig = go.Figure()
        imfig.add_trace(go.Heatmap(
            z=image,
            x=x,
            y=y,
            zmin=ranges[0],
            zmax=ranges[1],
        ))

    imfig.update_layout(
        uirevision=True,
        showlegend=False,
        autosize=True,
        plot_bgcolor='white'
    )
    imfig.update_xaxes(range=[x.min(), x.max()],
                       title=wcs.wcs.ctype[0]+" "+ulabel)
    imfig.update_yaxes(range=[y.min(), y.max()], scaleanchor = 'x',
                       title=wcs.wcs.ctype[1]+" "+ulabel)

    # px.imshow(image, x=x, y=y, zmin=ranges[0], zmax=ranges[1], binary_string=bs, origin='lower')
    margin=go.layout.Margin(
        l=10, #left margin
        r=10, #right margin
        b=20, #bottom margin
        t=0, #top margin
    )
    imfig.update_layout(margin=margin, height=400)
    return [hfig, imfig]


# common figure axis definitions
def getXYAxisLayouts():
    xaxis = dict(
        titlefont=dict(size=20),
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='black',
        linewidth=4,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=18,
            color='rgb(82, 82, 82)',
        ),
    )

    yaxis = dict(
        titlefont=dict(size=20),
        scaleanchor = 'x',
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='black',
        linewidth=4,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=18,
            color='rgb(82, 82, 82)',
        ),
    )
    return xaxis, yaxis


def getArrayOptions(path, options):
    g = glob(path+"toltec*.fits")
    for j, a in enumerate(['a1100', 'a1400', 'a2000']):
        f = [i for i in g if a in i]
        if(len(f) > 0):
            options[j]['disabled'] = False
        else:
            options[j]['disabled'] = True
    return options


DASHA_SITE = {
    "dasha": {
        "template": ToltecSignalFitsViewer,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", True),
    }
}
