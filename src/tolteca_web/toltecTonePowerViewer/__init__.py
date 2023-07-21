"""
To Do:
 - Needs commenting.
 - must deal with HPBW scaled parameters correctly.
"""
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
from PIL import Image
from dash import html
from glob import glob
from dash import dcc
from dash import ctx
import xarray as xr
import pandas as pd
import numpy as np
import functools
import netCDF4
import time
import json
import os


class ToltecTonePowerViewer(ComponentTemplate):
    class Meta:
        component_cls = dbc.Container

    def __init__(
        self,
        title_text="Toltec TonePower Viewer",
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
        dPath = "/Users/wilson/Desktop/tmp/sweeps/test_sweep_viewer/data_lmt/toltec/tcs/"
        g = glob("{}toltec*/*.nc".format(dPath))
        g += glob("/Users/wilson/Desktop/tmp/110232/*.nc")
        obsnums = []
        for f in g:
            b = f.split('/')[-1]
            o = b.split('_')[1]
            if o not in obsnums:
                obsnums.append(o)
        obsnums.sort()
        options = [{'label': str(o), 'value': o} for o in obsnums]
        
        # pull down to select obs stats file
        pulldownPanel, bigBox = body.grid(2, 1)
        telSelectBox = pulldownPanel.child(dbc.Row).child(dbc.Col, width=12)
        telSelectRow = telSelectBox.child(dbc.Row)
        telSelectCol = telSelectRow.child(dbc.Col, width=2)
        telTitle = telSelectCol.child(dbc.Row).child(
            html.H5, "Select Obsnum", className="mb-2"
        )
        obsnumList = telSelectCol.child(dbc.Row).child(
            dcc.Dropdown,
            options=options,
            placeholder="Select Obsnum",
            value="",
            searchable=False,
            style=dict(width="100%", verticalAlign="middle"),
        )
        powerDataStore = telSelectRow.child(dcc.Store)

        # direction
        telSelectRow.child(dbc.Col, width=2)
        directionCol = telSelectRow.child(dbc.Col, width=2)
        directionTitle= directionCol.child(dbc.Row).child(
            html.H5, "Power Estimate Direction", className="mb-2")
        directionSwitch = directionCol.child(dbc.Row).child(
            daq.ToggleSwitch, size=30, value=True, label=["DAC to ADC", "ADC to DAC"])
        controls = {
            'direction': directionSwitch,
            }
        
        telSelectBox.child(dbc.Row).child(html.Br)
        telSelectBox.child(dbc.Row).child(html.Hr)

        bigBox.child(dbc.Row).child(html.H3, "Transmission Line Total Power Flow")
        # Set up a row of images showing the power flow
        b = bigBox.child(dbc.Row)
        b1 = b.child(dbc.Col, width=1)
        b2 = b.child(dbc.Col, width=10).child(dbc.Row)
        b3 = b.child(dbc.Col, width=1)
        insertImage(b1, 'images/01h_DAC.png')
        insertImage(b2.child(dbc.Col, width=1), 'images/02h_driveAttenuator.png') 
        insertImage(b2.child(dbc.Col, width=1), 'images/03h_Node.png')
        insertImage(b2.child(dbc.Col, width=1), 'images/04h_CableAttenuation.png')
        insertImage(b2.child(dbc.Col, width=1), 'images/05h_KIDS.png')
        insertImage(b2.child(dbc.Col, width=1), 'images/03h_Node.png')
        insertImage(b2.child(dbc.Col, width=1), 'images/06h_LNA.png')
        insertImage(b2.child(dbc.Col, width=1), 'images/03h_Node.png')
        insertImage(b2.child(dbc.Col, width=1), 'images/07h_IFInputAmp.png')
        insertImage(b2.child(dbc.Col, width=1), 'images/03h_Node.png')
        insertImage(b2.child(dbc.Col, width=1), 'images/08h_IFBoardGain.png')
        insertImage(b2.child(dbc.Col, width=1), 'images/02h_driveAttenuator.png') 
        insertImage(b2.child(dbc.Col, width=1), 'images/03h_Node.png')
        insertImage(b3, 'images/09h_ADC.png')
        
        # Make a dictionary of output values
        bb = bigBox.child(dbc.Row)
        bb1 = bb.child(dbc.Col, width=1)
        bb2 = bb.child(dbc.Col, width=10)
        bb3 = bb.child(dbc.Col, width=1)
        # Labels
        bb1.child(dbc.Row).child(html.Center).child(html.B, children="Network")
        r = bb2.child(dbc.Row)
        for label in ['Drive Attn', 'IF Out', 'Cryo Atn', 'KIDs Pow', 'LNA In', 'LNA Gain',
                      'Cryo Out', 'IF Input Gain', 'IF Board In', '', 'Sense Attn', 'ADC In']:
            makeLabel(r, label)
        bb3.child(dbc.Row).child(html.Center).child(html.B, children="SnapBlock Frac")
        
        vals = dict()
        for i in range(13):
            netVals = dict()
            r1 = bb1.child(dbc.Row)
            r2 = bb2.child(dbc.Row)
            r3 = bb3.child(dbc.Row)
            netVals['network'] = r1.child(html.Center).child(html.Div, children="N{}".format(i))
            netVals['Adrive'] = makeOutputDiv(r2)
            netVals['AdriveOut'] = makeOutputDiv(r2)
            netVals['cryoAtten'] = makeOutputDiv(r2)
            netVals['kids'] = makeOutputDiv(r2)
            netVals['lnaIn'] = makeOutputDiv(r2)
            netVals['lnaGain'] = makeOutputDiv(r2)
            netVals['lnaOut'] = makeOutputDiv(r2)
            netVals['ifInGain'] = makeOutputDiv(r2)
            netVals['ifBoardIn'] = makeOutputDiv(r2)
            netVals['ifBoardGain'] = makeOutputDiv(r2)
            netVals['Asense'] = makeOutputDiv(r2)
            netVals['AdcInPower'] = makeOutputDiv(r2)
            netVals['AdcSnapFrac'] = r3.child(html.Center).child(html.Div, children="-")
            vals['Net{}'.format(i)] = netVals

        bigBox.child(dbc.Row).child(html.Br)
        bigBox.child(dbc.Row).child(html.H3, "Per-Tone Powers")
        plotRow = bigBox.child(dbc.Row)
        figs = makeEmptyFigs(2)
        ampPlot = plotRow.child(dbc.Col, width=6).child(dbc.Row).child(dcc.Graph, figure=figs[0])
        powPlot = plotRow.child(dbc.Col, width=6).child(dbc.Row).child(dcc.Graph, figure=figs[1])
        plots = {
            'ampPlot': ampPlot,
            'powPlot': powPlot,
            }
    
        super().setup_layout(app)

        self._registerCallbacks(
            app,
            obsnumList,
            powerDataStore,
            vals,
            controls,
            plots,
        )
        return

    
    def _registerCallbacks(
            self,
            app,
            obsnumList,
            powerDataStore,
            vals,
            controls,
            plots,
    ):
        # ---------------------------
        # obsnum select dropdown
        # ---------------------------
        @app.callback(
            [
                Output(powerDataStore.id, "data"),
                Output(plots['ampPlot'].id, "figure"),
                Output(plots['powPlot'].id, "figure"),
            ],
            [
                Input(obsnumList.id, "value"),
                Input(controls['direction'].id, "value"),
            ],
            prevent_initial_call=True,
        )
        def obsnumListDropdown(obsnum, direction):
            if (obsnum == "") | (obsnum is None):
                raise PreventUpdate

            # fetch the files associated with this obsnum
            dPath = "/Users/wilson/Desktop/tmp/sweeps/test_sweep_viewer/data_lmt/toltec/tcs/"
            g = glob("{}toltec*/*.nc".format(dPath))
            g += glob("/Users/wilson/Desktop/tmp/110232/*.nc")
            files = []
            for f in g:
                if str(obsnum) in f:
                    files.append(f)
            files.sort()

            # extract the power data
            if(direction):
                powerData = fetchPowerADC2DAC(files)
            else:
                powerData = fetchPowerDAC2ADC(files)

            ampFig = makeAmpFig(files)
            powFig = makePowFig(files)
            return [powerData, ampFig, powFig]

        
        # ---------------------------
        # fill out all the values
        # ---------------------------
        outputList = []
        for k in vals.keys():
            for c in vals[k].keys():
                outputList.append(Output(vals[k][c].id, "children"))
        for k in vals.keys():
            for c in vals[k].keys():
                outputList.append(Output(vals[k][c].id, "style"))
        @app.callback(
            outputList,
            [
                Input(powerDataStore.id, "data"),
            ],
            prevent_initial_call=True,
        )
        def fillOutValues(data):
            if (data == "") | (data is None):
                raise PreventUpdate

            outList = []
            # take care of the values first
            for k in data.keys():
                for c in data[k].keys():
                    outList.append(data[k][c])

            # use data validation to set the styles
            styles = validateData(data)
            
            # set the styles
            for k in data.keys():
                for c in data[k].keys():
                    outList.append(styles[k][c])            
            return outList


def makeAmpFig(files):
    fig = go.Figure()
    xaxis, yaxis = getXYAxisLayouts()
    for i in range(13):
        f = [j for j in files if 'toltec{}_'.format(i) in j]
        if len(f) == 1:
            f = f[0]
            nc = netCDF4.Dataset(f)
            toneFreq = nc.variables['Header.Toltec.ToneFreq'][:].data.T[:,0]
            toneAmps = nc.variables['Header.Toltec.ToneAmps'][:].data
            s = np.argsort(toneFreq)
            fig.add_trace(
                go.Scatter(
                    x = toneFreq[s]*1.e-6,
                    y = toneAmps[s],
                    name="N{}".format(i)
                ),
            )
    fig.update_layout(
        height=400,
        plot_bgcolor="white",
        xaxis=xaxis,
        yaxis=yaxis,
        font={'size': 12,},
        title="Unnormalized Tone Amplitudes",
        xaxis_title="Tone Frequency [MHz]",
        yaxis_title="Amplitude (unnormalized)", 
        margin=go.layout.Margin(
            l=10,
            r=10,
            b=5,
            t=40,
        ),)
    return fig


def makePowFig(files):
    fig = go.Figure()
    xaxis, yaxis = getXYAxisLayouts()
    for i in range(13):
        f = [j for j in files if 'toltec{}_'.format(i) in j]
        if len(f) == 1:
            f = f[0]
            nc = netCDF4.Dataset(f)
            Ps = calcPerTonePower(nc.variables)
            Asense = float(nc.variables['Header.Toltec.SenseAtten'][:].data)
            Ps = calcPowerAtKids(Ps, i, Asense)
            toneFreq = nc.variables['Header.Toltec.ToneFreq'][:].data.T[:,0]
            s = np.argsort(toneFreq)
            fig.add_trace(
                go.Scatter(
                    x = toneFreq[s]*1.e-6,
                    y = Ps[s],
                    name="N{}".format(i)
                ),
            )
    fig.update_layout(
        height=400,
        plot_bgcolor="white",
        xaxis=xaxis,
        yaxis=yaxis,
        font={'size': 12,},
        title="Per-Tone Powers at KIDs",
        xaxis_title="Tone Frequency [MHz]",
        yaxis_title="ADC Tone Power [dBm]", 
        margin=go.layout.Margin(
            l=10,
            r=10,
            b=5,
            t=40,
        ),)
    return fig

    

# returns a matching dictionary of css style dicts depending on some data values
def validateData(data):
    style = data.copy()
    for net in style.keys():
        style[net] = data[net].copy()
        for k in style[net].keys():
            style[net][k] = {'color': 'blue'}
            if ((k == 'lnaIn') & (data[net][k] != "-")):
                d = float(data[net][k].replace('dBm', ''))
                if(d > -52):
                    style[net][k] = {'color': 'red'}
            elif ((k == 'lnaOut') & (data[net][k] != "-")):
                d = float(data[net][k].replace('dBm', ''))
                if(d > -15):
                    style[net][k] = {'color': 'red'}
            elif ((k == 'ifBoardIn') & (data[net][k] != "-")):
                d = float(data[net][k].replace('dBm', ''))
                if(d > -5):
                    style[net][k] = {'color': 'red'}
            elif ((k == 'AdcSnapFrac') & (data[net][k] != "-")):
                d = float(data[net][k].replace('%', ''))
                if(d < 30):
                    style[net][k] = {'color': 'red'}
    return style


    
def fetchPowerADC2DAC(files):
    vals = dict()
    for i in range(13):
        netVals = dict()
        netVals['network'] = 'N{}'.format(i)
        netVals['Adrive'] = '-'
        netVals['AdriveOut'] = '-'
        netVals['cryoAtten'] = '-'
        netVals['kids'] = '-'
        netVals['lnaIn'] = '-'
        netVals['lnaGain'] = '-'
        netVals['lnaOut'] = '-'
        netVals['ifInGain'] = '-'
        netVals['ifBoardIn'] = '-'
        netVals['ifBoardGain'] = '-'
        netVals['Asense'] = '-'
        netVals['AdcInPower'] = '-'
        netVals['AdcSnapFrac'] = '-'
        vals['Net{}'.format(i)] = netVals
    for i in range(13):
        f = [j for j in files if 'toltec{}_'.format(i) in j]
        if len(f) == 1:
            f = f[0]
            #print("Fetching data from {}".format(f))
            nc = netCDF4.Dataset(f)
            adcPower = calcAdcPower(nc.variables)
            ifBGain = ifBoardGain(i)
            Asense = float(nc.variables['Header.Toltec.SenseAtten'][:].data)
            Adrive = float(nc.variables['Header.Toltec.DriveAtten'][:].data)

            data = vals['Net{}'.format(i)]
            dacPower = adcPower+Asense-ifBGain-27-30+2+35+Adrive
            data['Adrive'] = "{0:3.1f} dB".format(-Adrive)
            data['AdriveOut'] = "{0:3.1f} dBm".format(adcPower+Asense-ifBGain-27-30+2+35)
            data['cryoAtten'] = '-35 dB'
            data['kids'] = "{0:3.1f} dBm".format(adcPower+Asense-ifBGain-27-30+2)
            data['lnaIn'] = "{0:3.1f} dBm".format(adcPower+Asense-ifBGain-27-30)
            data['lnaGain'] = '30 dB'
            data['lnaOut'] = "{0:3.1f} dBm".format(adcPower+Asense-ifBGain-27)
            data['ifInGain'] = "30 dB"
            data['ifBoardIn'] = "{0:3.1f} dBm".format(adcPower+Asense-ifBGain)
            data['ifBoardGain'] = "{0:3.1f} dB".format(ifBGain)
            data['Asense'] = "{0:3.1f} dB".format(-Asense)
            data['AdcInPower'] = "{0:3.1f} dBm".format(adcPower)
            adcSnap = calcAdcSnap(nc.variables)
            data['AdcSnapFrac'] = "{0:3.1f}%".format(adcSnap)
            nc.close()
    return vals


def fetchPowerDAC2ADC(files):
    vals = dict()
    for i in range(13):
        netVals = dict()
        netVals['network'] = 'N{}'.format(i)
        netVals['Adrive'] = '-'
        netVals['AdriveOut'] = '-'
        netVals['cryoAtten'] = '-'
        netVals['kids'] = '-'
        netVals['lnaIn'] = '-'
        netVals['lnaGain'] = '-'
        netVals['lnaOut'] = '-'
        netVals['ifInGain'] = '-'
        netVals['ifBoardIn'] = '-'
        netVals['ifBoardGain'] = '-'
        netVals['Asense'] = '-'
        netVals['AdcInPower'] = '-'
        netVals['AdcSnapFrac'] = '-'
        vals['Net{}'.format(i)] = netVals
    for i in range(13):
        f = [j for j in files if 'toltec{}_'.format(i) in j]
        if len(f) == 1:
            f = f[0]
            #print("Fetching data from {}".format(f))
            nc = netCDF4.Dataset(f)
            ifBGain = ifBoardGain(i)
            Asense = float(nc.variables['Header.Toltec.SenseAtten'][:].data)
            Adrive = float(nc.variables['Header.Toltec.DriveAtten'][:].data)

            data = vals['Net{}'.format(i)]
            dacPower = -15.1
            data['Adrive'] = "{0:3.1f} dB".format(-Adrive)
            data['AdriveOut'] = "{0:3.1f} dBm".format(dacPower-Adrive)
            data['cryoAtten'] = '-35 dB'
            data['kids'] = "{0:3.1f} dBm".format(dacPower-Adrive-35)
            data['lnaIn'] = "{0:3.1f} dBm".format(dacPower-Adrive-35-2)
            data['lnaGain'] = '30 dB'
            data['lnaOut'] = "{0:3.1f} dBm".format(dacPower-Adrive-35-2+27)
            data['ifInGain'] = "30 dB"
            data['ifBoardIn'] = "{0:3.1f} dBm".format(dacPower-Adrive-35-2+27+30)
            data['ifBoardGain'] = "{0:3.1f} dB".format(ifBGain)
            data['Asense'] = "{0:3.1f} dB".format(-Asense)
            data['AdcInPower'] = "{0:3.1f} dBm".format(dacPower-Adrive-35-2+27+30+ifBGain-Asense)
            adcSnap = calcAdcSnap(nc.variables)
            data['AdcSnapFrac'] = "{0:3.1f}%".format(adcSnap)
            nc.close()
    return vals


def calcAdcPower(ncVars):
    Ps = calcPerTonePower(ncVars)
    Ps = 10.**(Ps/10.)
    Ps = Ps.sum()
    Ps = 10.*np.log10(Ps)
    return Ps
    

def calcPerTonePower(ncVars):
    Is = np.array(ncVars['Data.Toltec.Is'][:].data, dtype='float')
    Qs = np.array(ncVars['Data.Toltec.Qs'][:].data, dtype='float')
    Ps = Is**2 + Qs**2
    Ps = 10.*np.log10(Ps.mean(axis=0))
    Ps -= 162.0
    return Ps


def calcPowerAtKids(Ps, network, Asense):
    Ps += (np.abs(Asense) - ifBoardGain(network) - 30 - 27. +2.)
    return Ps


# Conversion from ADC power (ADCu^2) to dBm at the input to the IFs
def ifBoardGain(network):
    # IF board gain for each of the IFs.
    # index of array matches network number
    # measured on 1/28/2023
    gbIF = np.array([16.8, 5.7, 9.2, 17.1, 10.7, 17.1, 10.8,
                     14., 17., 17., 0., 17., 17.])
    return gbIF[network]


# Converts the snap block readings to 12 bits and returns the percentage of the full scale ranged used.
def calcAdcSnap(ncVars):
    snap = ncVars['Header.Toltec.AdcSnapData'][:]
    # convert from 16 bits to 12 bits
    x0 = snap[0].view(np.int16)/16
    x1 = snap[1].view(np.int16)/16
    r0 = (x0.max()-x0.min())/2**12
    r1 = (x1.max()-x1.min())/2**12
    return np.array([r0, r1]).mean()*100.


# Estimates the total power out of the DAC in dBm
def calcDacPower(ncVars):
    # This is how I would have calculated it from first principles and the actual c-code
    if(0):
        combFftLen = 2**21
        sampleFreq = int(512e6)
        daqFullScale = (1 << 15) - 1
        VmaxDaq = 1.2
        toneFreq = ncVars['Header.Toltec.ToneFreq'][:].data.T[:,0]
        toneAmps = ncVars['Header.Toltec.ToneAmps'][:].data
        combLen = len(toneAmps)
        phases = np.random.uniform(0., 2.*np.pi, size=len(toneAmps))
        icomplex = complex(0., 1.)
        spec = np.zeros(combFftLen, dtype=complex)
        for i in range(len(toneFreq)):
            bin = fft_bin_idx(toneFreq, i, combFftLen, sampleFreq)
            spec[bin] = toneAmps[i]*np.exp(icomplex*phases[i])
        wave = np.fft.fft(spec, norm='backward')
        wave /= combFftLen
        wave /= np.abs(wave).max()
        wave *= VmaxDaq
        Vrms2 = (np.abs(wave)**2).mean()
        Prms = Vrms2/50.
        return 10.*np.log10(Prms*1.e3)
    # But this is what Adrian's spreadsheet says based off of measurements.
    return -15.1


def fft_bin_idx(freq, freq_idx, FFT_LEN, samp_freq):
    return int(round(freq[freq_idx] / samp_freq * FFT_LEN))



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
            yaxis=yaxis,
        margin=go.layout.Margin(
            l=10,
            r=10,
            b=5,
            t=40,
        ))
        figs.append(fig)
    return figs



def insertImage(col, image):
    image = Image.open(image)
    df = go.Figure()
    df.add_layout_image(
        dict(
            source=image,
            xref="paper", yref="paper",
            x=1, y=0,
            sizex=1.0, sizey=1.0,
            xanchor="right", yanchor="bottom"
        )
    )
    width=120
    height=60
    df.update_layout(
        autosize=True,
        height=height,
        width=width,
        plot_bgcolor="white",
        margin=dict(r=0, l=0, b=0, t=0),
    )
    df.update_xaxes(showticklabels=False)
    df.update_yaxes(showticklabels=False)
    fig = col.child(dcc.Graph(figure=df))
    return


def makeOutputDiv(r2):
    return r2.child(dbc.Col, width=1, align='center').child(html.Center).child(html.Div, children='-')

def makeLabel(row, label):
    return row.child(dbc.Col, width=1, align='center').child(html.Center).child(html.B, children=label)


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
            size=12,
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
            size=12,
            color="rgb(82, 82, 82)",
        ),
    )
    return xaxis, yaxis


def rad2arcsec(angle):
    return np.rad2deg(angle)*3600.

def rad2arcmin(angle):
    return np.rad2deg(angle)*60.



DASHA_SITE = {
    "dasha": {
        "template": ToltecTonePowerViewer,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", True),
    }
}
