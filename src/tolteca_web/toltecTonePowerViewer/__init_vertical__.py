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
        telSelectCol = telSelectRow.child(dbc.Col, width=6)
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
        telSelectBox.child(dbc.Row).child(html.Br)
        telSelectBox.child(dbc.Row).child(html.Hr)

        # Add images to left side of window
        bb = bigBox.child(dbc.Row)
        powerCol = bb.child(dbc.Col, width=11)

        dacRow = powerCol.child(dbc.Row)
        image = "01_DAC.png"
        dacFig, dacVals = insertImage(dacRow, image)

        driveAttenRow = powerCol.child(dbc.Row)
        image = "02_DriveAttenuator.png"
        driveAttenFig, driveAttenVals = insertImage(driveAttenRow, image)

        driveIFOutRow = powerCol.child(dbc.Row)
        image = "03_Node.png"
        driveIFOutFig, driveIFOutVals = insertImage(driveIFOutRow, image)
        
        cableAttenRow = powerCol.child(dbc.Row)
        image = "04_CableAttenuation.png"
        cableAttenFig, cableAttenVals = insertImage(cableAttenRow, image)
        for k in cableAttenVals.keys():
            cableAttenVals[k].children='-35 dB'

        kidsInRow = powerCol.child(dbc.Row)
        image = "03_Node.png"
        kidsInFig, kidsInVals = insertImage(kidsInRow, image)
        
        kidsRow = powerCol.child(dbc.Row)
        image = "05_KIDS.png"
        kidsFig = insertImage(kidsRow, image, noVals=True)
        
        lnaRow = powerCol.child(dbc.Row)
        image = "06_LNA.png"
        lnaFig = insertImage(lnaRow, image, noVals=True, size='tall')
        
        lnaValsRow = lnaRow.child(dbc.Col, width=11, align='center').child(dbc.Row)
        lnaVals = dict()
        for i in range(13):
            if(i==10):
                continue
            lnaVals['N{}'.format(i)] = lnaValsRow.child(dbc.Col, width=1, align='center').child(
                daq.Gauge,
                color={"gradient":False,"ranges":{"green":[60,100],"yellow":[50,60],"red":[0,50]}},
                value=70,
                max=100,
                min=0,
                size=80,
                showCurrentValue=True,
                scale={'interval': 10},
                style={'margin-left': '0px', 'margin-top': '-40px', 'margin-bottom': '0px', 'margin-right': '0px'},
            )

        lnaOutRow = powerCol.child(dbc.Row)
        image = "03_Node.png"
        lnaOutFig, lnaOutVals = insertImage(lnaOutRow, image)

        cableAtten2Row = powerCol.child(dbc.Row)
        image = "04_CableAttenuation.png"
        cableAtten2Fig, cableAtten2Vals = insertImage(cableAtten2Row, image)
        for k in cableAtten2Vals.keys():
            cableAtten2Vals[k].children='-3 dB'
        
        ifInputAmpRow = powerCol.child(dbc.Row)
        image = "07_IFInputAmp.png"
        ifInputAmpFig = insertImage(ifInputAmpRow, image, noVals=True)
        
        ifInputAmpOutRow = powerCol.child(dbc.Row)
        image = "03_Node.png"
        ifInputAmpOutFig, ifInputAmpOutVals = insertImage(ifInputAmpOutRow, image)

        cableAtten3Row = powerCol.child(dbc.Row)
        image = "04_CableAttenuation.png"
        cableAtten3Fig, cableAtten3Vals = insertImage(cableAtten3Row, image)
        for k in cableAtten3Vals.keys():
            cableAtten3Vals[k].children='-12 dB'

        ifInputAmp2Row = powerCol.child(dbc.Row)
        image = "07_IFInputAmp.png"
        ifInputAmp2Fig = insertImage(ifInputAmp2Row, image, noVals=True)

        ifInputAmpOut2Row = powerCol.child(dbc.Row)
        image = "03_Node.png"
        ifInputAmpOut2Fig, ifInputAmpOut2Vals = insertImage(ifInputAmpOut2Row, image)

        senseAttenRow = powerCol.child(dbc.Row)
        image = "02_DriveAttenuator.png"
        senseAttenFig, senseAttenVals = insertImage(senseAttenRow, image)
        
        senseAttenOutRow = powerCol.child(dbc.Row)
        image = "03_Node.png"
        senseAttenOutFig, senseAttenOutVals = insertImage(senseAttenOutRow, image)

        adcRow = powerCol.child(dbc.Row)
        image = "08_ADC.png"
        adcFig, adcVals = insertImage(adcRow, image)

        vals = {
            'Adrive': driveAttenVals,
            'driveIFOut': driveIFOutVals,
            'kidsIn': kidsInVals,
            'lnaIn': lnaVals,
            'lnaOut': lnaOutVals,
            'ifInputAmpOut': ifInputAmpOutVals,
            'ifInputAmpOut2': ifInputAmpOut2Vals,
            'Asense': senseAttenVals,
            'senseAttenOut': senseAttenOutVals,
            'ADC': adcVals,
            }
        
        super().setup_layout(app)

        self._registerCallbacks(
            app,
            obsnumList,
            powerDataStore,
            vals,
        )
        return

    
    def _registerCallbacks(
            self,
            app,
            obsnumList,
            powerDataStore,
            vals,
    ):
        # ---------------------------
        # obsnum select dropdown
        # ---------------------------
        @app.callback(
            [
                Output(powerDataStore.id, "data"),
            ],
            [
                Input(obsnumList.id, "value"),
            ],
            prevent_initial_call=True,
        )
        def obsnumListDropdown(obsnum):
            if (obsnum == "") | (obsnum is None):
                return [None]

            # fetch the files associated with this obsnum
            dPath = "/Users/wilson/Desktop/tmp/sweeps/test_sweep_viewer/data_lmt/toltec/tcs/"
            g = glob("{}toltec*/*.nc".format(dPath))
            files = []
            for f in g:
                if str(obsnum) in f:
                    files.append(f)
            files.sort()

            # extract the power data
            # powerData = fetchPowerDAC2ADC(files)
            powerData = fetchPowerADC2DAC(files)
            return [powerData]

        
        # ---------------------------
        # fill out all the values
        # ---------------------------
        outputList = []
        for k in vals.keys():
            for c in vals[k].keys():
                if(k == 'lnaIn'):
                    outputList.append(Output(vals[k][c].id, "value"))
                else:
                    outputList.append(Output(vals[k][c].id, "children"))
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
            for k in data.keys():
                for c in data[k].keys():
                    outList.append(data[k][c])
            return outList


def fetchPowerADC2DAC(files):
    # Here is the format we're aiming for.
    # Each dictionary element below has entries for the networks (except 10)
    data = {
        'Adrive': dict(),
        'driveIFOut': dict(),
        'kidsIn': dict(),
        'lnaIn': dict(),
        'lnaOut': dict(),
        'ifInputAmpOut': dict(),
        'ifInputAmpOut2': dict(),
        'Asense': dict(),
        'senseAttenOut': dict(),
        'ADC': dict(),
    }
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]:
        f = [j for j in files if 'toltec{}_'.format(i) in j]
        if len(f) == 1:
            f = f[0]
            print("Fetching data from {}".format(f))
            nc = netCDF4.Dataset(f)
            adcPower = calcAdcPower(nc.variables)
            ifBGain = ifBoardGain(i)
            Asense = float(nc.variables['Header.Toltec.SenseAtten'][:].data)
            Adrive = float(nc.variables['Header.Toltec.DriveAtten'][:].data)
                        
            dacPower = adcPower+Asense-ifBGain-27-30+2+35+Adrive
            data['Adrive']['N{}'.format(i)] = "{0:3.1f} dB".format(-Adrive)
            data['driveIFOut']['N{}'.format(i)] = "{0:3.1f} dBm".format(adcPower+Asense-ifBGain-27-30+2+35)
            data['kidsIn']['N{}'.format(i)] = "{0:3.1f} dBm".format(adcPower+Asense-ifBGain-27-30+2)
            data['lnaIn']['N{}'.format(i)] = np.abs(adcPower+Asense-ifBGain-27-30)
            data['lnaOut']['N{}'.format(i)] = "{0:3.1f} dBm".format(adcPower+Asense-ifBGain-27)
            data['ifInputAmpOut']['N{}'.format(i)] = "{0:3.1f} dBm".format(adcPower+Asense-ifBGain)
            data['ifInputAmpOut2']['N{}'.format(i)] = "{0:3.1f} dBm".format(adcPower+Asense)
            data['Asense']['N{}'.format(i)] = "{0:3.1f} dB".format(-Asense)
            data['senseAttenOut']['N{}'.format(i)] = "{0:3.1f} dBm".format(adcPower)
            adcSnap = calcAdcSnap(nc.variables)
            data['ADC']['N{}'.format(i)] = "{0:3.1f}%".format(adcSnap)
            nc.close()
        else:
            for k in data.keys():
                data[k]['N{}'.format(i)] = None
    return data


def calcAdcPower(ncVars):
    Is = np.array(ncVars['Data.Toltec.Is'][:].data, dtype='float')
    Qs = np.array(ncVars['Data.Toltec.Qs'][:].data, dtype='float')
    Is = Is-Is.mean(axis=0)
    Qs = Qs-Qs.mean(axis=0)
    Ps = Is**2 + Qs**2
    Ps = Ps.mean(axis=0).sum()
    p_adc = 10.*np.log10(Ps)
    p_dbm = p_adc -162.0
    return p_dbm
    


# Conversion from ADC power (ADCu^2) to dBm at the input to the IFs
def ifBoardGain(network):
    # IF board gain for each of the IFs.
    # index of array matches network number
    # measured on 1/28/2023
    gbIF = np.array([16.8, 5.7, 9.2, 17.1, 10.7, 17.1, 10.8,
                     14., 17., 17., 0., 17., 17.])
    return gbIF[network]

    
def fetchPowerDAC2ADC(files):
    # Here is the format we're aiming for.
    # Each dictionary element below has entries for the networks (except 10)
    data = {
        'Adrive': dict(),
        'driveIFOut': dict(),
        'kidsIn': dict(),
        'lnaIn': dict(),
        'lnaOut': dict(),
        'ifInputAmpOut': dict(),
        'ifInputAmpOut2': dict(),
        'Asense': dict(),
        'senseAttenOut': dict(),
        'ADC': dict(),
    }
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]:
        f = [j for j in files if 'toltec{}_'.format(i) in j]
        if len(f) == 1:
            f = f[0]
            print("Fetching data from {}".format(f))
            nc = netCDF4.Dataset(f)
            dacPower = calcDacPower(nc.variables)
            Adrive = float(nc.variables['Header.Toltec.DriveAtten'][:].data)
            data['Adrive']['N{}'.format(i)] = "{0:3.1f} dB".format(-Adrive)
            data['driveIFOut']['N{}'.format(i)] = "{0:3.1f} dBm".format(dacPower-Adrive)
            data['kidsIn']['N{}'.format(i)] = "{0:3.1f} dBm".format(dacPower-Adrive-35)
            data['lnaIn']['N{}'.format(i)] = np.abs(dacPower-Adrive-35-2)
            data['lnaOut']['N{}'.format(i)] = "{0:3.1f} dBm".format(dacPower-Adrive-35-2+30)
            data['ifInputAmpOut']['N{}'.format(i)] = "{0:3.1f} dBm".format(dacPower-Adrive-35-2+30-3+30)
            data['ifInputAmpOut2']['N{}'.format(i)] = "{0:3.1f} dBm".format(dacPower-Adrive-35-2+30-3+30+15)
            Asense = float(nc.variables['Header.Toltec.SenseAtten'][:].data)
            data['Asense']['N{}'.format(i)] = "{0:3.1f} dB".format(-Asense)
            data['senseAttenOut']['N{}'.format(i)] = "{0:3.1f} dBm".format(dacPower-Adrive-35-2+30-3+30+15-Asense)
            adcSnap = calcAdcSnap(nc.variables)
            data['ADC']['N{}'.format(i)] = "{0:3.1f}%".format(adcSnap)
            nc.close()
        else:
            for k in data.keys():
                data[k]['N{}'.format(i)] = None
    return data


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



def insertImage(row, image, noVals=False, size=None):
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
    if(size == 'tall'):
        height *=2
    # update layout properties
    df.update_layout(
        autosize=True,
        height=height,
        width=width,
        plot_bgcolor="white",
        margin=dict(r=0, l=0, b=0, t=0),
    )
    df.update_xaxes(showticklabels=False)
    df.update_yaxes(showticklabels=False)
    fig = row.child(dbc.Col, width=1).child(dcc.Graph(figure=df))

    if(not noVals):
        valsRow = row.child(dbc.Col, width=11, align='center').child(dbc.Row)
        vals = dict()
        for i in range(13):
            if(i==10):
                continue
            vals['N{}'.format(i)] = valsRow.child(dbc.Col, width=1, align='center').child(
                html.Center).child(html.Div, children='Net {}'.format(i))
        return fig, vals
    else:
        return fig




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
