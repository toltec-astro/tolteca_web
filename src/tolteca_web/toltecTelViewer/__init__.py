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
from ..base import ViewerBase


class ToltecTelViewer(ViewerBase):
    class Meta:
        component_cls = dbc.Container

    def __init__(
        self,
        title_text="Toltec Tel Viewer",
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
        dPath = "/Users/wilson/Desktop/tmp/cosmos/telFiles/"
        g = glob("{}tel*.nc".format(dPath))
        g = g+glob("/Users/wilson/Desktop/tmp/macs0717/pointing/telFiles/tel*.nc")
        g = g+glob("/Users/wilson/Desktop/tmp/macs0717/science/telFiles/tel*.nc")
        g.sort()
        telFiles = g
        telOptions = [{"label": p, "value": p} for p in telFiles]

        # pull down to select obs stats file
        pulldownPanel, bigBox = body.grid(2, 1)
        telSelectBox = pulldownPanel.child(dbc.Row).child(dbc.Col, width=12)
        telSelectRow = telSelectBox.child(dbc.Row)
        telSelectCol = telSelectRow.child(dbc.Col, width=6)
        telTitle = telSelectCol.child(dbc.Row).child(
            html.H5, "TEL File", className="mb-2"
        )
        telList = telSelectCol.child(dbc.Row).child(
            dcc.Dropdown,
            options=telOptions,
            placeholder="Select Tel File",
            value="",
            searchable=False,
            style=dict(width="100%", verticalAlign="middle"),
        )
        self.state_manager.register("telList", telList, ("options", "value"))
        headerDataStore = telSelectRow.child(dcc.Store)

        downBox = telSelectRow.child(dbc.Col, width=1)
        downTitle = downBox.child(dbc.Row).child(
            html.H5, "Downsample", className="mb-2"
        )
        downsample = downBox.child(dbc.Row).child(
            dcc.Input, value=10.,
            min=1., max=50.,
            debounce=True, type='number',
            style={'width': '75%',
                   'margin-right': '20px',
                   'margin-left': '20px'})

        frameBox = telSelectRow.child(dbc.Col, width=3)
        frameTitle = frameBox.child(dbc.Row).child(
            html.H5, "Frame", className="mb-2"
        )
        frame = frameBox.child(dbc.Row).child(
            dcc.RadioItems, options=[
                {'label': 'Telescope', 'value': 'telescope'},
                {'label': 'Source', 'value': 'source'},
            ],
            value='source',
            labelStyle={'display': 'inline-block'},
            inputStyle={"margin-right": "5px",
                        "margin-left": "20px"},)

        controls = {
            'downsample': downsample,
            'frame': frame,
            }
        
        pulldownPanel.child(dbc.Row).child(html.Br)
        pulldownPanel.child(dbc.Row).child(html.Hr)

        # The bigBox gets divided into a column (or two) of Header
        # Data and a column of plots.
        bb = bigBox.child(dbc.Row)
        headerBox = bb.child(dbc.Col, width=2)
        scanBox = bb.child(dbc.Col, width=2)
        header = dict(
            obsnum = makeHeaderEntry(headerBox, "Dcs", ["ObsNum"]),
            source = makeHeaderEntry(headerBox, "Source", ["SourceName", "Ra", "Dec"]),
            radiometer = makeHeaderEntry(headerBox, "Radiometer", ['Tau', 'UpdateDate']),
            sky = makeHeaderEntry(headerBox, "Sky", ['AzReq', 'ElReq', 'AzOff', 'ElOff']),
            telescope = makeHeaderEntry(headerBox, "Telescope", ['CraneInBeam']),
            m1 = makeHeaderEntry(headerBox, "M1", ['ZernikeC']),
            m2 = makeHeaderEntry(headerBox, "M2", ['ZReq', 'Alive']),
            m3 = makeHeaderEntry(headerBox, "M3", ['Alive', 'Fault']),
            )
        scan = dict(
            map = makeHeaderEntry(scanBox, "Raster", ['NumRepeats', 'NumScans', 'ScanAngle',
                                                      'XYLength', 'XYStep',
                                                      'XYRamp', 'ScanRate', 'MapCoord',
                                                      'MapMotion', 'MapPath'],
                                  bgColor='green'),
            liss = makeHeaderEntry(scanBox, "Lissajous",
                                   ['XYLength', 'XYOmega', 'XDelta',
                                    'XYLenMinor', 'XYOmMinor',
                                    'XDeltaMinor', 'ScanRate', 'TScan', 'CoordSys'],
                                   bgColor='Green'),
            )

        # A container for the plots
        plotsBox = bb.child(dbc.Col, width=8)
        tRow = plotsBox.child(dbc.Row)
        trajPlot = tRow.child(dbc.Col, width=6).child(dcc.Graph)
        velPlot = tRow.child(dbc.Col, width=6).child(dcc.Graph)
        mRow = plotsBox.child(dbc.Row)
        errPlot = mRow.child(dbc.Col, width=6).child(dcc.Graph)
        accPlot = mRow.child(dbc.Col, width=6).child(dcc.Graph)
        plots = {
            'trajectory': trajPlot,
            'velocity': velPlot,
            'error': errPlot,
            'acceleration': accPlot,
            }

        # Timestream plots
        tt = bigBox.child(dbc.Row)
        ttAz = tt.child(dbc.Col, width=6).child(dcc.Graph)
        ttEl = tt.child(dbc.Col, width=6).child(dcc.Graph)
        timestream = {
            'az': ttAz,
            'el': ttEl,
            }
        
        super().setup_layout(app)

        self._registerCallbacks(
            app,
            telList,
            headerDataStore,
            header,
            scan,
            plots,
            timestream,
            controls, 
        )
        return

    
    def _registerCallbacks(
            self,
            app,
            telList,
            headerDataStore,
            header,
            scan,
            plots,
            timestream,
            controls,
    ):

        # ---------------------------
        # telFile select dropdown
        # ---------------------------
        @app.callback(
            [
                Output(headerDataStore.id, "data"),
            ],
            [
                Input(telList.id, "value"),
            ],
            prevent_initial_call=True,
        )
        def telListDropdown(telFile):
            if (telFile == "") | (telFile is None):
                return [None]
            headerData = fetchTelHeaderData(telFile)
            return [headerData]

        # ---------------------------
        # Header column values
        # ---------------------------
        @app.callback(
            [
                Output(header['obsnum']['ObsNum']['value'].id, "children"),
                Output(header['source']['SourceName']['value'].id, "children"),
                Output(header['source']['Ra']['value'].id, "children"),
                Output(header['source']['Dec']['value'].id, "children"),
                Output(header['radiometer']['Tau']['value'].id, "children"),
                Output(header['radiometer']['UpdateDate']['value'].id, "children"),
                Output(header['sky']['AzReq']['value'].id, "children"),
                Output(header['sky']['ElReq']['value'].id, "children"),
                Output(header['sky']['AzOff']['value'].id, "children"),
                Output(header['sky']['ElOff']['value'].id, "children"),
                Output(header['telescope']['CraneInBeam']['value'].id, "children"),
                Output(header['m1']['ZernikeC']['value'].id, "children"),
                Output(header['m2']['ZReq']['value'].id, "children"),
                Output(header['m2']['Alive']['value'].id, "children"),
                Output(header['m3']['Alive']['value'].id, "children"),
                Output(header['m3']['Fault']['value'].id, "children"),
            ],
            [
                Input(headerDataStore.id, "data"),
            ],
            prevent_initial_call=True,
        )
        def headerDataUpdate(data):
            if (data == "") | (data is None):
                return ["-"]*16
            d = []
            d.append(data['Dcs']['ObsNum'])
            d.append(data['Source']['SourceName'])
            d.append("{0:3.5f} deg.".format(np.rad2deg(data['Source']['Ra'])))
            d.append("{0:3.5f} deg.".format(np.rad2deg(data['Source']['Dec'])))
            d.append("{0:3.2f}".format(data['Radiometer']['Tau']))
            d.append(data['Radiometer']['UpdateDate'])
            d.append("{0:3.2f} deg.".format(np.rad2deg(data['Sky']['AzReq'])))
            d.append("{0:3.2f} deg.".format(np.rad2deg(data['Sky']['ElReq'])))
            d.append("{0:3.2f} deg.".format(np.rad2deg(data['Sky']['AzOff'])))
            d.append("{0:3.2f} deg.".format(np.rad2deg(data['Sky']['ElOff'])))
            if(data['Telescope']['CraneInBeam']):
                d.append(" YES!!!")
            else:
                d.append(" Nope")
            d.append(data['M1']['ZernikeC'])
            d.append("{0:3.2f} mm".format(data['M2']['ZReq']))
            d.append(data['M2']['Alive'])
            d.append(data['M3']['Alive'])
            d.append(data['M3']['Fault'])
            return d


        # ---------------------------
        # Scan column values
        # ---------------------------
        @app.callback(
            [
                Output(scan['map']['NumRepeats']['value'].id, 'children'),
                Output(scan['map']['NumScans']['value'].id, 'children'),
                Output(scan['map']['ScanAngle']['value'].id, 'children'),
                Output(scan['map']['XYLength']['value'].id, 'children'),
                Output(scan['map']['XYStep']['value'].id, 'children'),
                Output(scan['map']['XYRamp']['value'].id, 'children'),
                Output(scan['map']['ScanRate']['value'].id, 'children'),
                Output(scan['map']['MapCoord']['value'].id, 'children'),
                Output(scan['map']['MapMotion']['value'].id, 'children'),
                Output(scan['map']['MapPath']['value'].id, 'children'),
                Output(scan['liss']['XYLength']['value'].id, 'children'),
                Output(scan['liss']['XYOmega']['value'].id, 'children'),
                Output(scan['liss']['XDelta']['value'].id, 'children'),
                Output(scan['liss']['XYLenMinor']['value'].id, 'children'),
                Output(scan['liss']['XYOmMinor']['value'].id, 'children'),
                Output(scan['liss']['XDeltaMinor']['value'].id, 'children'),
                Output(scan['liss']['ScanRate']['value'].id, 'children'),
                Output(scan['liss']['TScan']['value'].id, 'children'),
                Output(scan['liss']['CoordSys']['value'].id, 'children'),
            ],
            [
                Input(headerDataStore.id, "data"),
            ],
            prevent_initial_call=True,
        )
        def scanDataUpdate(data):
            if (data == "") | (data is None):
                return ["-"]*19
            d = []
            if(data['Map'] == {}):
                d = d+['-']*10
            else:            
                d.append(data['Map']['NumRepeats'])
                d.append(data['Map']['NumScans'])
                d.append("{0:3.2f} deg".format(np.rad2deg(data['Map']['ScanAngle'])))
                xlen = rad2arcmin(data['Map']['XLength'])
                ylen = rad2arcmin(data['Map']['YLength'])
                d.append("{0:3.2f}, {1:3.2f} arcmin".format(xlen, ylen))
                xStep = data['Map']['XStep']
                yStep = data['Map']['YStep']
                d.append("{0:3.2f}, {1:3.2f}".format(xStep, yStep))
                xRamp = data['Map']['XRamp']
                yRamp = data['Map']['YRamp']
                d.append("{0:3.2f}, {1:3.2f}".format(xRamp, yRamp))
                d.append('{0:3.0f} "/s'.format(rad2arcsec(data['Map']['ScanRate'])))
                d.append(data['Map']['MapCoord'])
                d.append(data['Map']['MapMotion'])
                d.append(data['Map']['MapPath'])
            xlen = rad2arcmin(data['Lissajous']['XLength'])
            ylen = rad2arcmin(data['Lissajous']['YLength'])
            d.append("{0:3.1f}, {1:3.1f} arcmin".format(xlen, ylen))
            xOmega = data['Lissajous']['XOmega']
            yOmega = data['Lissajous']['YOmega']
            d.append("{0:3.2f}, {1:3.2f}".format(xOmega, yOmega))
            d.append(data['Lissajous']['XDelta'])
            xLengthMinor = rad2arcmin(data['Lissajous']['XLengthMinor'])
            yLengthMinor = rad2arcmin(data['Lissajous']['YLengthMinor'])
            d.append("{0:3.1f}, {1:3.1f} arcmin".format(xLengthMinor, yLengthMinor))
            xOmegaMinor = data['Lissajous']['XOmegaMinor']
            yOmegaMinor = data['Lissajous']['YOmegaMinor']
            d.append("{0:3.2f}, {1:3.2f}".format(xOmegaMinor, yOmegaMinor))
            d.append(data['Lissajous']['XDeltaMinor'])
            d.append('{0:3.0f} "/s'.format(rad2arcsec(data['Lissajous']['ScanRate'])))
            d.append(data['Lissajous']['TScan'])
            d.append(data['Lissajous']['CoordSys'])
            return d
        

        # ---------------------------
        # Plots
        # ---------------------------
        @app.callback(
            [
                Output(plots['trajectory'].id, "figure"),
                Output(plots['velocity'].id, "figure"),
                Output(plots['error'].id, "figure"),
                Output(plots['acceleration'].id, "figure"),
                Output(timestream['az'].id, "figure"),
                Output(timestream['el'].id, "figure"),
            ],
            [
                Input(telList.id, "value"),
                Input(controls['downsample'].id, "value"),
                Input(controls['frame'].id, "value"),
            ],
            prevent_initial_call=False,
        )
        def fileUpdate(telFile, downsample, frame):
            time.sleep(0.5)
            if (telFile == "") | (telFile is None):
                return makeEmptyFigs(6)
            plotData = fetchTelPlotData(telFile, downsample)
            trajectory = makeTrajectoryPlot(plotData, frame)
            velocity, acceleration = makeVelocityPlot(plotData, frame)
            error = makeErrorPlot(plotData, frame)
            azFig, elFig = makeTimestreamFigs(plotData)          
            figs = [trajectory, velocity, error, acceleration, azFig, elFig]
            return figs


def makeTimestreamFigs(plotData):
    azActd = plotData['telAzAct']
    elActd = plotData['telElAct']
    azDesd = plotData['telAzDes']
    elDesd = plotData['telElAct']
    td = plotData['telTime']-plotData['telTime'].min()
    azErr = azActd-azDesd
    elErr = elActd-elDesd
    error = np.sqrt(azErr**2 + elErr**2)
    xaxis, yaxis = getXYAxisLayouts()
    azFig = px.scatter(x=td,
                       y=np.rad2deg(azActd),
                       color=rad2arcsec(error),
     )
    azFig.update_layout(
        coloraxis_colorbar=dict(
        title="Error<br>[arcsec]",
        ),
        plot_bgcolor="white",
        xaxis=xaxis,
        yaxis=yaxis,
        font={'size': 12,},
        title="Absolute Azimuth vs Time: Obsnum {}".format(plotData['obsnum']),
        xaxis_title="Observation Time [s]",
        yaxis_title="TelAzAct [deg.]",
        margin=go.layout.Margin(
            l=10,
            r=10,
            b=5,
            t=40,
        ),)
    elFig = px.scatter(x=td,
                       y=np.rad2deg(elActd),
                       color=rad2arcsec(error),
     )
    elFig.update_layout(
        coloraxis_colorbar=dict(
        title="Error<br>[arcsec]",
        ),
        plot_bgcolor="white",
        title="Absolute Elevation vs Time: Obsnum {}".format(plotData['obsnum']),
        xaxis=xaxis,
        yaxis=yaxis,
        font={'size': 12,},
        xaxis_title="Observation Time [s]",
        yaxis_title="TelElAct [deg.]",
        margin=go.layout.Margin(
            l=10,
            r=10,
            b=5,
            t=40,
        ),)

    return azFig, elFig


def makeTrajectoryPlot(plotData, frame):
    if(frame == 'source'):
        az = rad2arcmin(plotData['telAzMap'])
        el = rad2arcmin(plotData['telElMap'])
        xtitle = "TelAzMap [arcmin]"
        ytitle = "TelElMap [arcmin]"
    elif(frame == 'telescope'):
        az = np.rad2deg(plotData['telAzAct'])
        el = np.rad2deg(plotData['telElAct'])
        xtitle = "telAzAct [deg.]"
        ytitle = "telElAct [deg.]"
    xaxis, yaxis = getXYAxisLayouts()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=az,
            y=el,
        ))
    fig.update_layout(
        height=400,
        plot_bgcolor="white",
        xaxis=xaxis,
        yaxis=yaxis,
        font={'size': 12,},
        title="Trajectory ({0:} Frame): Obsnum {1:}".format(frame, plotData['obsnum']),
        xaxis_title=xtitle,
        yaxis_title=ytitle, 
        margin=go.layout.Margin(
            l=10,
            r=10,
            b=5,
            t=40,
        ),)
    return fig


def makeErrorPlot(plotData, frame):
    if(frame == "source"):
        az = rad2arcmin(plotData['telAzMap'])
        el = rad2arcmin(plotData['telElMap'])
        xtitle = "TelAzMap [arcmin]"
        ytitle = "TelElMap [arcmin]"
    elif(frame == "telescope"):
        az = np.rad2deg(plotData['telAzAct'])
        el = np.rad2deg(plotData['telElAct'])
        xtitle = "TelAzAct [deg.]"
        ytitle = "TelElAct [deg.]"
    azActd = plotData['telAzAct']
    elActd = plotData['telElAct']
    azDesd = plotData['telAzDes']
    elDesd = plotData['telElDes']    
    azErr = azActd-azDesd
    elErr = elActd-elDesd
    error = np.sqrt(azErr**2 + elErr**2)
    xaxis, yaxis = getXYAxisLayouts()
    fig = px.scatter(x=az,
                     y=el,
                     color=rad2arcsec(error),
     )
    fig.update_layout(
        height=400,
        coloraxis_colorbar=dict(
        title="Error<br>[arcsec]",
        ),
        plot_bgcolor="white",
        xaxis=xaxis,
        yaxis=yaxis,
        font={'size': 12,},
        title="Error Magnitude ({0:} Frame): Obsnum {1:}".format(frame, plotData['obsnum']),
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        margin=go.layout.Margin(
            l=10,
            r=10,
            b=5,
            t=40,
        ),)
    return fig


def makeVelocityPlot(plotData, frame):
    az, el, vel, acc = getVelAcc(plotData, frame)
    if(frame == "source"):
        xtitle = "TelAzMap [arcmin]"
        ytitle = "TelElMap [arcmin]"
        az = rad2arcmin(az)
        el = rad2arcmin(el)
    elif(frame == "telescope"):
        xtitle = "TelAzAct [deg.]"
        ytitle = "TelElAct [deg.]"
        az = np.rad2deg(az)
        el = np.rad2deg(el)
    vel = rad2arcsec(vel)
    acc = rad2arcsec(acc)
        
    xaxis, yaxis = getXYAxisLayouts()
    vfig = px.scatter(x=az,
                      y=el, 
                      color=vel,
     )
    vfig.update_layout(
        height=400,
        coloraxis_colorbar=dict(
        title="Peak Vel<br>[arcsec/s]",
        ),
        plot_bgcolor="white",
        xaxis=xaxis,
        yaxis=yaxis,
        font={'size': 12,},
        title="Velocity Magnitude ({0:} Frame): Obsnum {1:}".format(frame, plotData['obsnum']),
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        margin=go.layout.Margin(
            l=10,
            r=10,
            b=5,
            t=40,
        ),)
    
    afig = px.scatter(x=az, 
                      y=el, 
                      color=acc,
     )
    afig.update_layout(
        height=400,
        coloraxis_colorbar=dict(
        title="Peak Accel<br>[arcsec/s/s]",
        ),
        plot_bgcolor="white",
        xaxis=xaxis,
        yaxis=yaxis,
        font={'size': 12,},
        title="Acceleration Magnitude ({0:} Frame): Obsnum {1:}".format(frame, plotData['obsnum']),
        xaxis_title=xtitle,
        yaxis_title=ytitle, 
        margin=go.layout.Margin(
            l=10,
            r=10,
            b=5,
            t=40,
        ),)
    return vfig, afig


def getVelAcc(plotData, frame):
    if(frame == "source"):
        az = plotData['telAzMap']
        el = plotData['telElMap']
    elif(frame == "telescope"):
        az = plotData['telAzAct']
        el = plotData['telElAct']
    time = plotData['telTime']
    # calculate velocity and acceleration vector lengths
    dt = np.ediff1d(time)
    azVel = np.ediff1d(az)/dt
    elVel = np.ediff1d(el)/dt
    vel = np.sqrt(azVel**2 + elVel**2)
    azAcc = np.ediff1d(azVel)/dt[0:-1]
    elAcc = np.ediff1d(elVel)/dt[0:-1]
    acc = np.sqrt(azAcc**2 + elAcc**2)
    # make sure everyone is the same length on output
    minLen = min([len(az), len(el), len(vel), len(acc)])
    az = az[0:minLen]
    el = el[0:minLen]
    vel = vel[0:minLen]
    acc = acc[0:minLen]
    return az, el, vel, acc


def subsampleMean(x, n):
    end = n*int(len(x)/n)
    return np.mean(x[0:end].reshape(-1, n), 1)

def subsampleStd(x, n):
    end = n*int(len(x)/n)
    return np.std(x[0:end].reshape(-1, n), 1)

        
def fetchTelPlotData(telFile, downsample):
    nc = netCDF4.Dataset(telFile)
    keys = nc.variables.keys()
    dataKeys = [k for k in keys if 'Data' in k]
    plotData = dict(
        telAzMap = subsampleMean(nc.variables['Data.TelescopeBackend.TelAzMap'][:].data, downsample),
        telElMap = subsampleMean(nc.variables['Data.TelescopeBackend.TelElMap'][:].data, downsample),
        telTime = subsampleMean(nc.variables['Data.TelescopeBackend.TelTime'][:].data, downsample),
        telAzDes = subsampleMean(nc.variables['Data.TelescopeBackend.TelAzDes'][:].data, downsample),
        telElDes = subsampleMean(nc.variables['Data.TelescopeBackend.TelElDes'][:].data, downsample),
        telAzAct = subsampleMean(nc.variables['Data.TelescopeBackend.TelAzAct'][:].data, downsample),
        telElAct = subsampleMean(nc.variables['Data.TelescopeBackend.TelElAct'][:].data, downsample),
        obsnum = int(nc.variables['Header.Dcs.ObsNum'][:].data)
    )
    nc.close()
    return plotData


        
def makeHeaderEntry(box, name, keys, bgColor='blue', valueColor='black'):        
    h = dict()
    h['title'] = box.child(dbc.Row).child(
        html.H4, name, style={'backgroundColor':bgColor, 'color':'white', 'text-align':'center'})
    for k in keys:
        hrow = box.child(dbc.Row)
        h[k] = dict()
        h[k]['name'] = hrow.child(dbc.Col, width=4).child(html.H5, "{}: ".format(k),
                                                          style={'color':'purple'})
        h[k]['value'] = hrow.child(dbc.Col, width=8).child(html.H5, "-",
                                                           style={'color':valueColor})
    return h
    

def fetchTelHeaderData(telFile):
    print("Reading data from {}".format(telFile))
    header = dict()
    for cat in ['Dcs', 'Source', 'Sky', 'Telescope', 'M1', 'M2', 'M3', 'DCS', 'Radiometer', 'Map', 'Lissajous']:
        header[cat] = buildHeaderDict('Header.{}.'.format(cat), telFile)
        # Debug
        if(0):
            print()
            print(cat)
            print("----------")
            print(header[cat].keys())
    return header


def buildHeaderDict(cat, telFile):
    nc = netCDF4.Dataset(telFile)
    keys = nc.variables.keys()
    a = dict()
    catKeys = [k for k in keys if cat in k]
    # We have to get rid of all the byte array strings (or deal with them separately)
    rem = []
    for k in catKeys:
        dt = nc.variables[k].datatype
        if(dt == 'S1'):
            rem.append(k)
    for k in rem:
        catKeys.remove(k)
    for k in catKeys:
        a[k.replace(cat, "")] = float(nc.variables[k][0].data)
    for k in rem:
        a[k.replace(cat, "")] = b''.join(nc.variables[k][:]).decode().strip()
    nc.close()
    return a


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
        "template": ToltecTelViewer,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", True),
    }
}
