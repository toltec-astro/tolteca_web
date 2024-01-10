"""
To Do:
 - 
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
from dash import callback_context
from astropy.table import Table
from astropy import units as u
import plotly.graph_objs as go
import plotly.express as px
from dash import dash_table
from pathlib import Path
import cachetools.func
import dash_daq as daq
from dash import html
from glob import glob
from dash import dcc
from dash import ctx
import xarray as xr
import pandas as pd
import numpy as np
import functools
import datetime
import netCDF4
import time
import json
import yaml
import dash
import os



class ToltecHousekeepingViewer(ComponentTemplate):
    class Meta:
        component_cls = dbc.Container

    def __init__(
        self,
        title_text="Toltec Housekeeping Viewer",
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
        # Note use of Path class (more modern than os.path.join)
        rPath = Path("/Users/wilson/GitHub/tolteca_web/src/tolteca_web")
        dPath = rPath/"toltecHousekeepingViewer/test_data"
        thermPath = dPath/"thermetry"
        dilPath = dPath/"dilutionFridge"
        cryocmpPath = dPath/"cryocmp"
        thermetryFile = list(thermPath.glob('therm*.nc'))
        thermetryFile.sort()
        thermetryFile = thermetryFile[-1]
        dilFile = list(dilPath.glob('dil*.nc'))
        dilFile.sort()
        dilFile = dilFile[-1]
        cryocmpFile = list(cryocmpPath.glob('cryocmp*.nc'))
        cryocmpFile.sort()
        cryocmpFile = cryocmpFile[-1]

        nc = {
            'thermetry': _get_fileobject(thermetryFile),
            'thermetryFile': thermetryFile,
            'dilution': _get_fileobject(dilFile),
            'dilutionFile': dilFile,
            'cryocmp': _get_fileobject(cryocmpFile),
            'cryocmpFile': cryocmpFile}

        # Read in the configuration yaml file
        cPath = rPath/"toltecHousekeepingViewer"
        with open(cPath/'housekeepingConfig.yaml', 'r') as configFile:
            c = yaml.safe_load(configFile)

        # Here is the layout
        box1 = body.child(dbc.Row)
        tbox = box1.child(dbc.Col, width=6)
        tboxTitle = tbox.child(dbc.Row).child(html.H4, 'Temperatures (update in 0s) - Click Value to Plot',
                                              style={'backgroundColor':'#F9BA97',
                                                     'color':'#4D2C29',
                                                     'text-align':'center'})
        tvalsbox = tbox.child(dbc.Row)
        t45box = tvalsbox.child(dbc.Col, width=3)
        t45 = makeTemperaturesColumn(t45box, '45K', c, nc)
        t4box = tvalsbox.child(dbc.Col, width=3)
        t4 = makeTemperaturesColumn(t4box, '4K', c, nc)
        t1box = tvalsbox.child(dbc.Col, width=3)
        t1 = makeTemperaturesColumn(t1box, '1K', c, nc)
        t0p1box = tvalsbox.child(dbc.Col, width=3)
        t0p1 = makeTemperaturesColumn(t0p1box, '100mK', c, nc)
        tvalsbox.child(dbc.Row).child(html.Hr)
        temperatures = {'title': tboxTitle,
                        't45': t45,
                        't4': t4,
                        't1': t1,
                        't0p1': t0p1}
        tpbox = box1.child(dbc.Col, width=6)
        tControlBox = tpbox.child(dbc.Row)
        tControlBox.child(dbc.Col, width=7)
        tDurationTitle = tControlBox.child(dbc.Col, width=3).child(
            html.H5, "Plot Duration [hrs]:", style={'textAlign': 'right'})
        tDuration = tControlBox.child(
            dbc.Col, width=1).child(
                dcc.Input, value=2.,
                min=0.1, max=50.,
                debounce=True, type='number',
                style={'width': '75%',
                       'height': '75%',
                       'margin-left': '-10px'})
        tPlot = tpbox.child(dbc.Row).child(dcc.Graph)
        tStore = tpbox.child(dcc.Store)
        tpbox.child(dbc.Row).child(html.Hr)
        temperaturesPlot = {'plot': tPlot,
                            'signal choice': tStore,
                            'plot duration': tDuration}
        
        
        # Compressors
        box2 = body.child(dbc.Row)
        cbox = box2.child(dbc.Col, width=6)
        cboxTitle = cbox.child(dbc.Row).child(html.H4, 'Compressors (update in 0s)',
                                              style={'backgroundColor':'#D4603C',
                                                     'color':'white',
                                                     'text-align':'center'})
        clabelRow = cbox.child(dbc.Row)
        cryocmpLabel = clabelRow.child(dbc.Col, width=6).child(html.H5, 'AuxPTC Compressor',
                                                               style={'color':'#4D2C29',
                                                                      'text-align':'center',
                                                                      'font-weight': 'bold'})  
        dilcmpLabel = clabelRow.child(dbc.Col, width=6).child(html.H5, 'Dilution Fridge Compressor',
                                                              style={'color':'#4D2C29',
                                                                     'text-align':'center',
                                                                     'font-weight': 'bold'})      
        cvalsbox = cbox.child(dbc.Row)
        cryocmpStatsBox = cvalsbox.child(dbc.Col, width=3)
        cryocmpTempsBox = cvalsbox.child(dbc.Col, width=3)
        dilcmpStatsBox = cvalsbox.child(dbc.Col, width=3)
        dilcmpTempsBox = cvalsbox.child(dbc.Col, width=3)
        cryocmpStats = makeCompressorsColumn(cryocmpStatsBox, 'cryocmp stats', c, nc['cryocmp'])
        cryocmpTemps = makeCompressorsColumn(cryocmpTempsBox, 'cryocmp temps', c, nc['cryocmp'])
        dilcmpStats = makeCompressorsColumn(dilcmpStatsBox, 'dilcmp stats', c, nc['dilution'])
        dilcmpTemps = makeCompressorsColumn(dilcmpTempsBox, 'dilcmp temps', c, nc['dilution'])
        cvalsbox.child(dbc.Row).child(html.Hr)
        compressors = {'title': cboxTitle,
                       'cryocmpStats': cryocmpStats,
                       'cryocmpTemps': cryocmpTemps,
                       'dilcmpStats': dilcmpStats,
                       'dilcmpTemps': dilcmpTemps}


        cpbox = box2.child(dbc.Col, width=6)
        cControlBox = cpbox.child(dbc.Row)
        cControlBox.child(dbc.Col, width=7)
        cDurationTitle = cControlBox.child(dbc.Col, width=3).child(
            html.H5, "Plot Duration [hrs]:", className="mb-2", style={'textAlign': 'right'},
        )
        cDuration = cControlBox.child(
            dbc.Col, width=1).child(dcc.Input, value=2.,
                                    min=0.1, max=50.,
                                    debounce=True, type='number',
                                    style={'width': '75%',
                                           'margin-left': '-10px'})
        cPlot = cpbox.child(dbc.Row).child(dcc.Graph)
        cpbox.child(dbc.Row).child(html.Hr)
        compressorsPlot = {'plot': cPlot,
                           'plot duration': cDuration}


        # Oxford Dilution Fridge
        box3 = body.child(dbc.Row)
        dbox = box3.child(dbc.Col, width=6)
        dboxTitle = dbox.child(dbc.Row).child(html.H4, 'Oxford Dilution Fridge (update in 0s)',
                                              style={'backgroundColor':'#565D85',
                                                     'color':'white',
                                                     'text-align':'center'})
        dlabelRow = dbox.child(dbc.Row)
        dvalsbox = dbox.child(dbc.Row)
        dvalsbox.child(dbc.Col, width=1)
        dfStatsBox = dvalsbox.child(dbc.Col, width=4)
        dfTurboBox = dvalsbox.child(dbc.Col, width=4)
        dfValvesBox = dvalsbox.child(dbc.Col, width=2)
        dvalsbox.child(dbc.Col, width=1)
        dfStats = makeDFColumn(dfStatsBox, 'df stats', c, nc['dilution'])
        dfTurbo = makeDFColumn(dfTurboBox, 'df Turbo Pump', c, nc['dilution'])
        dfValves = makeDFColumn(dfValvesBox, 'df valves', c, nc['dilution'])
        dvalsbox.child(dbc.Row).child(html.Hr)
        dilutionFridge = {'title': dboxTitle,
                           'dfStats': dfStats,
                           'dfTurbo': dfTurbo,
                           'dfValves': dfValves,}
        dfImageBox = box3.child(dbc.Col, width=3)
        dfImage = dfImageBox.child(dbc.Row).child(
            html.Img,
            src = 'http://lmtserver.astro.umass.edu/rss/toltec/tdf.png',
            style={'height':'600px', 'width':'auto'})
        toltecLogoBox = box3.child(dbc.Col, width=3)
        toltecLogo = toltecLogoBox.child(dbc.Row).child(
            html.Img,
            src = 'http://toltec.astro.umass.edu/images/toltec_logo.png',
            style={'height':'400px', 'width':'auto'})
        

        # Timers
        timerRow = body.child(dbc.Row)
        textUpdateInterval = 5
        textUpdateTimer  = timerRow.child(dcc.Interval,
                                          interval=textUpdateInterval*1000,
                                          n_intervals=0)
        temperatureUpdateInterval = 60
        temperatureTimer = timerRow.child(dcc.Interval,
                                          interval=temperatureUpdateInterval*1000,
                                          n_intervals=0)
        compressorUpdateInterval = 115
        compressorTimer = timerRow.child(dcc.Interval,
                                          interval=compressorUpdateInterval*1000,
                                          n_intervals=0)

        dilutionFridgeUpdateInterval = 130
        dilutionFridgeTimer = timerRow.child(dcc.Interval,
                                          interval=dilutionFridgeUpdateInterval*1000,
                                          n_intervals=0)
        
        timers = {'textUpdateInterval': textUpdateInterval,
                  'textUpdateTimer': textUpdateTimer,
                  'temperatureUpdateInterval': temperatureUpdateInterval,
                  'temperatureTimer': temperatureTimer,
                  'compressorUpdateInterval': compressorUpdateInterval,
                  'compressorTimer': compressorTimer,
                  'dilutionFridgeUpdateInterval': dilutionFridgeUpdateInterval,
                  'dilutionFridgeTimer': dilutionFridgeTimer,}

        super().setup_layout(app)

        self._registerCallbacks(
            app,
            nc,
            timers,
            temperatures,
            temperaturesPlot,
            compressors,
            compressorsPlot,
            dilutionFridge,
            dfImage,
        )
        return

    
    def _registerCallbacks(
            self,
            app,
            nc,
            timers,
            temperatures,
            temperaturesPlot,
            compressors,
            compressorsPlot,
            dilutionFridge,
            dfImage,
    ):

        
        # ---------------------------
        # update signal choice and temperatures plot
        # ---------------------------
        inputList = []
        for k in [key for key in temperatures.keys() if key != 'title']:
            for c in temperatures[k].keys():
                if(c != 'title'):
                    inputList.append(Input(temperatures[k][c]['value'].id, "n_clicks"))
        # Create a set of valid IDs from inputList
        valid_ids = {input.component_id for input in inputList}


        @app.callback(
            [Output(temperaturesPlot['signal choice'].id, "data")],
            inputList,
        )
        def updateTemperatureSignalChoice(*args):
            ctx = callback_context
            clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]
            # If the clicked ID is not in our list of valid IDs, prevent update
            if clicked_id not in valid_ids:
                raise PreventUpdate
            return [clicked_id]

        
        @app.callback(
            [Output(temperaturesPlot['plot'].id, "figure")],
            [
                Input(temperaturesPlot['plot duration'].id, "value"),
                Input(temperaturesPlot['signal choice'].id, "data"),
            ],
        )
        def updateTemperaturePlot(plotDuration, signal_id):
            time.sleep(0.5)
            default_id = temperatures['t0p1']['1.1mm array']['id']
            if(signal_id is None):
                signal_id = default_id
            r = find_dictionaries_with_value(temperatures, 'id', signal_id)
            config = r[0][1]['config']
            if not config['plotable']:
                raise PreventUpdate
            fig = getTemperatureFig(plotDuration, r)
            return [fig]


        def getTemperatureFig(plotDuration, r):
            plotDuration *= 3600.
            d = r[0][1]
            config = d['config']
            # Determine which time signal to use
            name = config['ncVarName']
            if 'Data.ToltecThermetry' in name:
                n = name.split('Temperature')[1]
                timeVar = 'Data.ToltecThermetry.Time'+n
                sampleRate = 1./30.
            else:
                timeVar = 'Data.ToltecDilutionFridge.SampleTime'
                sampleRate = 1./60.
            ncdf = d['nc']
            tv = ncdf.variables[timeVar]
            nSamples = min([int(plotDuration*sampleRate), tv.shape[0]])
            time = tv[:].data[-nSamples:-1]
            time = [datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in time]

            s = ncdf.variables[name][:].data[-nSamples:-1]
            xtitle = 'Universal Time'
            ytitle = 'Temperature [K]'
            xaxis, yaxis = getXYAxisLayouts()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=s,
                ))
            fig.update_layout(
                height=325,
                plot_bgcolor="white",
                xaxis=xaxis,
                yaxis=yaxis,
                font={'size': 12,},
                title=config['label'],
                xaxis_title=xtitle,
                yaxis_title=ytitle, 
                margin=go.layout.Margin(
                    l=10,
                    r=10,
                    b=5,
                    t=40,
                ),)
            return fig

        
        # ---------------------------
        # update temperatures
        # ---------------------------
        outputList = []
        for k in [key for key in temperatures.keys() if key != 'title']:
            for c in temperatures[k].keys():
                if(c != 'title'):
                    outputList.append(Output(temperatures[k][c]['value'].id, "children"))
        for k in [key for key in temperatures.keys() if key != 'title']:
            for c in temperatures[k].keys():
                if(c != 'title'):
                    outputList.append(Output(temperatures[k][c]['value'].id, "style"))
        @app.callback(
            outputList,
            [
                Input(timers['temperatureTimer'].id, "n_intervals"),
            ],
        )
        def updateTemperatures(n):
            tData = updateTemperatureData()
            if(tData is None):
                raise PreventUpdate
            return tData


        def updateTemperatureData():
            nc['thermetry'].sync()
            nc['dilution'].sync()
            values = []
            styles = []
            for k in [key for key in temperatures.keys() if key != 'title']:
                for c in temperatures[k].keys():
                    if(c != 'title'):
                        d = temperatures[k][c]
                        s = d['nc'].variables[d['config']['ncVarName']][-1].data
                        if(s<0):
                            values.append("No data")
                            styles.append({'color': 'blue'})
                        else:
                            values.append("{0:3.2f}".format(s))
                            if((s < d['config']['lowerLimit']) |
                               (s > d['config']['upperLimit'])):
                                styles.append({'color': 'red'})
                            else:
                                styles.append({'color': 'black'})
            return values+styles


        
        # ---------------------------
        # update compressors plot
        # ---------------------------       
        @app.callback(
            [
                Output(compressorsPlot['plot'].id, "figure")
            ],
            [
                Input(compressorsPlot['plot duration'].id, "value"),
                Input(timers['compressorTimer'].id, "n_intervals"),
            ],
        )
        def updateCompressorsPlot(plotDuration, n):
            time.sleep(0.5)
            fig = getCompressorFig(plotDuration)
            return [fig]


        def getCompressorFig(plotDuration):
            plotDuration *= 3600.
            tVarNames = ["Data.ToltecCryocmp.Time",
                         "Data.ToltecCryocmp.Time",
                         "Data.ToltecDilutionFridge.SampleTime",
                         "Data.ToltecDilutionFridge.SampleTime"]
            sVarNames = ["Data.ToltecCryocmp.OilTemp",
                         "Data.ToltecCryocmp.CoolOutTemp",
                         "Data.ToltecDilutionFridge.StsDevC1PtcSigOilt",
                         "Data.ToltecDilutionFridge.StsDevC1PtcSigWot"]
            labels = ['AuxPTC Oil', 'AuxPTC H2O Out',
                      'DF Oil', 'DF H2O Out']
            ncdf = [nc['cryocmp'], nc['cryocmp'],
                    nc['dilution'], nc['dilution']]
            sampleRate = 1./60.
            nSamples = int(plotDuration*sampleRate)
            ymax = -1.
            ymin = 100.
            
            xtitle = 'Universal Time'
            ytitle = 'Temperature [C]'
            xaxis, yaxis = getXYAxisLayouts()
            fig = go.Figure()
            for t, s, l, n in zip(tVarNames, sVarNames, labels, ncdf):
                time = n.variables[t][-nSamples:].data
                time = [datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in time]
                v = n.variables[s][-nSamples:].data
                ymax = max([ymax, v.max()])
                ymin = min([ymin, v.min()])
                fig.add_trace(
                    go.Scatter(
                        x=time, 
                        y=v,
                        name=l,
                        mode='lines',
                    ))
            fig.add_trace(
                go.Scatter(
                    x = np.array([time[0], time[-1]]),
                    y = np.array([52, 52]),
                    mode = 'lines',
                    line=dict(color='red', dash='dash',),
                    name = 'Upper Limit',
                    ))
                    
            fig.update_layout(
                height=200,
                plot_bgcolor="white",
                xaxis=xaxis,
                yaxis=yaxis,
                font={'size': 12,},
                xaxis_title=xtitle,
                yaxis_title=ytitle, 
                margin=go.layout.Margin(
                    l=10,
                    r=10,
                    b=5,
                    t=0,
                ),)
            ymax = max(ymax, 53.)
            fig.layout.yaxis.range = [ymin, ymax]
            return fig


        
        # ---------------------------
        # update compressors
        # ---------------------------
        outputList = []
        for k in [key for key in compressors.keys() if key != 'title']:
            for c in compressors[k].keys():
                if(c != 'title'):
                    outputList.append(Output(compressors[k][c]['value'].id, "children"))
        for k in [key for key in compressors.keys() if key != 'title']:
            for c in compressors[k].keys():
                if(c != 'title'):
                    outputList.append(Output(compressors[k][c]['value'].id, "style"))
        @app.callback(
            outputList,
            [
                Input(timers['compressorTimer'].id, "n_intervals"),
            ],
        )
        def updateCompressors(n):
            tData = updateCompressorData()
            if(tData is None):
                raise PreventUpdate
            return tData


        def updateCompressorData():
            time.sleep(0.25)
            nc['cryocmp'].sync()
            nc['dilution'].sync()
            values = []
            styles = []
            for k in [key for key in compressors.keys() if key != 'title']:
                for c in compressors[k].keys():
                    if(c != 'title'):
                        d = compressors[k][c]
                        s = d['nc'].variables[d['config']['ncVarName']][-1].data
                        
                        # Change Enabled=1/0 to ON/OFF
                        if d['config']['label'] == "Enabled":
                            if(s):
                                values.append('ON')
                                styles.append({'color': 'black'})
                            else:
                                values.append('OFF')
                                styles.append({'color': 'red'})
                        elif isinstance(s, bytes) or (isinstance(s, np.ndarray) and s.dtype.type is np.bytes_):
                            # Handle byte strings
                            # Decode and process as required
                            if s.size == 1:  # Single byte string
                                s = s.item().decode('utf-8')
                            else:  # Array of byte strings
                                non_empty_strings = [byte_str.decode('utf-8') for byte_str in s if byte_str]
                                s = ''.join(non_empty_strings)
                                values.append(s)
                                styles.append({'color': 'black'})
                        else:
                            # check for scaling
                            scale = d['config'].get('scale', 1.)
                            values.append("{0:3.1f}".format(s*scale))
                            if((d['config']['lowerLimit'] is not None) &
                               (d['config']['upperLimit'] is not None)):
                                if((s < d['config']['lowerLimit']) |
                                   (s > d['config']['upperLimit'])):
                                    styles.append({'color': 'red'})
                                else:
                                    styles.append({'color': 'black'})
                            else:
                                styles.append({'color': 'black'})
            return values+styles


        # ---------------------------
        # update dilution fridge
        # ---------------------------
        outputList = []
        for k in [key for key in dilutionFridge.keys() if key != 'title']:
            for c in dilutionFridge[k].keys():
                if(c != 'title'):
                    outputList.append(Output(dilutionFridge[k][c]['value'].id, "children"))
        for k in [key for key in dilutionFridge.keys() if key != 'title']:
            for c in dilutionFridge[k].keys():
                if(c != 'title'):
                    outputList.append(Output(dilutionFridge[k][c]['value'].id, "style"))
        @app.callback(
            outputList, 
            [
                Input(timers['dilutionFridgeTimer'].id, "n_intervals"),
            ],
        )
        def updateDilutionFridge(n):
            tData = updateDilutionFridgeData()
            if(tData is None):
                raise PreventUpdate
            return tData

        
        @app.callback(
            [
                Output(dfImage.id, "src"),
            ],
            [
                Input(timers['dilutionFridgeTimer'].id, "n_intervals"),
            ],
        )
        def updateDilutionFridgeImage(n):
            src = "http://lmtserver.astro.umass.edu/rss/toltec/tdf.png"
            return [src]
        


        def updateDilutionFridgeData():
            time.sleep(1.)
            nc['dilution'].sync()
            values = []
            styles = []
            for k in [key for key in dilutionFridge.keys() if key != 'title']:
                for c in dilutionFridge[k].keys():
                    if(c != 'title'):
                        d = dilutionFridge[k][c]
                        s = d['nc'].variables[d['config']['ncVarName']][-1].data
                        
                        if isinstance(s, bytes) or (isinstance(s, np.ndarray) and s.dtype.type is np.bytes_):
                            non_empty_strings = [byte_str.decode('utf-8') for byte_str in s if byte_str]
                            s = ''.join(non_empty_strings)
                            values.append(s)
                            if d['config']['label'] in ["V1 State", "V5 State"]:
                                if (s=='CLOSED'):
                                    styles.append({'color': 'red'})
                                else:
                                    styles.append({'color': 'black'})
                            else:
                                if (s=='OPEN'):
                                    styles.append({'color': 'red'})
                                else:
                                    styles.append({'color': 'black'})
                        else:
                            # check for scaling
                            scale = d['config'].get('scale', 1.)
                            values.append("{0:3.1f}".format(s*scale))
                            if((d['config']['lowerLimit'] is not None) &
                               (d['config']['upperLimit'] is not None)):
                                if((s < d['config']['lowerLimit']) |
                                   (s > d['config']['upperLimit'])):
                                    styles.append({'color': 'red'})
                                else:
                                    styles.append({'color': 'black'})
                            else:
                                styles.append({'color': 'black'})
            return values+styles

        
        
        # ---------------------------
        # countdown for temperature update
        # ---------------------------
        @app.callback(
            [
                Output(temperatures['title'].id, "children"),
            ],            
            [
                Input(timers['temperatureTimer'].id, "n_intervals"),
                Input(timers['textUpdateTimer'].id, "n_intervals"),
            ],
        )
        def updateTemperaturesText(data_n, countdown_n):
            if dash.callback_context.triggered_id == 'data-update-interval':
                txt = f"Temperatures (update in: {timers['temperatureUpdateInterval']}s - Click Value to Plot)"
            else:
                n = countdown_n % (timers['temperatureUpdateInterval']/timers['textUpdateInterval'])
                dt = timers['temperatureUpdateInterval'] - \
                    timers['textUpdateInterval']*(n % timers['temperatureUpdateInterval'])
                txt = f"Temperatures (update in: {dt}s - Click Value to Plot)"
            return [txt]

        
        # ---------------------------
        # countdown for compressors update
        # ---------------------------
        @app.callback(
            [
                Output(compressors['title'].id, "children"),
            ],            
            [
                Input(timers['temperatureTimer'].id, "n_intervals"),
                Input(timers['textUpdateTimer'].id, "n_intervals"),
            ],
        )
        def updateCompressorsText(data_n, countdown_n):
            if dash.callback_context.triggered_id == 'data-update-interval':
                txt = f"Compressors (update in: {timers['compressorUpdateInterval']}s)"
            else:
                n = countdown_n % (timers['compressorUpdateInterval']/timers['textUpdateInterval'])
                dt = timers['compressorUpdateInterval'] - \
                    timers['textUpdateInterval']*(n % timers['compressorUpdateInterval'])
                txt = f"Compressors (update in: {dt}s)"
            return [txt]


        # ---------------------------
        # countdown for dilution fridge update
        # ---------------------------
        @app.callback(
            [
                Output(dilutionFridge['title'].id, "children"),
            ],            
            [
                Input(timers['temperatureTimer'].id, "n_intervals"),
                Input(timers['textUpdateTimer'].id, "n_intervals"),
            ],
        )
        def updateDilutionFridgeText(data_n, countdown_n):
            if dash.callback_context.triggered_id == 'data-update-interval':
                txt = f"DilutionFridge (update in: {timers['dilutionFridgeUpdateInterval']}s)"
            else:
                n = countdown_n % (timers['dilutionFridgeUpdateInterval']/timers['textUpdateInterval'])
                dt = timers['dilutionFridgeUpdateInterval'] - \
                    timers['textUpdateInterval']*(n % timers['dilutionFridgeUpdateInterval'])
                txt = f"DilutionFridge (update in: {dt}s)"
            return [txt]


def makeEmptyFigs(nfigs):
    figs = []
    xaxis, yaxis = getXYAxisLayouts()
    for i in range(nfigs):
        fig = go.Figure()
        fig.update_layout(
            xaxis=xaxis,
            yaxis=yaxis,
            plot_bgcolor='white',
            height=357,
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


def makeTemperaturesColumn(box, columnName, config, nc, valueColor="black"):
    h = dict()
    label = columnName
    h['title'] = box.child(dbc.Row).child(html.H5, label,
                                          style={'color':'#4D2C29',
                                                 'text-align':'center',
                                                 'font-weight': 'bold'})
    
    entries = find_dictionaries_with_value(config, 'column', columnName)
    for e in entries:
        path = e[0]
        v = e[1]
        k = v['label']
        fileName = config[path[0]][path[1]]['ncFileName']
        hrow = box.child(dbc.Row)
        h[k] = dict()
        h[k]['name'] = hrow.child(dbc.Col, width=8).child(html.H5, "{}: ".format(k),
                                                          style={'color':'#8D534E'})
        h[k]['value'] = hrow.child(dbc.Col, width=4).child(html.H5, "-",
                                                           style={'color':valueColor})
        h[k]['config'] = v
        h[k]['config path'] = path
        h[k]['fileName'] = fileName
        h[k]['id'] = h[k]['value'].id
        if 'therm' in fileName:
            h[k]['nc'] = nc['thermetry']
        else:
            h[k]['nc'] = nc['dilution']
    return h


def makeCompressorsColumn(box, columnName, config, nc, valueColor="black"):
    h = dict()
    label = columnName.replace('cryocmp ', '')
    label = label.replace('dilcmp ', '')
    if label == 'temps':
        label += ' [C]'
    label = label.capitalize()
    label = label.replace("[c]", "[C]")
    h['title'] = box.child(dbc.Row).child(html.H5, label,
                                          style={'color':'#4D2C29',
                                                 'text-align':'center',
                                                 'font-weight': 'bold'})
    
    entries = find_dictionaries_with_value(config, 'column', columnName)
    for e in entries:
        path = e[0]
        v = e[1]
        k = v['label']
        k = k.replace("Pressure", "Pres [psi]")
        hrow = box.child(dbc.Row)
        h[k] = dict()
        h[k]['name'] = hrow.child(dbc.Col, width=8).child(html.H5, "{}: ".format(k),
                                                          style={'color':'#8D534E'})
        h[k]['value'] = hrow.child(dbc.Col, width=4).child(html.H5, "-",
                                                           style={'color':valueColor})
        h[k]['config'] = v
        h[k]['config path'] = path
        h[k]['nc'] = nc
    return h


def makeDFColumn(box, columnName, config, nc, valueColor="black"):
    h = dict()
    label = columnName.replace('df ', '')
    label = label.capitalize()
    h['title'] = box.child(dbc.Row).child(html.H5, label,
                                          style={'color':'#4D2C29',
                                                 'text-align':'center',
                                                 'font-weight': 'bold'})
    
    entries = find_dictionaries_with_value(config, 'column', columnName)
    for e in entries:
        path = e[0]
        v = e[1]
        k = v['label'].replace(" State", '')
        hrow = box.child(dbc.Row)
        h[k] = dict()
        h[k]['name'] = hrow.child(dbc.Col, width=8).child(html.H5, "{}: ".format(k),
                                                          style={'color':'#8D534E'})
        h[k]['value'] = hrow.child(dbc.Col, width=4).child(html.H5, "-",
                                                           style={'color':valueColor})
        h[k]['config'] = v
        h[k]['config path'] = path
        h[k]['nc'] = nc
    return h



def find_dictionaries_with_value(d, target_key, target_value, path=None, results=None):
    """
    Recursively searches the dictionary 'd' for entries where 'target_key' has 'target_value'.
    Returns a list of tuples, each containing the path and the subdictionary.
    """
    if path is None:
        path = []
    if results is None:
        results = []

    if isinstance(d, dict):
        for key, value in d.items():
            new_path = path + [key]
            if key == target_key and value == target_value:
                results.append((new_path[:-1], d))
            elif isinstance(value, dict):
                find_dictionaries_with_value(value, target_key, target_value, new_path, results)
    
    return results


def _get_fileobject(filepath):
    return netCDF4.Dataset(filepath)


DASHA_SITE = {
    "dasha": {
        "template": ToltecHousekeepingViewer,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", True),
    }
}
