"""
To Do:
 - 
"""
from dash_component_template import ComponentTemplate
from dash.dependencies import Input, Output, State
from .utilities import extract_and_parse_jpl_file, readHorizonsData
from datetime import datetime, date, timedelta, timezone
from astropy.coordinates import EarthLocation, AltAz
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from astropy.time import Time, TimeDelta
import dash_bootstrap_components as dbc
from .almaData import AlmaGridSources
from tollan.utils.log import logger
from ..common import LabeledInput
from dash import callback_context
from astropy.table import Table
from astropy import units as u
from astroplan import Observer
import plotly.graph_objs as go
from astropy.time import Time
import plotly.express as px
from dash import dash_table
from pathlib import Path
import cachetools.func
import dash_daq as daq
from PIL import Image
from dash import html
from dash import dcc
from dash import ctx
import numpy as np
import functools
import time
import dash
import os


class ToltecAlmaCalAvailability(ComponentTemplate):
    class Meta:
        component_cls = dbc.Container

    def __init__(
        self,
        title_text="Toltec Calibrator Availability Viewer",
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


        # I need a cute sun image.
        rPath = Path("/Users/wilson/GitHub/tolteca_web/src/tolteca_web")
        dPath = rPath/"toltecCalibratorAvailability"
        sunPath = str(dPath/"sun.png")
        sunny = Image.open(sunPath)

            
        # Hard code the data path for testing.
        # Note use of Path class (more modern than os.path.join)
        rPath = Path("/Users/wilson/GitHub/tolteca_web/src/tolteca_web")
        dPath = rPath/"toltecCalibratorAvailability/horizons_data"
        sourceFiles = list(dPath.glob('*_horizons.txt'))
        sourceFiles.sort()
        data = {}
        for s in sourceFiles:
            h, d = readHorizonsData(s)
            data[h['target_body_name']] = {'header': h, 'data': d}


        # Create Start time input and duration
        # Note that default start time should be current UTC time
        c_body = body.child(dbc.Row).child(dbc.Col, width=2).child(dbc.Row)
        defaultTime = datetime.now(timezone.utc)
        minTime = datetime(2024, 2, 1, tzinfo=timezone.utc)
        defaultTime = max(defaultTime, minTime)
        
        # The date selection control
        obsDateRow = c_body.child(html.Div, className='d-flex justify-content-end', justify='end')
        obsDateRow.child(html.Label,
                         "Obs Date: ",
                         style = {
                             'display': 'flex',
                             'align-items': 'center', 
                             'height': '100%'
                         })
        obsDate = obsDateRow.child(dcc.DatePickerSingle,
                                   min_date_allowed=date(2024, 2, 1),
                                   max_date_allowed=date(2028, 8, 31),
                                   initial_visible_month=defaultTime,
                                   date=defaultTime,
                                   style=dict(width='45%',))

        # And the uptimes plot
        plot = body.child(dbc.Row).child(dcc.Graph)

        # This is the lower figure with the ALMA grid sources plotted
        almaPlot = body.child(dbc.Row).child(dcc.Graph)
        srow = body.child(dbc.Row)
        srow.child(dbc.Col, width=1).child(html.H6, "Pointing Flux Lower Limit [Jy]")
        almaFluxLim = srow.child(dbc.Col, width=1).child(
            dcc.Slider, min=1, max=3, step=0.5, value=1.5,
            style={'width': '75%'})
        ags = AlmaGridSources()
        almaSources = ags.sources
        pointing = {
            'flux limit': almaFluxLim,
            'plot': almaPlot,
            'sources': almaSources,
            }
                
        super().setup_layout(app)

        self._registerCallbacks(
            app,
            obsDate,
            plot,
            data,
            pointing,
            sunny,
        )
        return

    
    def _registerCallbacks(
            self,
            app,
            obsDate,
            plot,
            data,
            pointing,
            sunny,
    ):

        
        # ---------------------------
        # Update the ALMA sources plot
        # ---------------------------
        @app.callback(
            [
                Output(pointing['plot'].id, 'figure')
            ],            
            [
                Input(obsDate.id, 'date'),
                Input(pointing['flux limit'].id, 'value')
            ],
        )
        def updateAlmaSourcesPlot(date, flux_limit):
            date = date.split('T')[0]
            date = date+'T06:30:00'
            dt = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
            dt = dt.replace(tzinfo=timezone.utc)
            w = np.where(pointing['sources']['FluxDensity'] >= flux_limit)[0]
            sources = pointing['sources'][w]
            fig = getAlmaPlot(dt, sources)
            return [fig]


        def getAlmaPlot(dt, almaSources):
            # deal with times first
            td = timedelta(hours=14)
            startTime = dt-td
            sun_rise, sun_set = getSunTimes(startTime)
            increment = timedelta(minutes=10)
            sr = Time(sun_rise)
            ss = Time(sun_set)
            increment = TimeDelta(600, format='sec')
            times = Time(ss + increment * range(int((sr - ss) / increment)))
            dtimes = [time.to_datetime() for time in times]
            dtimes = [d.replace(tzinfo=timezone.utc) for d in dtimes]

            # Need an lmt to calculate source elevations
            lmt = getLMT()

            # Generate the plot data in a big dictionary.
            sources = dict()
            altaz = AltAz(obstime=times, location=lmt.location)
            for s in almaSources:
                coords = s['coords']
                aa = coords.transform_to(altaz)
                if any(alt.alt.deg > 40 for alt in aa):
                    name = s['name']
                    cname = s['name']
                    sources[name] = {
                        'common name': cname,
                        '1mm flux value': s['FluxDensity'],
                        'elevation': aa.alt.deg,
                    }
            print("Found {} sources!".format(len(sources.keys())))
            
                         
            xtitle = 'Universal Time'
            ytitle = 'Elevation [deg]'
            xaxis, yaxis = getXYAxisLayouts()
            fig = go.Figure()
            for i, s in enumerate(sources):
                name = s
                if '--' not in sources[s]['common name']:
                    name = sources[s]['common name']
                label = "{0:}<br>({1:3.1f} Jy)".format(name, sources[s]['1mm flux value'])
                fig.add_trace(
                    go.Scatter(
                        x=dtimes,
                        y=sources[s]['elevation'],
                        mode='lines', name=name),
                )
                w = np.argmax(sources[s]['elevation'])
                fig.add_annotation(
                    x=dtimes[w],
                    y=sources[s]['elevation'][w]-2.5,
                    text=label,
                    align='center',
                    showarrow=False,
                    arrowhead=1,
                    ax=30,
                    ay=10,
                    font=dict(size=12),
                )
            fig.update_layout(
                height=500,
                plot_bgcolor="white",
                title='Bright ALMA Grid Sources',
                xaxis=xaxis,
                yaxis=yaxis,
                font={'size': 16,},
                xaxis_title=xtitle,
                yaxis_title=ytitle, 
            )
            fig.layout.yaxis.range = [40, 90]
            fig.update_layout(showlegend=False)
            fig.update_layout(autosize=True)
            return fig

        
    
        # ---------------------------
        # Update the uptimes plot
        # ---------------------------
        @app.callback(
            [
                Output(plot.id, 'figure')
            ],            
            [
                Input(obsDate.id, 'date')
            ],
        )
        def updateUptimesPlot(date):
            date = date.split('T')[0]
            date = date+'T12:00:00'
            dt = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
            dt = dt.replace(tzinfo=timezone.utc)
            fig = getUptimesPlot(dt)
            return [fig]


        def getUptimesPlot(dt):
            xtitle = 'Universal Time'
            ytitle = 'Elevation [deg]'
            xaxis, yaxis = getXYAxisLayouts()
            fig = go.Figure()
            td = timedelta(hours=14)
            startTime = dt-td
            endTime = dt+td
            sun_rise, sun_set = getSunTimes(startTime)
            fig.add_trace(
                go.Scatter(
                    x=[sun_rise]*2,
                    y=[0, 90],
                    mode='lines',
                    line=dict(color='red', dash='dash',),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=[sun_set]*2,
                    y=[0, 90],
                    mode='lines',
                    line=dict(color='black', dash='dash',),
                    )
                )
            
            for source in data:
                t = data[source]['data']
                st = t[(t['date_time'] >= startTime) &
                       (t['date_time'] <= endTime)]
                fig.add_trace(
                    go.Scatter(
                        x=st['date_time'],
                        y=st['elevation'],
                        name=source
                    ))

                w = np.argmax(st['elevation'])
                fig.add_annotation(
                    x=st['date_time'][w],
                    y=st['elevation'][w]-5,
                    text=source,
                    showarrow=False,
                    arrowhead=1,
                    ax=30,
                    ay=10
                )
                
            fig.update_layout(
                height=500,
                plot_bgcolor="white",
                title='Calibrator Targets',
                xaxis=xaxis,
                yaxis=yaxis,
                font={'size': 16,},
                xaxis_title=xtitle,
                yaxis_title=ytitle, 
            )
            fig.layout.yaxis.range = [30, 90]


            # Add shading for daytime
            fig.add_shape(
                # Rectangle reference to the axes
                type="rect",
                xref="x",
                yref="paper",
                x0=startTime,
                y0=0,
                x1=sun_set,
                y1=1,
                fillcolor="LightYellow",
                opacity=0.5,
                layer="below",
                line_width=0,
            )
            fig.add_shape(
                # Rectangle reference to the axes
                type="rect",
                xref="x",
                yref="paper",
                x0=sun_rise,
                y0=0,
                x1=endTime,
                y1=1,
                fillcolor="LightYellow",
                opacity=0.5,
                layer="below",
                line_width=0,
            )
            # Add annotations for sunset and sunrise
            fig.add_annotation(
                x=sun_set,
                y=90.,
                text="Sunset",
                showarrow=False,
                yshift=10,
                font=dict(color="black")
            )
            fig.add_annotation(
                x=sun_rise,
                y=90.,
                text="Sunrise",
                showarrow=False,
                yshift=10,
                font=dict(color="red")
            )

            fig.add_layout_image(
                dict(
                    source=sunny,
                    xref="paper", yref="paper",
                    x=0.66, y=0.65,
                    sizex=0.3, sizey=0.3,
                    xanchor="right", yanchor="bottom"
                )
            )
            fig.update_layout(showlegend=False)
            fig.update_layout(autosize=True)
            return fig


def getSunTimes(startTime):
    lmt = getLMT()
    # find the sunrise and sunset times in this range
    sun_rise = lmt.sun_rise_time(Time(startTime), which='next').datetime
    sun_set = lmt.sun_set_time(Time(startTime), which='next').datetime
    sun_rise = sun_rise.replace(tzinfo=timezone.utc)
    sun_set = sun_set.replace(tzinfo=timezone.utc)
    return sun_rise, sun_set


def getLMT():
    LMT = EarthLocation.from_geodetic(-97.31481605209875,
                                      18.98578175043638, 4500.)
    lmt = Observer(location=LMT, name="LMT")
    return lmt


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
            size=16,
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
            size=16,
            color="rgb(82, 82, 82)",
        ),
    )
    return xaxis, yaxis


DASHA_SITE = {
    "dasha": {
        "template": ToltecAlmaCalAvailability,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", False),
    }
}
