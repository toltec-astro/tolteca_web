"""
To Do:
 - 
"""
from dash_component_template import ComponentTemplate
from dash.dependencies import Input, Output, State
from .utilities import readTargetData, fetchProjectID
from datetime import datetime, date, timedelta
from astropy.coordinates import EarthLocation, AltAz
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from astropy.time import Time, TimeDelta
import dash_bootstrap_components as dbc
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
from dash import html
from dash import dcc
from dash import ctx
import numpy as np
import functools
import time
import dash
import os


class ToltecCommissioningProjects(ComponentTemplate):
    class Meta:
        component_cls = dbc.Container

    def __init__(
        self,
        title_text="Toltec Commissioning Projects Viewer",
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
            
        # Make an initial read of the 2024 Commissioning Projects Spreadsheet
        sourceTable = readTargetData()

        # A little setup 
        spacing = body.child(dbc.Row).child(html.H1)
        hbox = body.child(dbc.Row)

        # Create Start time input and duration
        # Note that default start time should be current time
        c_body = hbox.child(dbc.Col, width=2).child(dbc.Row)
        defaultTime = datetime.now()
        minTime = datetime(2024, 1, 15)
        
        # The date selection control
        obsDateRow = c_body.child(html.Div, className='d-flex justify-content-end',
                                  justify='end')
        obsDateRow.child(html.Label,
                         "Obs Date: ",
                         style = {
                             'display': 'flex',
                             'align-items': 'center', 
                             'height': '100%'
                         })
        obsDate = obsDateRow.child(dcc.DatePickerSingle,
                                   min_date_allowed=date(2024, 1, 15),
                                   initial_visible_month=defaultTime,
                                   date=defaultTime,
                                   style=dict(width='45%',))

        # A pair of buttons that reset the visibilities of projects
        foo = c_body.child(dbc.Row).child(html.Hr)
        style = {
            'width': '33%',
            'border-radius': '15px',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'display': 'block'}
        allOn = c_body.child(dbc.Row).child(html.Button, 'Plot All', n_clicks=0, style=style)
        c_body.child(dbc.Row)
        allOff = c_body.child(dbc.Row).child(html.Button, 'Plot None', n_clicks=0, style=style)
        controls = {
            'plot all': allOn,
            'plot none': allOff,}

        # Make the grid of project buttons
        b_body = hbox.child(dbc.Col, width=10).child(dbc.Row)
        projects = makeProjectGrid(b_body, sourceTable)

        # And the uptimes plot
        plot = body.child(dbc.Row).child(dcc.Graph)
                
        super().setup_layout(app)

        self._registerCallbacks(
            app,
            obsDate,
            plot,
            sourceTable,
            projects,
            controls,
        )
        return

    
    def _registerCallbacks(
            self,
            app,
            obsDate,
            plot,
            sourceTable,
            projects,
            controls,
    ):

        visOutputs = [Output(projects[p]['visible'].id, 'on') for p in projects]
        # ---------------------------
        # Update the Source Visibilities from button inputs
        # ---------------------------
        @app.callback(
            visOutputs,
            [
                Input(controls['plot all'].id, 'n_clicks'),
                Input(controls['plot none'].id, 'n_clicks'),
            ],
        )
        def updateAllVis(nAll, nNone):
            triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0] if callback_context.triggered else None
            if triggered_id == controls['plot all'].id:
                print("Plot All button was pressed")
                return [True] * len(projects)
            elif triggered_id == controls['plot none'].id:
                print("Plot None button was pressed")
                return [False] * len(projects)
            else:
                pass
            return dash.no_update

        

        visInputs = [Input(projects[p]['visible'].id, 'on') for p in projects]
        # ---------------------------
        # Update the Source Uptimes plot
        # ---------------------------
        @app.callback(
            [
                Output(plot.id, 'figure')
            ],            
            [
                Input(obsDate.id, 'date')
            ] + visInputs,
        )
        def updateSourcesPlot(date, *args):
            date = date.split('T')[0]
            date = date+'T06:30:00'
            dt = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
            fig = getSourcesPlot(dt, list(args))
            return [fig]


        def getSourcesPlot(dt, visible):
            # deal with times first
            td = timedelta(hours=14)
            startTime = dt-td
            sun_rise, sun_set = getSunTimes(startTime)
            quarterHour = timedelta(hours=0.25)
            sr = Time(sun_rise+quarterHour)
            ss = Time(sun_set-quarterHour)
            increment = TimeDelta(900, format='sec')
            times = Time(ss + increment * range(int((sr - ss) / increment)))
            dtimes = [time.to_datetime() for time in times]

            # Need an lmt to calculate source elevations
            lmt = getLMT()

            # Generate the plot data in a big dictionary.
            altaz = AltAz(obstime=times, location=lmt.location)
            sources = dict()
            for p, v in zip(projects, visible):
                t = fetchProjectID(p, sourceTable)
                for s in t:
                    coords = s['coords']
                    aa = coords.transform_to(altaz)
                    if any(alt.alt.deg > 30 for alt in aa):
                        name = s['Source']
                        sources[name] = dict()
                        sources[name]['elevation'] = aa.alt.deg
                        sources[name]['PID'] = s['Proposal Id']
                        sources[name]['plot'] = v

            # Now we can loop through the sources and plot them
            xtitle = 'Universal Time'
            ytitle = 'Elevation [deg]'
            xaxis, yaxis = getXYAxisLayouts()
            fig = go.Figure()
            for s in sources.keys():
                if sources[s]['plot'] == False:
                    continue
                template = f"{sources[s]['PID']}<br>"
                template += f'Source: {s}<br>Time: %{{x}}<br>'
                template += f'Elevation: %{{y:.2f}}<extra></extra>'
                fig.add_trace(
                    go.Scatter(
                        x=dtimes,
                        y=sources[s]['elevation'],
                        mode='lines',
                        name=s,
                        hoverinfo='x+y',  # Display x and y values
                        hovertemplate=template,
                    ),
                )
                w = np.argmax(sources[s]['elevation'])
                fig.add_annotation(
                    x=dtimes[w],
                    y=sources[s]['elevation'][w]-2.5,
                    text=s,
                    showarrow=False,
                    arrowhead=1,
                    ax=30,
                    ay=10
                )
            fig.update_layout(
                height=600,
                plot_bgcolor="white",
                title='Commissioning Targets ({})'.format(dt.strftime("%d-%b-%Y")),
                xaxis=xaxis,
                yaxis=yaxis,
                font={'size': 16,},
                xaxis_title=xtitle,
                yaxis_title=ytitle, 
            )
            fig.layout.yaxis.range = [30, 90]

            #sunrise and sunset lines
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
            
            # Add shading for daytime
            fig.add_shape(
                # Rectangle reference to the axes
                type="rect",
                xref="x",
                yref="paper",
                x0=dtimes[0],
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
                x1=dtimes[-1],
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
            
            fig.update_layout(showlegend=False)
            fig.update_layout(autosize=True)
            return fig


def makeProjectGrid(box, sourceTable):
    propIds = sorted(list(set(sourceTable['Proposal Id'])))
    nProps = len(propIds)
    nCols = 4
    nRows = np.ceil(nProps/nCols)

    # Create a set of column labels
    style = {'fontWeight': 'bold', 'textAlign': 'center'}
    row = box.child(dbc.Row)
    for i in range(nCols):
        cell = row.child(dbc.Col, width=3).child(dbc.Row)
        name = cell.child(dbc.Col, width=4).child(html.H6, 'Project ID', style=style)
        rank = cell.child(dbc.Col, width=1).child(html.H6, 'Rank', style=style)
        prio = cell.child(dbc.Col, width=1).child(html.H6, "Prio", style=style)
        vis = cell.child(dbc.Col, width=1).child(html.H6, "Vis?", style=style)
        comp = cell.child(dbc.Col, width=3).child(html.H6, "Comp%", style=style)
    
    # Generate a dictionary for the projects
    style = {'textAlign': 'center'}
    projects = dict()
    for i, p in enumerate(propIds):
        projects[p] = dict()
        t = fetchProjectID(p, sourceTable)
        projects[p]['nSources'] = len(t['Source'])
        projects[p]['Ranking'] = t[0]['Ranking']
        projects[p]['Priority'] = t[0]['Priority']
        if((i % nCols) == 0):
            row = box.child(dbc.Row)
        cell = row.child(dbc.Col, width=3).child(dbc.Row)
        label = cell.child(dbc.Col, width=4).child(html.H6, p, style=style)
        rank = cell.child(dbc.Col, width=1).child(html.H6, projects[p]['Ranking'],
                                                  style=style)
        prio = cell.child(dbc.Col, width=1).child(html.H6, "1", style=style)
        check = cell.child(dbc.Col, width=1).child(
            daq.BooleanSwitch,
            on=True,
            color='lightgreen',
            style={'transform': 'scale(0.5)'})
        complete = cell.child(dbc.Col, width=3).child(html.H6, "0%", style=style)
        projects[p]['visible'] = check
        projects[p]['label'] = label
        projects[p]['priority cell'] = prio
        projects[p]['rank cell'] = rank
        projects[p]['complete cell'] = complete
    return projects


def getSunTimes(startTime):
    lmt = getLMT()
    # find the sunrise and sunset times in this range
    sun_rise = lmt.sun_rise_time(Time(startTime), which='next').datetime
    sun_set = lmt.sun_set_time(Time(startTime), which='next').datetime
    return sun_rise, sun_set


def getLMT():
    LMT = EarthLocation.from_geodetic(-97.31481605209875,
                                      18.98578175043638, 4500.)
    lmt = Observer(location=LMT, name="LMT", timezone="US/Central")
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



DASHA_SITE = {
    "dasha": {
        "template": ToltecCommissioningProjects,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", True),
    }
}
