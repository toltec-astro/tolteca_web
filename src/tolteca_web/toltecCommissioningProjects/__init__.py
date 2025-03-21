"""
To Do:
 - 
"""
from dash_component_template import ComponentTemplate
from dash.dependencies import Input, Output, State
from .utilities import readTargetData, fetchProjectID, convertPandas2Table
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
import pandas as pd
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
        # and store it in a dcc.Store
        sourceTable = readTargetData()
        sourceTable_dict = sourceTable.to_dict('records')
        sourceData = body.child(dcc.Store, data=sourceTable_dict)
        sourceTable = convertPandas2Table(sourceTable)

        # A little setup 
        spacing = body.child(dbc.Row).child(html.H1)
        hbox = body.child(dbc.Row)
        
        # Create Start time input and duration
        # Note that default start time should be current time
        c_body = hbox.child(dbc.Col, width=2).child(dbc.Row)
        defaultTime = determine_default_date()
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
            'width': '66%',
            'border-radius': '15px',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'display': 'block'}
        foo = c_body.child(dbc.Row)
        bCol = foo.child(dbc.Col, width=6)
        allOn = bCol.child(dbc.Row).child(html.Button, 'Plot All', n_clicks=0, style=style)
        c_body.child(dbc.Row)
        allOff = bCol.child(dbc.Row).child(html.Button, 'Plot None', n_clicks=0, style=style)
        ranks = sorted(list(set(sourceTable['Ranking'])))
        checkCol = foo.child(dbc.Col, width=6)
        check = checkCol.child(dcc.Checklist, options=ranks, value=ranks,
                               inputStyle={"margin-right": "10px"})
        controls = {
            'plot all': allOn,
            'plot none': allOff,
            'ranks': check}

        # Make the grid of project buttons
        b_body = hbox.child(dbc.Col, width=10).child(dbc.Row)
        projects = makeProjectGrid(b_body, sourceTable)

        # And the uptimes plot
        plot = body.child(dbc.Row).child(dcc.Graph)

        # Timers
        timerRow = body.child(dbc.Row)
        projectUpdateInterval = 10 # in seconds
        projectUpdateTimer  = timerRow.child(dcc.Interval,
                                             interval=projectUpdateInterval*1000,
                                             n_intervals=0)
        nowUpdateInterval = 3600 # in seconds
        nowUpdateTimer = timerRow.child(dcc.Interval,
                                        interval=nowUpdateInterval*1000,
                                        n_intervalse=0)
        timers = {
            'project update timer': projectUpdateTimer,
            'now update timer': nowUpdateTimer,
            }

        # Put notes at the bottom of the page
        noteRow = body.child(dbc.Row)
        noteRow.child(
            dbc.Col, width=2,
            style={'backgroundColor': '#e6ffe6'},
        ).child(
            html.H6, "Green denotes that project has sources up now.")
    
        super().setup_layout(app)

        self._registerCallbacks(
            app,
            obsDate,
            plot,
            sourceData,
            projects,
            controls,
            timers,
        )
        return

    
    def _registerCallbacks(
            self,
            app,
            obsDate,
            plot,
            sourceData,
            projects,
            controls,
            timers,
    ):


        outputList = [Output(projects[p]['complete cell'].id, 'children')
                      for p in projects] 
        # ---------------------------
        # Sync the projects completion status with the spreadsheet
        # This needs a finalization of formatting in the Google Sheet.
        # ---------------------------
        @app.callback(
            outputList,
            [
                Input(sourceData.id, 'data'),
            ],
        )
        def updateCompletions(sourceTable_dict):
            return ["0%"]*len(projects)


        # ---------------------------
        # Sync the project data from the Google Sheet with the local data
        # ---------------------------
        @app.callback(
            [
                Output(sourceData.id, 'data'),
            ],
            [
                Input(timers['project update timer'].id, "n_intervals"),
            ],
        )
        def updateCompletions(n):
            sourceTable = readTargetData()
            sourceTable_dict = sourceTable.to_dict('records')
            return [sourceTable_dict]


        # ---------------------------
        # Update the checkbox values from button inputs
        # ---------------------------
        @app.callback(
            [
                Output(controls['ranks'].id, 'value')
            ],
            [
                Input(controls['plot all'].id, 'n_clicks'),
                Input(controls['plot none'].id, 'n_clicks'),
            ],
        )
        def updateCheckbox(nAll, nNone):
            triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0] \
                if callback_context.triggered else None
            if triggered_id == controls['plot all'].id:
                return [[r for r in controls['ranks'].options]]
            elif triggered_id == controls['plot none'].id:
                return [[]]
            else:
                pass
            return dash.no_update


        visOutputs = [Output(projects[p]['visible'].id, 'on') for p in projects]
        # ---------------------------
        # Update the Source Visibilities from button inputs
        # ---------------------------
        @app.callback(
            visOutputs,
            [
                Input(controls['plot all'].id, 'n_clicks'),
                Input(controls['plot none'].id, 'n_clicks'),
                Input(controls['ranks'].id, 'value'),
            ],
        )
        def updateAllVis(nAll, nNone, ranks):
            triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0] if callback_context.triggered else None
            if triggered_id == controls['plot all'].id:
                return [True] * len(projects)
            elif triggered_id == controls['plot none'].id:
                return [False] * len(projects)
            elif triggered_id == controls['ranks'].id:
                visout = [project['Ranking'] in ranks for project in projects.values()]
                return visout
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
                Input(sourceData.id, 'data'),
                Input(obsDate.id, 'date'),
                Input(timers['now update timer'].id, "n_intervals"),
                State(plot.id, 'figure'),
            ] + visInputs,
        )
        def updateSourcesPlot(sourceTable_dict, date, n, fig, *args):
            # convert the dict to an astropy table
            sourceTable = convertPandas2Table(pd.DataFrame(sourceTable_dict))
            
            date = date.split('T')[0]
            date = date+'T06:30:00'
            dt = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
            
            # If the timer goes off, then fig already exists and we
            # want to add the line.  Otherwise, we just modify the
            # figure accordingly to the changed inputs.
            ctx = callback_context
            if not ctx.triggered:
                # This is the first load case.
                fig = getSourcesPlot(dt, list(args), sourceTable)
            else:
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
                if trigger_id == timers['now update timer'].id:
                    fig = go.Figure(fig)
                    fig = addNowLine(dt, fig)
                else:
                    fig = getSourcesPlot(dt, list(args), sourceTable)
            return [fig]


        outputList = [Output(projects[p]['label'].id, 'style')
                      for p in projects] 
        # ---------------------------
        # Changes the background color of the labels for projects with
        # sources that are up at the current time.
        # ---------------------------
        @app.callback(
            outputList,
            [
                Input(obsDate.id, 'date'),
                Input(timers['now update timer'].id, "n_intervals"),
                Input(sourceData.id, 'data'),
            ],
        )
        def updateVisibleProjects(date, n, sourceTable_dict):
            now = datetime.utcnow()
            nd = now.date()
            if date:
                date_only = date.split('T')[0]
                sd = datetime.strptime(date_only, '%Y-%m-%d').date()
            else:
                sd = None
            if nd != sd:
                return [{'backgroundColor': 'white'}]*len(projects)

            # This is a little roundabout, but convert to an astropy table.
            # This will give us a convenient column of coordinates.
            sourceTable = convertPandas2Table(pd.DataFrame(sourceTable_dict))
            lmt = getLMT()
            obstime = Time(now)
            altaz = AltAz(obstime=obstime, location=lmt.location)
            styles = []
            for p in projects:
                t = fetchProjectID(p, sourceTable)
                up = False
                for s in t:
                    coords = s['coords']
                    alt = coords.transform_to(altaz).alt.deg
                    if alt > 30.:
                        up = True
                if(up):
                    styles.append({'backgroundColor': '#e6ffe6'})
                else:
                    styles.append({'backgroundColor': 'white'})
            return styles
        

        def addNowLine(dt, fig):
            now = datetime.utcnow()
            dtimes = fig['data'][0]['x']
            start_time = datetime.fromisoformat(dtimes[0])
            end_time = datetime.fromisoformat(dtimes[-1])
            if(start_time <= now <= end_time):
                fig = modify_trace(fig, 'Now', {'x': [now]*2,
                                                'y': [0, 90]})
                fig.layout.annotations[-1].update(x=now, y=90, text="Current")
            return fig
            
        
        def getSourcesPlot(dt, visible, sourceTable):
            # deal with times first
            td = timedelta(hours=14)
            startTime = dt-td
            sun_rise, sun_set = getSunTimes(startTime)
            quarterHour = timedelta(hours=0.25)
            sr = Time(sun_rise+quarterHour+quarterHour+quarterHour)
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

            # add a line denoting 'Now' but place it at the center of
            # the plot and make it invisible.
            fig.add_trace(
                go.Scatter(
                    x=[dtimes[0]]*2,
                    y=[0, 0],
                    mode='lines',
                    line=dict(color='purple', dash='dash',),
                    name="Now",
                    )
                )

            # and a corresponding invisible annotation
            fig.add_annotation(
                x=dtimes[0],
                y=0.,
                text='',
                showarrow=False,
                yshift=10,
                font=dict(color="purple"),
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
        prio = cell.child(dbc.Col, width=2).child(html.H6, "Prio", style=style)
        vis = cell.child(dbc.Col, width=1).child(html.H6, "Vis?", style=style)
        comp = cell.child(dbc.Col, width=2).child(html.H6, "Comp%", style=style)
    
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
        prio = cell.child(dbc.Col, width=2).child(html.H6, projects[p]['Priority'],
                                                  style=style)
        check = cell.child(dbc.Col, width=1).child(
            daq.BooleanSwitch,
            on=True,
            color='lightgreen',
            style={'transform': 'scale(0.5)'})
        complete = cell.child(dbc.Col, width=2).child(html.H6, "0%", style=style)
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


def determine_default_date():
    now = datetime.utcnow()
    # Get today's sunrise and sunset times
    sunrise_today, sunset_today = getSunTimes(now)
    if sunrise_today <= now <= sunset_today:
        # It's still daytime, use tomorrow's date
        return now + timedelta(days=1)
    else:
        # It's nighttime, use today's date
        return now


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


def modify_trace(fig, trace_name, new_properties):
    """
    Modify a trace in a Plotly figure.

    :param fig: Plotly figure object containing multiple traces
    :param trace_name: Name of the trace to modify
    :param new_properties: Dictionary of new properties to apply to the trace
    """
    for trace in fig.data:
        if trace.name == trace_name:
            for prop, value in new_properties.items():
                setattr(trace, prop, value)
            break
    return fig


DASHA_SITE = {
    "dasha": {
        "template": ToltecCommissioningProjects,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", True),
    }
}
