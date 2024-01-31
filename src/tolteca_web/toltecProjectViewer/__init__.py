"""
To Do:
 - 
"""
from dash_component_template import ComponentTemplate
from .utilities import fetchProjectID, readTargetData
from dash.dependencies import Input, Output, State
from datetime import datetime, date, timedelta
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from astropy.time import Time, TimeDelta
import dash_bootstrap_components as dbc
from tollan.utils.log import logger
from collections import defaultdict
from ..common import LabeledInput
from dash import callback_context
from astropy.table import Table
from astropy import units as u
from astroplan import Observer
import plotly.graph_objs as go
from astropy.time import Time
from .Project import Project
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


class ToltecProjectViewer(ComponentTemplate):
    class Meta:
        component_cls = dbc.Container

    def __init__(
        self,
        title_text="Toltec Project Viewer",
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
        title_container.child(html.H1(self._title_text, className="my-2"))
        if self._subtitle_text is not None:
            title_container.child(
                html.P(self._subtitle_text, className="text-secondary mx-2")
            )


        # Here is the organization of the data used in this app.
        # The "project" data comes from the LMT database from Kamal and is
        # the actual data collected at the telescope.
        # The "plan" data comes from a spreadsheet that holds the intentions
        # for each project.  Note that this may be replaced by a database
        # query one day, but for now let's keep it as simple as possible.

        # Read the plan data and put it into a dcc.Store.  We can update
        # this with an interval timer later.
        plan = readTargetData()
        planStore = body.child(dbc.Row).child(dcc.Store, data=plan.to_json())
        planIds = sorted(list(set(plan['Proposal Id'])))
        planTimer  = body.child(dbc.Row).child(dcc.Interval,
                                               interval=2*3600.*1000,
                                               n_intervals=0)
        plan = {
            'store': planStore,
            'timer': planTimer,
            }

        # For the project data, we need to see what .sqlite files exist to select from.
        # The options should be a union of the plan and the existing files (for now).
        # A pulldown to select the Project ID
        # For the moment, this is all hard-coded for testing.
        # The SQLite files are supplied by Kamal.
        # Hard code the input path for testing.
        rPath = Path("/Users/wilson/GitHub/tolteca_web/src/tolteca_web")
        dPath = rPath/"toltecProjectViewer"
        dbPath = dPath/"test_data"
        projectFiles = list(dbPath.glob("*.sqlite"))
        projectFiles.sort()
        projectOptions = [{"label": p.stem, "value": str(p)} for p in projectFiles]
        planOptions = [{"label": p,
                        "value": str(dbPath/f"{p}.sqlite")} for p in planIds]

        # pull down to select obs stats file
        pulldownPanel, bigBox = body.grid(2, 1)
        projectSelectBox = pulldownPanel.child(dbc.Row).child(dbc.Col, width=12)
        projectSelectRow = projectSelectBox.child(dbc.Row)
        projectSelectCol = projectSelectRow.child(dbc.Col, width=2)
        projectTitle = projectSelectCol.child(dbc.Row).child(
            html.H5, "Select Project ID", style = {
                'textAlign': 'center',
                'fontWeight': 'bold',
                'color': 'darkblue',},
        )
        projectList = projectSelectCol.child(dbc.Row).child(
            dcc.Dropdown,
            options=projectOptions + planOptions,
            placeholder="Select Project ID",
            value=str(dbPath/"2024-C1-COM-01.sqlite"),
            searchable=False,
            style=dict(width="100%", verticalAlign="middle"),
        )
        controls = {
            'project pull down': projectList,
            }

        # This is the project title
        style = {
            'textAlign': 'center',
            'fontWeight': 'bold',
            'color': 'darkblue',
            'marginTop': '20px',
            'marginBottom': '20px',
        }
        _ = projectSelectRow.child(dbc.Col, width=2)
        titleBox = projectSelectRow.child(dbc.Col, width=4)
        title = titleBox.child(dbc.Row).child(
            html.H2, style=style)

        # The project summary data
        summaryRow = bigBox.child(dbc.Row)
        _ = summaryRow.child(dbc.Col, width=4)
        summaryDiv = summaryRow.child(dbc.Col, width=4).child(html.Div)
        
        # The table of obsnums from the sqlite databases
        style = {
            'textAlign': 'center',
            'fontWeight': 'bold',
            'color': '#4B0082',
            'marginTop': '20px',
            'marginBottom': '0px',
        }
        tableBox = bigBox.child(dbc.Row)
        scienceCol = tableBox.child(dbc.Col, width=6)
        scienceHeader = scienceCol.child(dbc.Row).child(
            html.H4, "Science Observations", style=style)
        scienceDiv = scienceCol.child(
            html.Div, style={'marginLeft': '30px',
                             'marginRight': '30px'})
        pointingCol = tableBox.child(dbc.Col, width=5)
        pointingHeader = pointingCol.child(dbc.Row).child(
            html.H4, "Pointing Observations", style=style)        
        pointingDiv = pointingCol.child(dbc.Row).child(
            html.Div, style={'marginRight': '30px'})
        divs = {
            'science': scienceDiv,
            'pointing': pointingDiv,
            'summary': summaryDiv,
            }

        # Put convenient lists of obsnums at the bottom of the page
        noteRow = body.child(dbc.Row).child(html.Div,
                                            style={'backgroundColor':'darkblue',
                                                   'marginTop':'50px',})
        scienceObsnumList = noteRow.child(dbc.Row).child(html.Div)
        pointingObsnumList = noteRow.child(dbc.Row).child(html.Div)
        projectObsnumList = noteRow.child(dbc.Row).child(html.Div)
        notes = {
            'science obsnums': scienceObsnumList,
            'pointing obsnums': pointingObsnumList,
            'project obsnums': projectObsnumList,
            }
        
        super().setup_layout(app)

        self._registerCallbacks(
            app,
            controls,
            divs,
            title,
            notes,
            plan,
        )
        return

    
    def _registerCallbacks(
            self,
            app,
            controls,
            divs,
            title,
            notes,
            plan,
    ):

        # ---------------------------
        # Reread the plan for the Excel spreadsheet
        # ---------------------------
        @app.callback(
            [
                Output(plan['store'].id, 'data'),
            ],            
            [
                Input(plan['timer'].id, "n_intervals"),
            ],
        )
        def updatePlan(n):
            plan = readTargetData()
            plan = plan.to_json()
            return [plan]


        # ---------------------------
        # Read the data and set the section title
        # ---------------------------
        @app.callback(
            [
                Output(title.id, 'children')
            ],            
            [
                Input(controls['project pull down'].id, 'value')
            ],
        )
        def fetchData(dataFile):
            if dataFile is None:
                raise PreventUpdate
            else:
                pid = dataFile.split('/')[-1]
                pid = pid.split('.')[0]
                title = "Project ID: {}".format(pid)
                return [title]


        # ---------------------------
        # Read the project data and create the summary div
        # ---------------------------
        @app.callback(
            [
                Output(divs['summary'].id, 'children'),
            ],            
            [
                Input(controls['project pull down'].id, 'value'),
                Input(plan['store'].id, 'data'),
            ],
        )
        def updateSummaryDiv(dataFile, plan):
            if dataFile is None:
                raise PreventUpdate
            else:
                pid = dataFile.split('/')[-1]
                pid = pid.split('.')[0]

            # Fetch the data from the project planning spreadsheet.
            plan = pd.read_json(plan)
            planIds = sorted(list(set(plan['Proposal Id'])))
            if pid not in planIds:
                return [html.Div(html.H4(f"{pid} is not in project plan spreadsheet",
                                         style={'textAlign': 'center'}))]
            else:
                project = Project(dataFile)
                plan = plan[plan['Proposal Id'] == pid]
                d = createSummaryDiv(plan, project)
                return [d]

            
        # ---------------------------
        # Read the data and create the output div
        # ---------------------------
        @app.callback(
            [
                Output(divs['science'].id, 'children'),
                Output(divs['pointing'].id, 'children'),
                Output(notes['science obsnums'].id, 'children'),
                Output(notes['pointing obsnums'].id, 'children'),
                Output(notes['project obsnums'].id, 'children'),
            ],            
            [
                Input(controls['project pull down'].id, 'value')
            ],
        )
        def updateScienceDiv(dataFile):
            if dataFile is None:
                raise PreventUpdate
            
            p = Project(dataFile)
            if(len(p.tables) == 0):
                scienceDiv = html.Div(html.H3('No Database For Project',
                                              style={'textAlign': 'center'}))
                pointingDiv = html.Div(html.H3('No Database For Project',
                                              style={'textAlign': 'center'}))
                sciObsDiv = html.Div(html.H5(''))
                poiObsDiv = html.Div(html.H5(''))
                proObsDiv = html.Div(html.H5(''))
            else:
                scienceData = p.createScienceReportData()
                scienceDiv = makeScienceDiv(scienceData)
                pointingData = p.createPointingReportData()
                pointingDiv = makePointingDiv(pointingData)
                style={'color': 'white'}
                sciObsDiv = html.Div(html.H6("science_obsnums = "+repr(p.science_obsnums),
                                             style=style))
                poiObsDiv = html.Div(html.H6("pointing_obsnums = "+repr(p.pointing_obsnums),
                                             style=style))
                proObsDiv = html.Div(html.H6("all_obsnums = "+repr(p.obsnums),
                                             style=style))
            return [scienceDiv, pointingDiv, sciObsDiv, poiObsDiv, proObsDiv]



def createSummaryDiv(plan, project):
    sources = list(set(plan['Source']))
    actual = []
    progress = []
    for s in sources:
        act = project.getSourceInttime(s)
        actual.append(act)
        p = plan[plan['Source'] == s]
        if not p.empty and act != 0:
            progress_value = (p['Time (hr,optimal)'].iloc[0] / act)
        else:
            progress_value = 0.
        progress.append(progress_value)
    plan['Actual'] = actual
    plan['Progress'] = progress
    columns=[
            {'name': 'Source', 'id': 'Source'},
            {'name': 'Desired [hrs]', 'id': 'Time (hr,optimal)'},
            {'name': 'Actual [hrs]', 'id': 'Actual'},
            {'name': 'Progress', 'id': 'Progress', 'type': 'numeric', 
             'format': dash_table.FormatTemplate.percentage(2)}
        ]
    data = plan.to_dict('records')

    table = html.Div([
        dash_table.DataTable(
            id='table',
            columns=columns,
            data=data,
            style_cell={'textAlign': 'center',
                        'padding': '10px',
                        'border': 'none'},
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'border': 'none',
                'borderBottom': '1px solid purple',
            },
        )
    ])
    return table



def makePointingDiv(data):
    # Convert the dictionary of dictionaries to a list of dictionaries
    data_list = list(data.values())

    # Assume the first item has all keys for column headers
    columns = [{'name': key.capitalize(), 'id': key} for key in data_list[0].keys()
               if key != 'valid']

    div = html.Div([
        dash_table.DataTable(
            id='pointing-table',
            columns=columns,
            data=data_list,
            style_cell={'textAlign': 'center'},
            style_data_conditional=[
                # Styles for odd rows
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)',
                }
            ],
            style_header={
                'backgroundColor': '#5bc0de',
                'color': 'white',
                'fontWeight': 'bold'
            },
            export_format="xlsx",
        )
    ])
    return div

            
def makeScienceDiv(data):
    processed_data = process_science_data(data)

    # Assume the first item has all keys for column headers
    columns = [{'name': key.capitalize(), 'id': key} for key in processed_data[0].keys()
               if key != 'valid']

    # Generate some custom styles for the table rows
    rowStyles = getScienceRowStyles()

    div = html.Div([
        dash_table.DataTable(
            id='table',
            columns=columns,
            data=processed_data,
            style_cell={'textAlign': 'center'},
            style_header={
                'backgroundColor': '#31708F',
                'color': 'white',  
                'fontWeight': 'bold'
            },
            style_data_conditional=rowStyles,
            merge_duplicate_headers=True,
            cell_selectable=True,
            export_format="xlsx",
        )
    ])
    return div


def process_science_data(data):
    # Get column names from the first item in the data
    columns = list(next(iter(data.values())).keys())

    # Sort by 'source name' and exclude invalid rows from the total calculation
    sorted_data = sorted(
        (dict(item, **{'int time': round(item['int time'])}) for item in data.values()), 
        key=lambda x: x['source name']
    )

    # Calculate total integration time for each source, excluding invalid rows
    totals = defaultdict(float)
    for item in sorted_data:
        if item.get('valid', 1):  # Check if the row is marked as valid
            totals[item['source name']] += item['int time']

    # Insert total rows and mark invalid rows
    result = []
    last_source = None
    for item in sorted_data:
        if item['source name'] != last_source:
            if last_source is not None:
                result.append(create_total_row(last_source, totals[last_source], columns))
            last_source = item['source name']
        result.append(item)
    if last_source is not None:
        result.append(create_total_row(last_source, totals[last_source], columns))
    return result


def create_total_row(source, total_int_time, columns):
    # Create a dictionary with empty strings for all keys except the last two
    total_row = {col: '' for col in columns[:-1]}  # Empty strings for all but the last column
    total_row[columns[-2]] = f"Total {source}:"  # Label in the second-to-last column
    total_row[columns[-1]] = f"{total_int_time:3.0f}s"  # Total time in the last column
    total_row['valid'] = 1  # Mark the total row as valid
    return total_row


def getScienceRowStyles():
    style_data_conditional=[
        # Styles for odd rows
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)',
        },
        # Style for the total row's label cell
        {
            'if': {
                'filter_query': '{tau} contains "Total"',
                'column_id': 'tau'
            },
            'textAlign': 'right',
            'fontWeight': 'bold',
            'backgroundColor': '#D9EDF7',
            'color': '#212529'
        },
        # Style for the total row's integration time value cell
        {
            'if': {
                'column_id': 'int time',
                'filter_query': '{tau} contains "Total"'
            },
            'fontWeight': 'bold',
            'backgroundColor': '#D9EDF7',
            'color': '#212529'
        },
        {
            'if': {
                'filter_query': '{tau} contains "Total"',
                'column_id': 'date'
            },
            'borderRight': 'none',
        },
        {
            'if': {
                'filter_query': '{tau} contains "Total"',
                'column_id': 'time'
            },
            'borderRight': 'none',
        },
        {
            'if': {
                'filter_query': '{tau} contains "Total"',
                'column_id': 'obsnum'
            },
            'borderRight': 'none',
        },
        {
            'if': {
                'filter_query': '{tau} contains "Total"',
                'column_id': 'source name'
            },
            'borderRight': 'none',
        },
        {
            'if': {
                'filter_query': '{tau} contains "Total"',
                'column_id': 'instrument'
            },
            'borderRight': 'none',
        },
        {
            'if': {
                'filter_query': '{valid} = 0',
            },
            'textDecoration': 'line-through',
            'color': 'red',
        },
    ]
    return style_data_conditional


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
        "template": ToltecProjectViewer,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", True),
    }
}
