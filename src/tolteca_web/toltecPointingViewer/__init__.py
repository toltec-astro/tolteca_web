from dash_component_template import ComponentTemplate, NullComponent
from ..toltec_dp_utils.ToltecSignalFits import ToltecSignalFits
from ..common.plots.surface_plot import SurfacePlot
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from tollan.utils.log import logger
from astropy.nddata import Cutout2D
import dash_core_components as dcc
from ..common import LabeledInput
from astropy.table import Table
from astropy import units as u
import plotly.graph_objs as go
import plotly.express as px
import dash_daq as daq
from dash import html
from glob import glob
import numpy as np
import dash_table
import functools
import sys
import os


class ToltecPointingViewer(ComponentTemplate):
    class Meta:
        component_cls = dbc.Container

    def __init__(
        self,
        title_text="Toltec Pointing Observation Viewer",
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

        # Again ... cheating
        self.tsf = None

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
        dPath = "/Users/wilson/Desktop/tmp/macs0717/pointing/102381/"
        g = glob("{}redu*/raw/".format(dPath))
        gp = glob("/Users/wilson/Desktop/tmp/cosmos/redu*/")
        g = g + gp
        g = g + list(glob("./redu*/coadded/raw/"))
        pathOptions = [{"label": gg, "value": gg} for gg in g]

        # pull down to select signal fits path
        controls_panel, views_panel, bigBox = body.grid(3, 1)
        controlBox = controls_panel.child(dbc.Row).child(dbc.Col, width=5)
        settingsRow = controlBox.child(dbc.Row)
        fitsPath_select = settingsRow.child(dbc.Col).child(
            dcc.Dropdown,
            options=pathOptions,
            placeholder="Select Path",
            value="",
            searchable=False,
            style=dict(width="100%", verticalAlign="middle"),
        )

        # Put in a break
        bigBox.child(dbc.Row).child(html.Br)

        # The image and table layout
        box = bigBox.child(dbc.Row, class_name='align-items-center')
        imageCol = box.child(dbc.Col, width=8)
        images = imageCol.child(dcc.Graph)
        zoomSwitch = imageCol.child(dbc.Row).child(dbc.Col, width=2).child(
            daq.ToggleSwitch, size=30, value=False, label=["Full Map", "Zoom on Source"])

        
        tableCol = box.child(dbc.Col, width=4)
        # The table with the fit values
        columns = [
            {"name": "Array", "id": "array"},
            {"name": "Flux [mJy]", "id": "fitted_flux"},
            {"name": 'dAz ["]', "id": "az_offset"},
            {"name": 'dEl ["]', "id": "el_offset"},
            {"name": 'Az FWHM ["]', "id": "az_fwhm"},
            {"name": 'El FWHM ["]', "id": "el_fwhm"}
        ]
        fitTableTitle = tableCol.child(
            dbc.Row, style={'paddingTop': '10px',
                            'textAlign': 'center'}).child(
                html.H4, "Fitted Beam Parameters")
        fitTable = tableCol.child(dbc.Row, style={'paddingTop': '10px'}).child(
            dash_table.DataTable,
            columns=columns,
            data=[
                {"array": a, "fitted_flux": "-", "az_offset": "-",
                 "el_offset": "-", "az_fwhm": "-", "el_fwhm": "-"}
                for a in ['a1100', 'a1400', 'a2000']],
            merge_duplicate_headers=True,
            style_cell={
                'textAlign': 'center',
                'padding': '10px'},
            style_header={
                'backgroundColor': 'lightyellow',
                'fontWeight': 'bold'
            },
            style_cell_conditional=[
                {'if': {'column_id': 'array'},
                 'fontWeight': 'bold'}
            ],
        )

        # The table with the telescope values
        columns = [
            {"name": "Offset Type", "id": "type"},
            {"name": "Az/x", "id": "Az"},
            {"name": "El/y", "id": "El"},
        ]
        telTableTitle = tableCol.child(
            dbc.Row, style={'paddingTop': '50px',
                            'textAlign': 'center'}).child(
                html.H4, "Telescope Offsets")
        telTable = tableCol.child(dbc.Row, style={'paddingTop': '10px'}).child(
            dash_table.DataTable,
            columns=columns,
            data=[
                {"type": "M2", "Az": "-", "El": "-"},
                {"type": "Receiver", "Az": "-", "El": "-"},
                {"type": "User", "Az": "-", "El": "-"},
                {"type": "Paddle", "Az": "-", "El": "-"},
            ],
            merge_duplicate_headers=True,
            style_cell={
                'textAlign': 'center',
                'padding': '10px'},
            style_header={
                'backgroundColor': '#9DEAB8',
                'fontWeight': 'bold'
            },
            style_cell_conditional=[
                {'if': {'column_id': 'type'},
                 'fontWeight': 'bold'}
            ],
        )
        
        super().setup_layout(app)
        self._registerCallbacks(
            app,
            images,
            fitTable,
            telTable,
            fitsPath_select,
            zoomSwitch,
        )
        return

    def _registerCallbacks(
            self,
            app,
            images,
            fitTable,
            telTable,
            fitsPath_select,
            zoomSwitch,
    ):

        # ---------------------------
        # FITS Path dropdown
        # ---------------------------
        @app.callback(
            [
                Output(images.id, "figure"),
                Output(fitTable.id, "data"),
                Output(fitTable.id, "style_data_conditional"),
                Output(telTable.id, "data"),
            ],
            [
                Input(fitsPath_select.id, "value"),
                Input(zoomSwitch.id, "value"),
            ],
            prevent_initial_call=True,
        )
        def primaryDropdown(path, zoom):
            if (path == "") | (path is None):
                raise PreventUpdate
            ppt = glob(path+'ppt*.ecsv')[0]
            fitData = read_fit_data(ppt)
            table_style = getFitTableStyle(fitData)
            fig = makePointingImage(path, zoom, fitData)
            telData = read_tel_data(path)
            return [fig, fitData, table_style, telData]

        
def getFitTableStyle(rows):
    # Define your thresholds for each row
    thresholds = {
        'a1100': {'az_fwhm': 6.5, 'el_fwhm': 6.5},
        'a1400': {'az_fwhm': 8.5, 'el_fwhm': 8.5},
        'a2000': {'az_fwhm': 11.5, 'el_fwhm': 11.5},
    }
    style = []
    for i, row in enumerate(rows):
        array = row['array']
        if array in thresholds:
            if float(row['az_fwhm'].split('±')[0]) > thresholds[array]['az_fwhm']:
                style.append({
                    'if': {'row_index': i, 'column_id': 'az_fwhm'},
                    'backgroundColor': '#FF4136',
                    'color': 'white'
                })
            if float(row['el_fwhm'].split('±')[0]) > thresholds[array]['el_fwhm']:
                style.append({
                    'if': {'row_index': i, 'column_id': 'el_fwhm'},
                    'backgroundColor': '#FF4136',
                    'color': 'white'
                })
    return style



def read_fit_data(file_path):
    # Read the ECSV file
    table = Table.read(file_path, format='ascii.ecsv')
    arrays = ['a1100', 'a1400', 'a2000']

    # Process the data
    tableData = []
    for row in table:
        array = arrays[int(row['array'])]
        fitted_flux = "{0:3.0f}±{1:1.0f}".format(row['amp'], row['amp_err'])
        daz = "{0:3.1f}".format(row['x_t'])
        off_el = "{0:3.1f}".format(row['y_t'])
        az_fwhm = "{0:2.1f}±{1:2.1f}".format(row['a_fwhm'], row['a_fwhm_err'])
        el_fwhm = "{0:2.1f}±{1:2.1f}".format(row['b_fwhm'], row['b_fwhm_err'])

        tableData.append({
            'array': array,
            'fitted_flux': fitted_flux,
            'az_offset': daz,
            'el_offset': off_el,
            'az_fwhm': az_fwhm,
            'el_fwhm': el_fwhm
        })
    return tableData


def makePointingImage(path, zoom, fitData):
    arrays = ['a1100', 'a1400', 'a2000']
    images, wcs_list = getImage('signal_I', path, 0.2, arrays, trimEdge=True)
    fixed_range = dict(x=(-150, 150), y=(-150, 150))

    fig = make_subplots(rows=1, cols=3, shared_xaxes=True,
                        shared_yaxes=True, subplot_titles=arrays)

    for i, (image, wcs) in enumerate(zip(images, wcs_list), start=1):
        if i<3:
            ytitle = ''
        else:
            ytitle = 'Delta El (arcsec)'

        # Generate delta_az and delta_el coordinates based on WCS
        yp, xp = np.indices(image.shape)
        delta_az, delta_el = wcs.pixel_to_world(xp, yp)

        # Add heatmap trace for each image
        fig.add_trace(go.Heatmap(z=image, x=delta_az[0], y=delta_el[:, 0],
                                 showscale=False), row=1, col=i)
        
        # Add dashed lines at x=0 and y=0
        fig.add_shape(type="line", line=dict(dash="dash", color="white"),
                      x0=0, y0=fixed_range['y'][0],
                      x1=0, y1=fixed_range['y'][1], row=1, col=i)
        fig.add_shape(type="line", line=dict(dash="dash", color="white"),
                      x0=fixed_range['x'][0], y0=0,
                      x1=fixed_range['x'][1], y1=0, row=1, col=i)

        # Update x-axis and y-axis for each subplot with the fixed range
        fig.update_xaxes(title="Delta Az (arcsec)", range=fixed_range['x'], row=1, col=i)
        fig.update_yaxes(title=ytitle, range=fixed_range['y'], row=1, col=i,
                         showticklabels=(i == 1))

        # Limit view to 20x20 zoom around point source.
        # We will use the 2.0mm map's coordinates for the source.
        if(zoom):
            delta_az = float(fitData[2]['az_offset'])
            delta_el = float(fitData[2]['el_offset'])
            fig.update_xaxes(range=(delta_az-15., delta_az+15.))
            fig.update_yaxes(range=(delta_el-15., delta_el+15.))

    # Set global layout options
    fig.update_layout(title_text="Pointing Images", plot_bgcolor="white")

    return fig


def read_tel_data(path):
    header = getHeader(path)
    keys = [
        'HEADER.POINTMODEL.AZRECEIVEROFF',
        'HEADER.POINTMODEL.ELRECEIVEROFF',
        'HEADER.POINTMODEL.AZUSEROFF',
        'HEADER.POINTMODEL.ELUSEROFF',
        'HEADER.POINTMODEL.AZPADDLEOFF',
        'HEADER.POINTMODEL.ELPADDLEOFF',
        'HEADER.M2.XREQ',
        'HEADER.M2.YREQ',
    ]

    m2 = [header['HEADER.M2.XREQ'], header['HEADER.M2.XREQ']]
    rx = [header['HEADER.POINTMODEL.AZRECEIVEROFF'],
          header['HEADER.POINTMODEL.ELRECEIVEROFF']]
    us = [header['HEADER.POINTMODEL.AZUSEROFF'],
          header['HEADER.POINTMODEL.ELUSEROFF']]
    pa = [header['HEADER.POINTMODEL.AZPADDLEOFF'],
          header['HEADER.POINTMODEL.ELPADDLEOFF']]

    # convert angles to arsec
    for v in [rx, us, pa]:
        v[0] = np.rad2deg(v[0])*3600.
        v[0] = '{0:3.2f} ["]'.format(v[0])
        v[1] = np.rad2deg(v[1])*3600.
        v[1] = '{0:3.2f} ["]'.format(v[1])
    m2[0] = '{0:3.2f} [um]'.format(m2[0])
    m2[1] = '{0:3.2f} [um]'.format(m2[1])
    
    data=[
        {"type": "M2", "Az": m2[0], "El": m2[1]},
        {"type": "Receiver", "Az": rx[0], "El": rx[1]},
        {"type": "User", "Az": us[0], "El": us[1]},
        {"type": "Paddle", "Az": pa[0], "El": pa[1]},
    ]
    return data


def getHeader(path, array='a1100'):
    tsf = ToltecSignalFits(path=path, array=array)
    return tsf.headers[0]


def getImage(name, path, weightCut, arrays, trimEdge=False):
    images = []
    wcs_list = []

    # First pass to determine the max size needed across all arrays
    max_ysize, max_xsize = 0, 0
    for array in arrays:
        tsf = ToltecSignalFits(path=path, array=array)
        tsf.setWeightCut(weightCut)
        image = tsf.getMap(name)
        if trimEdge:
            nz = np.nonzero(image)
            ysize = nz[0].max() - nz[0].min() + 1
            xsize = nz[1].max() - nz[1].min() + 1
            max_ysize = max(max_ysize, ysize)
            max_xsize = max(max_xsize, xsize)

    # Second pass to create uniformly sized cutouts
    for array in arrays:
        tsf = ToltecSignalFits(path=path, array=array)
        tsf.setWeightCut(weightCut)
        image = tsf.getMap(name)
        wcs = tsf.getMapWCS(name)
        if trimEdge:
            nz = np.nonzero(image)
            ysize = nz[0].max() - nz[0].min() + 1
            xsize = nz[1].max() - nz[1].min() + 1
            ypos = (nz[0].min() + nz[0].max()) // 2
            xpos = (nz[1].min() + nz[1].max()) // 2

            # Calculate padding to apply
            pad_y = (max_ysize - ysize) // 2
            pad_x = (max_xsize - xsize) // 2

            # Apply padding
            padded_image = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)),
                                  'constant', constant_values=0)

            # Adjust WCS for padding
            padded_wcs = wcs.deepcopy()
            padded_wcs.wcs.crpix[0] += pad_x
            padded_wcs.wcs.crpix[1] += pad_y

            # Now create a cutout using the max size
            cutout = Cutout2D(padded_image, (xpos+pad_x, ypos+pad_y),
                              (max_xsize, max_ysize), wcs=padded_wcs)
            images.append(cutout.data)
            wcs_list.append(cutout.wcs)
        else:
            # No trimming, just append the original image and WCS
            images.append(image)
            wcs_list.append(wcs)

    return images, wcs_list



DASHA_SITE = {
    "dasha": {
        "template": ToltecPointingViewer,
        "THEME": dbc.themes.LUMEN,
        "DEBUG": os.environ.get("DASH_DEBUG", True),
    }
}
