"""A VNASweep viewer."""

import functools
import json
from multiprocessing import Lock

import astropy.units as u
import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from tollan.utils.log import logger, timeit
from tolteca_datamodels.toltec.ncfile import NcFileIO

from ..base import ViewerBase
from ..common import ChecklistPager, LabeledChecklist
from ..common.plots.utils import ColorPalette, make_subplots
from .kids_select import ObsnumNetworkArraySelect
from .sweep_check import CheckSweep, Despike, SweepDataBitMask


class SweepViewer(ViewerBase):
    """A viewer for sweep data."""

    class Meta:  # noqa: D106
        component_cls = dbc.Container

    def __init__(
        self,
        title_text="Sweep Viewer",
        **kwargs,
    ):
        kwargs.setdefault("fluid", True)
        super().__init__(**kwargs)
        self._title_text = title_text

    def setup_layout(self, app):  # noqa: C901, PLR0915
        """Set up the data prod viewr layout."""
        container = self
        header, body = container.grid(2, 1)
        header.child(html.H3, self._title_text)

        controls_panel, views_panel = body.grid(2, 1)
        kids_select = controls_panel.child(ObsnumNetworkArraySelect())
        data_items_store = controls_panel.child(dcc.Store)

        self.state_manager.register("data_items_by_roid", data_items_store, ("data",))

        data_items_selected = kids_select.selected

        # view setup
        graph_tabs = views_panel.child(dbc.Tabs)
        graph_style = {"min-width": "600px"}

        sweep_check_tab = graph_tabs.child(
            dbc.Tab,
            label="Channel Summary",
            tab_id="summary",
        )

        summary_table_container, summary_graph_container = sweep_check_tab.grid(2, 1)

        summary_table = summary_table_container.child(
            dag.AgGrid,
            columnSize="autoSize",
            # dashGridOptions={"domLayout": "autoHeight"},
            style={"height": 200},
        )

        sweep_check_graph = summary_graph_container.child(dcc.Loading).child(
            dcc.Graph,
            style=graph_style,
            className="mt-4",
        )

        sweep_view_tab = graph_tabs.child(dbc.Tab, label="S21-f", tab_id="S21-f")
        sweep_view_control_container, sweep_view_panel = sweep_view_tab.grid(2, 1)
        sweep_view_control_form = sweep_view_control_container.child(dbc.Form).child(
            dbc.Row,
        )
        downsampling_select = sweep_view_control_form.child(
            LabeledChecklist(
                label_text="Select Downsampling Factor",
                className="mt-3 w-auto mr-3",
                size="sm",
                multi=False,
            ),
        ).checklist
        downsampling_select.options = [
            {"label": f"{i}x", "value": i} for i in (1, 2, 4, 8, 16)
        ]
        downsampling_select.value = 16

        sweep_view_pager = sweep_view_control_form.child(
            ChecklistPager(
                className="mt-3 w-auto mr-3",
                title_text="Select Channels",
                n_items_per_page_options=[10, 50, 100],
            ),
        )

        sweep_view_graph = sweep_view_tab.child(dcc.Loading).child(
            dcc.Graph,
            style=graph_style,
            className="mt-4",
        )

        page_view_tab = graph_tabs.child(dbc.Tab, label="I-Q", tab_id="I-Q")
        page_view_panel = page_view_tab

        chan_select_panel, data_view_panel = page_view_panel.grid(2, 1)

        chan_select_graph = chan_select_panel.child(
            dcc.Graph,
            style=graph_style,
        )

        chan_data_graph = data_view_panel.child(
            dcc.Graph,
            style=graph_style,
        )

        super().setup_layout(app)

        # update controls
        @app.callback(
            [
                Output(kids_select.obsnum_select.id, "options"),
                Output(kids_select.obsnum_select.id, "value"),
            ],
            [
                Input(data_items_store.id, "data"),
            ],
        )
        def update_selects(data_items_by_roid):
            if data_items_by_roid is None:
                return dash.no_update
            options = [
                {
                    "label": roid,
                    "value": json.dumps(data_items),
                }
                for roid, data_items in data_items_by_roid.items()
            ]
            value = options[0]["value"] if options else None
            return options, value

        @app.callback(
            [
                Output(summary_table.id, "rowData"),
                Output(summary_table.id, "columnDefs"),
            ],
            [Input(data_items_selected.id, "data")],
        )
        def _make_summary_table(data_items):
            if not data_items:
                return [], []
            for d in data_items:
                d["meta"].update(get_kidsdata_meta(d["filepath"]))

            def _get_summary_meta(m):
                r = {}
                for k in [
                    "nw",
                    "n_chans",
                    "atten_drive",
                    "atten_sense",
                ]:
                    r[k] = m[k]
                r["flo_center (MHz)"] = f'{m["flo_center"] * 1e-6:.3f}'

                return r

            row_data = [_get_summary_meta(d["meta"]) for d in data_items]
            cdefs = [{"field": k} for k in row_data[0]]
            return row_data, cdefs

        @app.callback(
            Output(sweep_check_graph.id, "figure"),
            [Input(data_items_selected.id, "data")],
        )
        def _make_sweep_check_fig(data_items):
            data_items = data_items or []
            return make_sweep_check_fig(data_items)

        @app.callback(
            Output(sweep_view_pager.n_items_store.id, "data"),
            [
                Input(data_items_selected.id, "data"),
            ],
        )
        def _setup_pager(data_items):
            if not data_items:
                return 0
            for d in data_items:
                d["meta"].update(get_kidsdata_meta(d["filepath"]))
            return max(d["meta"]["n_chans"] for d in data_items)

        @app.callback(
            Output(sweep_view_graph.id, "figure"),
            [
                Input(data_items_selected.id, "data"),
                Input(downsampling_select.id, "value"),
                Input(sweep_view_pager.current_page_store.id, "data"),
                Input(graph_tabs.id, "active_tab"),
            ],
        )
        def _make_sweep_view_fig(
            data_items,
            downsampling_value,
            page_value,
            active_tab,
        ):
            if active_tab != sweep_view_tab.tab_id:
                return dash.no_update
            data_items = data_items or []
            chan_slice = (
                slice(page_value["start"], page_value["stop"]) if page_value else None
            )

            fig = make_sweep_view_fig(
                data_items,
                down_sampling=downsampling_value,
                chan_slice=chan_slice,
            )
            fig.update_layout(
                uirevision=",".join([d["meta"]["name"] for d in data_items]),
            )
            return fig

        @app.callback(
            Output(chan_select_graph.id, "figure"),
            [
                Input(data_items_selected.id, "data"),
                Input(graph_tabs.id, "active_tab"),
            ],
        )
        def _make_chan_select_fig(data_items, active_tab):
            if active_tab != page_view_tab.tab_id:
                return dash.no_update

            data_items = data_items or []
            return make_chan_select_fig(data_items)

        @app.callback(
            Output(chan_data_graph.id, "figure"),
            [
                Input(data_items_selected.id, "data"),
                Input(chan_select_graph.id, "clickData"),
                Input(chan_select_graph.id, "figure"),
                Input(graph_tabs.id, "active_tab"),
            ],
        )
        def _make_iq_fig(data_items, chan_select_data, chan_select_fig, active_tab):

            if active_tab != page_view_tab.tab_id:
                return dash.no_update

            data_items = data_items or []
            # print(chan_select_data)
            logger.debug(f"{chan_select_data=}")
            if not chan_select_data:
                trace_data = {
                    "p": 0,
                    "di": 0,
                    "c0": 0,
                    "c1": 25,
                }
            else:
                curve_num = chan_select_data["points"][0]["curveNumber"]
                trace_name = chan_select_fig["data"][curve_num]["name"]
                trace_data = json.loads(trace_name)
            return make_iq_fig(
                data_items[trace_data["di"]] if data_items else None,
                trace_data=trace_data,
            )


nc_read_lock = Lock()


@functools.lru_cache(maxsize=256)
@timeit
def get_kidsdata_io(file_loc):
    """Return the loaded kidsdata."""
    return NcFileIO(file_loc, open=True)


@functools.lru_cache(maxsize=256)
@timeit
def get_kidsdata_meta(file_loc):
    """Return the loaded kidsdata."""
    with nc_read_lock:
        m = get_kidsdata_io(file_loc).read_meta()
    m["chan_axis_data"].sort("f_chan")
    return m


@functools.lru_cache(maxsize=256)
@timeit
def get_kidsdata(file_loc):
    """Return the loaded kidsdata."""
    with nc_read_lock:
        swp = get_kidsdata_io(file_loc).read()
    meta = swp.meta
    # run swp check

    despike_step = Despike()
    checksweep_step = CheckSweep()

    despike_step.run(swp)
    checksweep_step.run(swp)

    # generate figure data for plotting.
    ctx_check_step = CheckSweep.get_or_create_workflow_context(swp)
    ctx_despike_step = Despike.get_or_create_workflow_context(swp)

    fs = swp.frequency
    s21_adu = swp.S21.to_value(u.adu)
    n_chan, n_sweepsteps = fs.shape
    S21_db_orig = ctx_despike_step["find_spike_S21"]["y"]
    S21_db_nospike = ctx_despike_step["despike"]["y_nospike"]
    S21_db_orig_range = np.max(S21_db_orig, axis=-1) - np.min(S21_db_orig, axis=-1)
    S21_db_nospike_range = np.max(S21_db_nospike, axis=-1) - np.min(
        S21_db_nospike,
        axis=-1,
    )
    # ctx_noise = ctx_check_step["check_noise"]
    S21_rms_db_mean = ctx_check_step["check_noise"]["S21_rms_db_mean"]

    chan_id = np.arange(n_chan)
    bm_chan = ctx_check_step["bitmask_chan"]
    m_chan_small_range = bm_chan & SweepDataBitMask.s21_small_range
    m_chan_rms_high = bm_chan & SweepDataBitMask.s21_high_rms
    m_chan_rms_low = bm_chan & SweepDataBitMask.s21_low_rms

    bm = ctx_check_step["bitmask"]
    m_spike = bm & SweepDataBitMask.s21_spike
    chan_spike_count = np.sum(m_spike, axis=1)
    return locals()


_color_palette = ColorPalette()

_fig_layout_default = {
    "xaxis": {
        "showline": True,
        "showgrid": False,
        "showticklabels": True,
        "linecolor": "black",
        "linewidth": 1,
        "ticks": "outside",
    },
    "yaxis": {
        "showline": True,
        "showgrid": False,
        "showticklabels": True,
        "linecolor": "black",
        "linewidth": 1,
        "ticks": "outside",
    },
    "plot_bgcolor": "white",
    "autosize": False,
    "margin": {
        "autoexpand": True,
        "l": 0,
        "r": 10,
        "b": 0,
        "t": 10,
    },
    "modebar": {
        "orientation": "v",
    },
}


def make_chan_select_fig(data_items):
    """Return chan select figure."""
    n_items = len(data_items)
    fig = make_subplots(1, 1, fig_layout=_fig_layout_default)
    fig.update_xaxes(
        title_text="",
        showline=True,
        showgrid=True,
        showticklabels=True,
        ticksuffix=" MHz",
    )
    fig.update_yaxes(
        showticklabels=True,
        showgrid=True,
        showline=False,
        autorange=False,
        range=[n_items, -1],
        tickmode="array",
        tickvals=list(range(n_items)) if n_items > 0 else [],
    )
    fig.update_layout(
        height=70 + 15 * n_items,
        autosize=False,
        showlegend=False,
        margin={
            "l": 0,
            "r": 0,
            "b": 30,
            "t": 30,
        },
    )
    if not data_items:
        return fig
    for d in data_items:
        d["meta"].update(get_kidsdata_meta(d["filepath"]))
    nw_list = [d["meta"]["roach"] for d in data_items]
    fig.update_yaxes(
        ticktext=nw_list,
    )
    n_chans_list = [d["meta"]["n_chans"] for d in data_items]
    n_chans_max = max(n_chans_list)
    chans_per_page = 50
    n_pages = n_chans_max // chans_per_page + (n_chans_max % chans_per_page != 0)
    color_cycle = _color_palette.cycles(0.3, 0.5)

    for p in range(n_pages):
        color, _ = next(color_cycle)
        for di, d in enumerate(data_items):
            m = d["meta"]
            n_chans = m["n_chans"]
            ci0 = p * chans_per_page
            ci1 = ci0 + chans_per_page
            if ci1 > n_chans:
                ci1 = n_chans
            fs = m["chan_axis_data"]["f_chan"].to_value(u.MHz)[ci0:ci1]
            nw = m["roach"]
            fig.add_trace(
                go.Scattergl(
                    x=fs,
                    y=[di + ((p % 2) - 0.5) * 0.0] * len(fs),
                    mode="lines",
                    line={
                        "color": color,
                        "width": 10,
                    },
                    name=json.dumps(
                        {
                            "p": p,
                            "nw": nw,
                            "c0": ci0,
                            "c1": ci1,
                            "di": di,
                        },
                    ),
                ),
            )
    return fig


def make_sweep_check_fig(data_items):
    """Return sweep check figure."""
    n_items = len(data_items)
    fig = make_subplots(
        7,
        1,
        fig_layout=_fig_layout_default,
        shared_xaxes=True,
        shared_yaxes="all",
        subplot_titles=[" " * (i + 1) for i in range(7)],
        vertical_spacing=np.interp(n_items, [0, 13], [0.08, 0.03]),
    )
    fig.update_layout(
        height=800 + 60 * n_items,
        showlegend=False,
        margin={
            "l": 0,
            "r": 0,
            "b": 30,
            "t": 30,
        },
    )
    fig.update_yaxes(
        showticklabels=True,
        showgrid=True,
        showline=False,
        autorange=False,
        range=[n_items, -1],
        tickmode="array",
        tickvals=list(range(n_items)) if n_items > 0 else [],
    )
    if not data_items:
        return fig
    for d in data_items:
        d["data"] = get_kidsdata(d["filepath"])
    name = data_items[0]["meta"]["name"]
    despike_step = data_items[0]["data"]["despike_step"]
    checksweep_step = data_items[0]["data"]["checksweep_step"]
    # image data keys
    fig_defs = [
        {
            "name": (
                f"{name} "
                f"Check S21 Small Range (<{despike_step.min_S21_range_db} dB)"
            ),
            "trace_kw": {
                "zmin": 0,
                "zmax": 1,
            },
            "fig_kw": {
                "col": 1,
                "row": 1,
            },
            "data_key": "m_chan_small_range",
        },
        {
            "name": f"Check S21 High RMS (>{checksweep_step.S21_rms_high_db} dB)",
            "trace_kw": {
                "zmin": 0,
                "zmax": 1,
            },
            "fig_kw": {
                "col": 1,
                "row": 2,
            },
            "data_key": "m_chan_rms_high",
        },
        {
            "name": f"Check S21 Low RMS (<{checksweep_step.S21_rms_low_db} dB)",
            "trace_kw": {
                "zmin": 0,
                "zmax": 1,
            },
            "fig_kw": {
                "col": 1,
                "row": 3,
            },
            "data_key": "m_chan_rms_low",
        },
        {
            "name": "S21 Range",
            "trace_kw": {
                "zmin": 0,
                "zmax": 20,
            },
            "fig_kw": {
                "col": 1,
                "row": 4,
            },
            "data_key": "S21_db_orig_range",
        },
        {
            "name": "S21 Range (filtered)",
            "trace_kw": {
                "zmin": 0,
                "zmax": 20,
            },
            "fig_kw": {
                "col": 1,
                "row": 5,
            },
            "data_key": "S21_db_nospike_range",
        },
        {
            "name": "S21 RMS",
            "trace_kw": {
                "zmin": 0,
                "zmax": checksweep_step.S21_rms_high_db * 5,
            },
            "fig_kw": {
                "col": 1,
                "row": 6,
            },
            "data_key": "S21_rms_db_mean",
        },
        {
            "name": "Channel Spike Count",
            "trace_kw": {},
            "fig_kw": {
                "col": 1,
                "row": 7,
            },
            "data_key": "chan_spike_count",
        },
    ]

    def _make_array(ds):
        # this checks the shape and make a common grid
        n = max(d.shape[0] for d in ds)
        dd = np.full((len(ds), n), np.nan)
        for i, d in enumerate(ds):
            dd[i, : d.shape[0]] = d
        return dd

    for i, fd in enumerate(fig_defs):
        # collate data from all data_items
        dk = fd["data_key"]
        z = _make_array([d["data"][dk] for d in data_items])
        fig.add_trace(
            go.Heatmap(
                z=z,
                colorbar={
                    "len": 0.5 / len(fig_defs),
                    "y": 1 - (0.4 + 1.05 * i) / len(fig_defs),
                },
                colorscale="RdYlGn_r",
                **fd["trace_kw"],
            ),
            **fd["fig_kw"],
        )
        fig.update_yaxes(
            ticktext=[d["meta"]["roach"] for d in data_items],
        )

    fig.for_each_annotation(lambda a: a.update(text=fig_defs[len(a.text) - 1]["name"]))

    return fig


def make_sweep_view_fig(data_items, down_sampling=4, chan_slice=None):
    """Return sweep view figure."""
    n_items = len(data_items)
    fig = make_subplots(
        max(n_items, 1),
        1,
        fig_layout=_fig_layout_default,
        shared_xaxes=True,
        shared_yaxes=True,
        # subplot_titles=[" " * (i + 1) for i in range(4)],
        vertical_spacing=np.interp(n_items, [0, 13], [0.05, 0.01]),
    )
    fig.update_layout(
        height=800 + 100 * n_items,
        showlegend=False,
        margin={
            "l": 0,
            "r": 0,
            "b": 30,
            "t": 30,
        },
    )
    if not data_items:
        return fig
    for d in data_items:
        d["data"] = get_kidsdata(d["filepath"])
    color_cycle = _color_palette.cycle_alternated(1, 0.5)

    for i, d in enumerate(data_items):
        fs = d["data"]["fs"]
        S21_db_orig = d["data"]["S21_db_orig"]
        n_chans = fs.shape[0]
        if chan_slice is None:
            chan_range = range(n_chans)
        else:
            chan_range = range(*(chan_slice.indices(n_chans)))
        for ci in chan_range:
            fig.add_trace(
                go.Scattergl(
                    x=fs[ci][::down_sampling],
                    y=S21_db_orig[ci][::down_sampling],
                    mode="markers",
                    marker={
                        "color": next(color_cycle),
                        "size": 3,
                    },
                ),
                row=i + 1,
                col=1,
            )
        fig.update_yaxes(
            title={
                "text": f'nw{d["meta"]["roach"]}',
            },
            showgrid=True,
            gridcolor="#dddddd",
            ticktext=[v if (v % 5) == 0 else "" for v in range(-1000, 1000)],
            tickvals=list(range(-1000, 1000)),
            row=i + 1,
            col=1,
        )
    m = data_items[0]["data"]["meta"]
    fig.update_layout(
        title=(
            f"{m['master_name']}-{m['obsnum']}-{m['subobsnum']}-{m['scannum']} "
            f"A_drv={m['atten_drive']} A_sen={m['atten_sense']}"
        ),
    )
    return fig


def make_iq_fig(data_item, trace_data):
    """Return IQ plot."""
    fig = make_subplots(
        5,
        5,
        vertical_spacing=0.02,
        fig_layout=_fig_layout_default,
        # shared_xaxes="all",
        # shared_yaxes="all",
    )
    fig.update_layout(
        showlegend=False,
        width=800,
        height=800,
        autosize=True,
        margin={
            "autoexpand": True,
            "l": 10,
            "r": 10,
            "t": 30,
        },
    )
    if not data_item:
        return fig
    color_cycle = _color_palette.cycles(1, 0.5)

    data = get_kidsdata(data_item["filepath"])

    ai = 0
    for i in np.arange(5):
        for j in np.arange(5):
            ci = 5 * i + j + 25 * trace_data["p"]
            s21 = data["s21_adu"][ci, :]
            c_dark, c_light = next(color_cycle)
            fig.add_trace(
                go.Scattergl(
                    x=s21.real,
                    y=s21.imag,
                    mode="markers",
                    marker={"color": c_dark, "size": 4},
                ),
                row=i + 1,
                col=j + 1,
            )
            ai += 1
            fig.update_yaxes(
                automargin=True,
                scaleanchor=f"x{ai}",
                scaleratio=1,
                row=i + 1,
                col=j + 1,
            )

    m = data["swp"].meta
    t = trace_data
    fig.update_layout(
        uirevision=False,
        title=(
            f"{m['interface']} [{t['c0']}:{t['c1']}] "
            f"A_drv={m['atten_drive']} A_sen={m['atten_sense']}"
        ),
    )
    return fig
