"""The TolTEC data product viewer."""

import functools
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cachetools.func
import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html
from tollan.utils.general import ObjectProxy
from tollan.utils.log import logger, timeit

from ..base import ViewerBase
from ..common import LabeledDropdown, LiveUpdateSection
from ..data_prod.collector import DataProdCollectorProtocol, QLDataProdCollector
from ..data_prod.conventions import make_toltec_raw_obs_uid

# from ..toltecFocusViewer import ToltecFocusViewer
# from ..toltecObsStatsViewer import ToltecObsStatsViewer
# from ..toltecSignalFitsViewer import ToltecSignalFitsViewer
from ..db import get_sqla_db
from ..toltec_sweep import SweepViewer
from ..toltecAptViewer import ToltecAptViewer
from ..toltecTelViewer import ToltecTelViewer
from ..toltecTonePowerViewer import ToltecTonePowerViewer

# from tollan.utils.fmt import pformat_yaml
from multiprocessing import Lock


class DataProdItemViewer(ViewerBase):
    """A dummy viewer for debugging purpose."""

    class Meta:  # noqa: D106
        component_cls = dbc.Container

    def __init__(
        self,
        title_text="Data Prod Item Viewer",
        **kwargs,
    ):
        kwargs.setdefault("fluid", True)
        super().__init__(**kwargs)
        self._title_text = title_text

    def setup_layout(self, app):
        """Set up the data prod viewr layout."""
        container = self
        header, body = container.grid(2, 1)
        header.child(html.H3, self._title_text)

        meta_container, data_items_container = body.colgrid(2, 1, gy=2)
        meta_table = meta_container.child(
            dag.AgGrid,
            columnDefs=[{"field": "name"}, {"field": "value"}],
        )
        data_items_table = data_items_container.child(
            dag.AgGrid,
            columnSize="autoSize",
        )

        def map_meta(data):
            return {
                "rowData": [{"name": k, "value": v} for k, v in data.items()],
            }

        self.state_manager.register(
            "meta",
            meta_table,
            ["rowData"],
            mapper_func=map_meta,
        )

        def _sort_cols(c):
            return {
                "name": -3,
                "master": -2,
                "interface": -1,
                "obsnum": 0,
                "subobsnum": 1,
                "scannum": 2,
                "cal_obsnum": 3,
                "cal_subobsnum": 4,
                "cal_scannum": 5,
                "data_kind": 6,
                "file_suffix": 7,
                "filepath": 8,
                "cal_filepath": 9,
            }.get(c["field"], 1000)

        _col_defs = {
            "filepath": {
                # "wrapText": True,
                # "autoHeight": True,
            },
            "cal_filepath": {
                # "wrapText": True,
                # "autoHeight": True,
            },
        }

        def map_data_items(data_items):
            row_data = [{"filepath": d["filepath"]} | d["meta"] for d in data_items]
            cdefs = sorted(
                [{"field": k} | _col_defs.get(k, {}) for k in row_data[0]],
                key=_sort_cols,
            )
            return {
                "rowData": row_data,
                "columnDefs": cdefs,
            }

        self.state_manager.register(
            "data_items",
            data_items_table,
            ["rowData", "columnDefs"],
            mapper_func=map_data_items,
        )
        super().setup_layout(app)


class DataProdViewer(ViewerBase):
    """The TolTEC data product viewer."""

    class Meta:  # noqa: D106
        component_cls = dbc.Container

    def __init__(
        self,
        title_text="TolTEC Data Product Viewer",
        subtitle_text="(test version)",
        **kwargs,
    ):
        kwargs.setdefault("fluid", True)
        super().__init__(**kwargs)
        self._title_text = title_text
        self._subtitle_text = subtitle_text
        self.fluid = True

    def setup_layout(self, app):  # noqa: C901, PLR0915
        """Set up the data prod viewr layout."""
        container = self
        header_container, body = container.grid(2, 1)
        header = header_container.child(
            LiveUpdateSection(
                title_component=html.H3(self._title_text),
                interval_options=[5000, 10000, 15000],
                interval_option_value=5000,
            ),
        )
        controls_panel, views_panel = body.grid(2, 1)
        dp_select_container = controls_panel.child(dbc.Form).child(
            dbc.Row,
            className="gx-2 gy-2",
        )

        # pull down to select data prod.
        dp_select = dp_select_container.child(
            LabeledDropdown(
                label_text="Data Prod",
                size="sm",
                placeholder="Select a data product ...",
                className="mb-2 w-auto align-items-start",
            ),
        ).dropdown

        dpa_select = dp_select_container.child(
            LabeledDropdown(
                label_text="Assoc. Data Prod",
                # className='w-auto',
                size="sm",
                placeholder="Select a data product ...",
                className="mb-2 w-auto align-items-start",
            ),
        ).dropdown

        dpa_as_dp_btn = dp_select_container.child(
            dbc.InputGroup,
            className="mb-2 w-auto align-items-start",
        ).child(
            dbc.Button,
            "Set current Assoc. DP as DP",
            size="sm",
        )
        dp_select_feedback = dp_select.parent.feedback
        viewer_defs = [
            # {"label": "Project", "value": "project"},
            #   {"label": "Hskp", "value": "hk"},
            {
                "label": "DataProdInfo",
                "value": "data_prod_item",
                "template": DataProdItemViewer(),
            },
            {
                "label": "TonePower",
                "value": "tone_power",
                "template": ToltecTonePowerViewer(
                    manager_kw={
                        "mapper_funcs": DataProd.get_tone_power_viewer_mapper_funcs(),
                    },
                ),
            },
            {
                "label": "Sweep",
                "value": "sweep",
                "template": SweepViewer(),
            },
            {
                "label": "Tel",
                "value": "tel",
                "template": ToltecTelViewer(
                    manager_kw={
                        "mapper_funcs": DataProd.get_tel_viewer_mapper_funcs(),
                    },
                ),
            },
            {
                "label": "Apt",
                "value": "apt",
                "template": ToltecAptViewer(
                    manager_kw={
                        "mapper_funcs": DataProd.get_apt_viewer_mapper_funcs(),
                    },
                ),
            },
            #   {"label": "Detector", "value": "detector"},
            #   {
            #       "label": "ObsStats",
            #       "value": "obs_stats",
            #       "template": ToltecObsStatsViewer(),
            #   },
            #   {
            #       "label": "SignalFits",
            #       "value": "signal_fits",
            #       "template": ToltecSignalFitsViewer(),
            #   },
            #   {"label": "Focus", "value": "focus", "template": ToltecFocusViewer()},
        ]
        viewers = {}

        viewer_tabs = views_panel.child(dbc.Tabs)
        for viewer_def in viewer_defs:
            template_inst = viewer_def.get(
                "template",
                DataProdItemViewer(
                    title_text=f"{viewer_def['label']} Viewer (WIP...)",
                ),
            )
            tab = viewer_tabs.child(
                dbc.Tab,
                label=viewer_def["label"],
            )
            tab.child(template_inst)
            viewers[viewer_def["value"]] = {
                "tab": tab,
                "content": template_inst,
            }

        # dp_info_store = controls_panel.child(dcc.Store)

        super().setup_layout(app)

        @app.callback(
            [
                Output(dp_select.id, "options"),
                Output(dp_select.id, "valid"),
                Output(dp_select.id, "invalid"),
                Output(dp_select_feedback.id, "type"),
                Output(dp_select_feedback.id, "children"),
                Output(header.loading.id, "children"),
            ],
            [
                Input(header.timer.n_calls_store.id, "data"),
            ],
        )
        def update_dp_select(
            _n_calls,
        ):
            dps, collector_info = collect_data_prods()
            # dps = get_latest_data_prods_from_dpdb()
            options = [
                {
                    "label": dp.make_display_label(),
                    "value": dp.index_filename,
                }
                for dp in dps
            ]
            # value = options[-1]["value"] if len(options) > 0 else dash.no_update
            # value = dash.no_update
            fb_type = "valid" if collector_info.is_active else "invalid"
            fb_content = collector_info.message or ""
            return (
                options,
                fb_type == "valid",
                fb_type == "invalid",
                fb_type,
                fb_content,
                "",
            )

        @app.callback(
            [
                Output(dpa_select.id, "options"),
                Output(dpa_select.id, "value"),
            ],
            [
                Input(dp_select.id, "value"),
            ],
            prevent_initial_call=True,
        )
        def update_dpa_select(index_filename):
            if not index_filename:
                return dash.no_update
            dp = load_data_prod(index_filename)
            assocs = dp.index.get("assocs", [])
            options = [
                {
                    "label": dp.make_display_label(prefix="self - "),
                    "value": dp.index_filename,
                },
            ]
            for dpa in assocs:
                dpa_type = dpa["data_prod_assoc_type"]
                dpa_path = _resolve_path(
                    Path(dpa["filepath"]),
                    Path(dp.index_filepath).parent,
                )
                # validate
                try:
                    dpa_dp = load_data_prod(dpa_path.name)
                except Exception:  # noqa: BLE001
                    valid = False
                else:
                    valid = True
                options.append(
                    {
                        "label": dpa_dp.make_display_label(prefix=f"{dpa_type} - "),
                        "value": dpa_dp.index_filename,
                        "disabled": not valid,
                    },
                )
            return options, options[0]["value"]

        @app.callback(
            Output(dp_select.id, "value"),
            [
                Input(dpa_as_dp_btn.id, "n_clicks"),
                State(dpa_select.id, "value"),
            ],
            prevent_initial_call=True,
        )
        def update_dp_select_value_from_dpa_value(_n_clicks, index_filename):
            return index_filename

        @app.callback(
            Output(dpa_as_dp_btn.id, "disabled"),
            [
                Input(header.timer.n_calls_store.id, "data"),
                Input(dp_select.id, "value"),
                Input(dpa_select.id, "value"),
            ],
            prevent_initial_call=True,
        )
        def check_dp_self(_n_times, dp_index_filename, dpa_index_filename):
            return dp_index_filename == dpa_index_filename

        def make_tab_label(text, color):
            symbol_map = {
                "green": "🟢",
                "red": "🔴",
                "yellow": "🟡",
                "white": "⚪",
            }
            if text[0] in symbol_map.values():
                text = text[1:]
            return symbol_map[color] + text

        def make_viewer_callback(vk, viewer, tab):
            @app.callback(
                [
                    Output(viewer.state_manager.data.id, "data"),
                    Output(tab.id, "label"),
                ],
                [
                    Input(dpa_select.id, "value"),
                    State(tab.id, "label"),
                ],
                prevent_initial_call=True,
            )
            def update_viewer_input_data(index_filename, tab_label):
                if not index_filename:
                    return dash.no_update
                dp = load_data_prod(index_filename)
                data = getattr(
                    dp,
                    f"get_{vk}_viewer_data",
                    dp.get_data_prod_item_viewer_data,
                )()
                if data is None:
                    return None, make_tab_label(tab_label, "red")
                return data, make_tab_label(tab_label, "green")

        for vk, vv in viewers.items():
            viewer = vv["content"]
            tab = vv["tab"]
            make_viewer_callback(vk, viewer, tab)


def _resolve_path(p, parent):
    p = Path(p)
    if p.is_absolute():
        return p
    return (parent / p).resolve()


@dataclass
class DataProd:
    """The data prod container for viewers."""

    index_filepath: str
    index: None | dict = field(repr=False)

    @property
    def type(self):
        """The data prod type."""
        return self.index["meta"]["data_prod_type"]

    @property
    def name(self):
        """The data prod name."""
        return self.index["meta"]["name"]

    @property
    def time_obs(self):
        """The data prod name."""
        return self.index["meta"]["time_obs"]

    def make_display_label(self, prefix=""):
        """Return the display label."""
        if self.type in [
            "dp_raw_obs",
            "dp_basic_reduced_obs",
        ]:
            dk = self.index["data_items"][0]["meta"].get("data_kind", None)
            dk = {
                "ToltecDataKind.VnaSweep": "vnasweep",
                "ToltecDataKind.TargetSweep": "targsweep",
                "ToltecDataKind.Tune": "tune",
                "ToltecDataKind.RawTimeStream": "timestream",
            }.get(dk, "")
            nw = {d["meta"]["roach"] for d in self.index["data_items"] if "roach" in d["meta"]}
            return f"{prefix}{self.name} - {dk}{nw}"
        return f"{prefix}{self.name}"

    @property
    def index_filename(self):
        """The index filename."""
        return Path(self.index_filepath).name

    def _resolve_path(self, p):
        return _resolve_path(p, Path(self.index_filepath).parent)

    def __post_init__(self):
        dt = self._data_items_by_data_kind = {}
        # logger.debug(f"parse dp index:\n{pformat_yaml(self.index)}")
        for d in self.index["data_items"]:
            k = d["meta"].get("data_kind", "ToltecDataKind.Unknown")
            d["filepath"] = self._resolve_path(d["filepath"]).as_posix()
            if "cal_filepath" in d["meta"] and d["meta"]["cal_filepath"] is not None:
                d["meta"]["cal_filepath"] = self._resolve_path(
                    d["meta"]["cal_filepath"],
                ).as_posix()
            if k not in dt:
                dt[k] = []
            dt[k].append(d)

    def get_apt_viewer_data(self):
        """Return apt viewer data if available."""
        files = self._data_items_by_data_kind.get("ArrayPropTable", None)
        if not files:
            return None
        return {
            "aptList": files,
        }

    @staticmethod
    def get_apt_viewer_mapper_funcs():
        """Return apt viewer data mapper funcs."""

        def map_aptList(apt_files):
            options = [{"label": p["path"], "value": p["path"]} for p in apt_files]
            return {
                "options": options,
                "value": options[0]["value"],
            }

        return {
            "aptList": map_aptList,
        }

    def get_tel_viewer_data(self):
        """Return apt viewer data if available."""
        files = self._data_items_by_data_kind.get("ToltecDataKind.LmtTel", None)
        if not files:
            return None
        return {
            "telList": files,
        }

    @staticmethod
    def get_tel_viewer_mapper_funcs():
        """Return the mapper functions for tone power viewer."""

        def map_telList(tel_files):
            options = [{"label": p["filepath"], "value": p["filepath"]} for p in tel_files]
            return {
                "options": options,
                "value": options[0]["value"],
            }

        return {
            "telList": map_telList,
        }

    def get_obs_stats_viewer_data(self):
        """Return apt viewer data if available."""
        return self._data_items_by_data_kind.get("CitlaliStats", None)

    def get_focus_viewer_data(self):
        """Return apt viewer data if available."""
        if self.index["meta"]["data_prod_type"] == "dp_m2":
            return self.index["data_items"]
        return None

    def get_signal_fits_viewer_data(self):
        """Return apt viewer data if available."""
        return self._data_items_by_data_kind.get("Image", None)

    def get_data_prod_item_viewer_data(self):
        """Return the index as-is."""
        return self.index

    def get_tone_power_viewer_data(self):
        """Return the tone power viewer data if available."""
        raw_kids_items = []
        for data_kind in [
            "ToltecDataKind.VnaSweep",
            "ToltecDataKind.TargetSweep",
            "ToltecDataKind.Tune",
            "ToltecDataKind.RawTimeStream",
        ]:
            raw_kids_items.extend(self._data_items_by_data_kind.get(data_kind, []))
        if raw_kids_items:
            # group by obsnum list
            files_by_obsnum = {}
            for d in raw_kids_items:
                obsnum = d["meta"]["obsnum"]
                # skip missing files
                if not Path(d["filepath"]).exists():
                    continue
                if obsnum in files_by_obsnum:
                    files_by_obsnum[obsnum].append(d["filepath"])
                else:
                    files_by_obsnum[obsnum] = [d["filepath"]]
            return {
                "obsnumList": files_by_obsnum,
            }
        return None

    @staticmethod
    def get_tone_power_viewer_mapper_funcs():
        """Return the mapper functions for tone power viewer."""

        def _map_obsnum_list_data(files_by_obsnum):
            # print(f"compose obsnum list data from {files_by_obsnum}")
            options = [
                {
                    "label": obsnum,
                    "value": json.dumps(files),
                }
                for obsnum, files in files_by_obsnum.items()
            ]
            value = options[0]["value"] if options else ""
            return {
                "options": options,
                "value": value,
            }

        return {
            "obsnumList": _map_obsnum_list_data,
        }

    def get_sweep_viewer_data(self):
        """Return the list of raw obsnums."""
        sweep_items = []
        for data_kind in [
            "ToltecDataKind.VnaSweep",
            "ToltecDataKind.TargetSweep",
            "ToltecDataKind.Tune",
        ]:
            sweep_items.extend(self._data_items_by_data_kind.get(data_kind, []))
        if sweep_items:
            # group by raw obs id
            di_by_roid = {}
            for d in sweep_items:
                roid = make_toltec_raw_obs_uid(d["meta"])
                if roid in di_by_roid:
                    di_by_roid[roid].append(d)
                else:
                    di_by_roid[roid] = [d]
            return {
                "data_items_by_roid": di_by_roid,
            }
        return None


# @cachetools.func.ttl_cache(maxsize=256, ttl=5)
# def _load_data_prod(index_filepath):
#     """Return data prod."""
#     index = yaml_load(index_filepath)
#     dp = DataProd(index_filepath=index_filepath, index=index)
#     logger.debug(f"loaded data prod: {dp}")
#     return dp


# @cachetools.func.ttl_cache(maxsize=256)
@functools.lru_cache(maxsize=2**12)
def load_data_prod(index_filename):
    """Return data prod."""
    store = data_prod_collector.data_prod_index_store
    index_filepath = store.get_filepath(index_filename)
    index = store[index_filename]
    dp = DataProd(
        index_filepath=index_filepath,
        index=index,
    )
    logger.debug(f"loaded data prod: {dp}")
    return dp


collect_data_prods_lock = Lock()

@timeit("collect_data_prods", level="INFO")
@cachetools.func.ttl_cache(maxsize=1, ttl=5)
def collect_data_prods():
    """Return the list of data prods."""
    dpc = data_prod_collector
    store = dpc.data_prod_index_store
    with collect_data_prods_lock:
        info = dpc.collect(n_items=50, n_updates=2)
    logger.debug(
        f"collected {len(store)} data prods in store, {info=}",
    )
    # load the dps
    # here we reload the last 20 data products in case they update
    # n_dps = len(store)

    # def _load(i, f):
    #     if i > 20:
    #         # cached
    #         return load_data_prod(f)
    #     return load_data_prod.__wrapped__(f)
    dps = [load_data_prod(f) for f in store.iter_filenames(reverse=True)]
    return dps, info


data_prod_collector: ObjectProxy | DataProdCollectorProtocol = ObjectProxy()


def _post_init():
    data_lmt_rootpath = (
        Path(
            os.environ.get("TOLTECA_WEB_DATA_LMT_ROOTPATH", "/data_lmt"),
        )
        .expanduser()
        .resolve()
    )
    data_prod_output_path = (
        Path(
            os.environ.get("TOLTECA_WEB_DATA_PROD_OUTPUT_PATH", "dataprod_toltec"),
        )
        .expanduser()
        .resolve()
    )

    toltec_db = get_sqla_db("toltec")
    if toltec_db is not None:
        # TODO: maybe make this configurable instead of guessing
        data_prod_collector.proxy_init(
            QLDataProdCollector(
                db=toltec_db,
                data_lmt_rootpath=data_lmt_rootpath,
                data_prod_output_path=data_prod_output_path,
                data_prod_index_filename_prefix="dp_toltec_",
            ),
        )
        return None
    # TODO: implement offline data prod collector.
    return NotImplemented


def DASHA_SITE():
    """Return the dasha site."""
    return {
        "dasha": {
            "template": DataProdViewer,
            "THEME": dbc.themes.LUMEN,
            # "DEBUG": os.environ.get("DASH_DEBUG", False),
        },
        "db": {
            "binds": [
                {
                    "name": "toltec",
                    "url": os.environ.get("TOLTECA_WEB_TOLTEC_DB_URL", None),
                    "reflect_tables": True,
                },
                {
                    "name": "dpdb",
                    "url": os.environ.get("TOLTECA_WEB_TOLTECA_DPDB_URL", None),
                },
            ],
        },
        "post_init": _post_init,
    }
