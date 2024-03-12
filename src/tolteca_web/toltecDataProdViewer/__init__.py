"""The TolTEC data product viewer."""

import os
import json

import dash_ag_grid as dag

import dash_bootstrap_components as dbc
from dash import html
from tollan.utils.log import logger
from dash import Input, Output, State
import cachetools.func
from pathlib import Path
from dataclasses import dataclass, field

from ..common import LabeledDropdown, LiveUpdateSection
from tollan.utils.yaml import yaml_load
from tollan.utils.log import timeit
from tollan.utils.fmt import pformat_yaml

from ..toltecTonePowerViewer import ToltecTonePowerViewer

# from ..toltecTelViewer import ToltecTelViewer
# from ..toltecAptViewer import ToltecAptViewer
# from ..toltecFocusViewer import ToltecFocusViewer
# from ..toltecObsStatsViewer import ToltecObsStatsViewer
# from ..toltecSignalFitsViewer import ToltecSignalFitsViewer
from ..db import get_sqla_db
from tollan.utils.general import ObjectProxy
from multiprocessing import Lock
from ..base import ViewerBase
from ..data_prod.collector import QLDataProdCollector, DataProdCollectorProtocol


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
                interval_options=[2000, 5000, 10000],
                interval_option_value=2000,
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
                className="mb-2 w-auto",
            ),
        ).dropdown

        dpa_select = dp_select_container.child(
            LabeledDropdown(
                label_text="Assoc. Data Prod",
                # className='w-auto',
                size="sm",
                placeholder="Select a data product ...",
                className="mb-2 w-auto",
            ),
        ).dropdown

        dpa_as_dp_btn = dp_select_container.child(
            dbc.Button,
            "Set current Assoc. DP as DP",
            size="sm",
            className="mb-2 w-auto",
        )
        dp_select_container.child(dbc.FormFeedback, type="invalid")
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
            #   {"label": "Detector", "value": "detector"},
            #   {"label": "Apt", "value": "apt", "template": ToltecAptViewer()},
            #   {"label": "Tel", "value": "tel", "template": ToltecTelViewer()},
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
                Output(header.loading.id, "children"),
            ],
            header.timer.inputs,
        )
        def update_dp_select(
            _n_calls,
        ):
            dps = collect_data_prods()
            # dps = get_latest_data_prods_from_dpdb()
            options = []
            for dp in dps:
                dp_type = dp.index["meta"]["data_prod_type"]
                index_filepath = dp.index_filepath
                options.append(
                    {
                        "label": f"{dp_type} - {index_filepath.stem}",
                        "value": index_filepath.as_posix(),
                    },
                )
            # value = options[-1]["value"] if len(options) > 0 else dash.no_update
            # value = dash.no_update
            return options, ""

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
        def update_dpa_select(index_filepath):
            dp = load_data_prod(Path(index_filepath))
            dt = dp.index["meta"]["data_prod_type"]
            assocs = dp.index.get("assocs", [])
            options = [
                {
                    "label": f"self - {dt} - {dp.name}",
                    "value": dp.index_filepath.as_posix(),
                },
            ]
            for dpa in assocs:
                dpa_type = dpa["data_prod_assoc_type"]
                dpa_path = _resolve_path(
                    Path(dpa["filepath"]), dp.index_filepath.parent
                )
                # validate
                try:
                    dpa_dp = load_data_prod(dpa_path)
                except Exception:  # noqa: BLE001
                    valid = False
                    dt = "(unknown)"
                else:
                    valid = True
                    dt = dpa_dp.index["meta"]["data_prod_type"]
                options.append(
                    {
                        "label": f"{dpa_type} - {dt} - {dpa_path.stem}",
                        "value": dpa_path.as_posix(),
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
        def update_dp_select_value_from_dpa_value(_n_clicks, index_filepath):
            return index_filepath

        @app.callback(
            Output(dpa_as_dp_btn.id, "disabled"),
            header.timer.inputs
            + [
                Input(dp_select.id, "value"),
                Input(dpa_select.id, "value"),
            ],
            prevent_initial_call=True,
        )
        def check_dp_self(_n_times, dp_index_filepath, dpa_index_filepath):
            # print(f"check {dp_index_filepath} {dpa_index_filepath}")
            return dp_index_filepath == dpa_index_filepath

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
            def update_viewer_input_data(index_filepath, tab_label):
                dp = load_data_prod(Path(index_filepath))
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

    name: str
    index_filepath: str
    index: None | dict = field(repr=False)

    def _resolve_path(self, p):
        return _resolve_path(p, self.index_filepath.parent)

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
        return {
            "aptList": self._data_items_by_data_kind.get("ArrayPropTable", None),
        }

    @staticmethod
    def get_apt_viewr_mapper_funcs():
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
        return self._data_items_by_data_kind.get("LmtTel", None)

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
            value = options[0]["value"]
            return {
                "options": options,
                "value": value,
            }

        return {
            "obsnumList": _map_obsnum_list_data,
        }


@cachetools.func.ttl_cache(maxsize=256, ttl=5)
def load_data_prod(index_filepath):
    """Return data prod."""
    name = index_filepath.stem
    index = yaml_load(index_filepath)
    dp = DataProd(name=name, index_filepath=index_filepath, index=index)
    logger.debug(f"loaded data prod: {dp}")
    return dp


@timeit("collect_data_prods", level="INFO")
@cachetools.func.ttl_cache(maxsize=1, ttl=5)
def collect_data_prods():
    """Return the list of data prods."""
    dp_dir = data_prod_collector.collect(n_items=20)
    dps = []
    for p in data_prod_collector.get_collected():
        dp = load_data_prod(p)
        dps.append(dp)
    logger.debug(f"collected {len(dps)} data prods from {dp_dir}")
    return sorted(dps, key=lambda dp: dp.name, reverse=True)


data_prod_collector: ObjectProxy | DataProdCollectorProtocol = ObjectProxy()
data_prod_output_lock = Lock()


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
                data_prod_output_lock=data_prod_output_lock,
            ),
        )
        return None
    return NotImplemented


def DASHA_SITE():
    """Return the dasha site."""
    return {
        "dasha": {
            "template": DataProdViewer,
            # "THEME": dbc.themes.LUMEN,
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
