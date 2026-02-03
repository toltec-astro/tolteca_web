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
import sqlalchemy as sa
from dash import Input, Output, State, dcc, html
from tollan.utils.general import ObjectProxy
from tollan.utils.log import logger, timeit

from ..base import ViewerBase
from ..common import CacheMonitorWidget, LabeledDropdown, LiveUpdateSection
from ..data_prod.collector import DataProdCollectorProtocol
from ..data_prod.tolteca_db_collector import ToltecaDBDataProdCollector
from ..data_prod.conventions import make_toltec_raw_obs_uid

# from ..toltecFocusViewer import ToltecFocusViewer
# from ..toltecObsStatsViewer import ToltecObsStatsViewer
# from ..toltecSignalFitsViewer import ToltecSignalFitsViewer
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
            # Handle empty data_items (e.g., for calibration groups)
            if not row_data:
                return {
                    "rowData": [],
                    "columnDefs": [],
                }
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

        cache_monitor = header_container.child(
            CacheMonitorWidget(
                poll_interval_ms=1000,
                className="ms-3 d-inline-block",
            ),
        )

        controls_panel, views_panel = body.grid(2, 1)
        dp_select_container = controls_panel.child(dbc.Form).child(
            dbc.Row,
            className="gx-2 gy-2",
        )

        # Data prod type filter
        # Get data product types from database
        dp_type_options = [{"label": "All", "value": ""}]
        try:
            tolteca_db_url = os.environ.get("TOLTECA_DB_URL", None)
            if tolteca_db_url:
                # Query data_prod_type table using the adapter
                from ..data_prod.tolteca_db_adapter import get_tolteca_db_adapter

                adapter = get_tolteca_db_adapter(tolteca_db_url)
                with adapter.get_session() as session:
                    # Sort by pk to maintain enum order (raw obs first)
                    result = session.execute(
                        sa.text(
                            "SELECT label FROM data_prod_type WHERE label != 'data_prod' ORDER BY pk"
                        )
                    )
                    for row in result:
                        dp_type_options.append(
                            {
                                "label": row[0],
                                "value": row[0],
                            }
                        )
        except Exception as e:
            logger.warning(f"Could not load data prod types from database: {e}")

        data_prod_type_input = dp_select_container.child(
            LabeledDropdown(
                label_text="Data Prod Type",
                size="sm",
                className="mb-2 w-auto align-items-start",
                dropdown_props={
                    "placeholder": "Select type...",
                    "options": dp_type_options,
                    "value": "",
                },
            ),
        ).dropdown

        # Master filter
        master_options = [
            {"label": "All", "value": ""},
            {"label": "TCS", "value": "tcs"},
            {"label": "ICS", "value": "ics"},
        ]
        master_input = dp_select_container.child(
            LabeledDropdown(
                label_text="Master",
                size="sm",
                className="mb-2 w-auto align-items-start",
                dropdown_props={
                    "placeholder": "Select master...",
                    "options": master_options,
                    "value": "",
                },
            ),
        ).dropdown

        # Min obsnum filter
        min_obsnum_filter = dp_select_container.child(
            dbc.InputGroup,
            size="sm",
            className="mb-2 w-auto align-items-start",
        )
        min_obsnum_filter.child(
            dbc.InputGroupText,
            "Min Obsnum",
            style={"minWidth": "120px"},
        )
        min_obsnum_input = min_obsnum_filter.child(
            dbc.Input,
            type="number",
            placeholder="Enter min obsnum...",
            style={"minWidth": "200px"},
        )

        # Max obsnum filter
        max_obsnum_filter = dp_select_container.child(
            dbc.InputGroup,
            size="sm",
            className="mb-2 w-auto align-items-start",
        )
        max_obsnum_filter.child(
            dbc.InputGroupText,
            "Max Obsnum",
            style={"minWidth": "120px"},
        )
        max_obsnum_input = max_obsnum_filter.child(
            dbc.Input,
            type="number",
            placeholder="Enter max obsnum...",
            style={"minWidth": "200px"},
        )

        # Observation date filter (dropdown with available dates)
        # Get unique dates from database (use obs_datetime if available, fallback to created_at)
        date_options = [{"label": "All", "value": ""}]
        try:
            if tolteca_db_url:
                with adapter.get_session() as session:
                    result = session.execute(
                        sa.text("""
                            SELECT DISTINCT 
                                COALESCE(
                                    DATE(json_extract(meta, '$.obs_datetime')),
                                    DATE(created_at)
                                ) as date 
                            FROM data_prod 
                            WHERE date IS NOT NULL
                            ORDER BY date DESC
                        """)
                    )
                    for row in result:
                        if row[0]:
                            date_options.append(
                                {
                                    "label": row[0],
                                    "value": row[0],
                                }
                            )
        except Exception as e:
            logger.warning(f"Could not load dates from database: {e}")

        obs_date_input = dp_select_container.child(
            LabeledDropdown(
                label_text="Obs Date",
                size="sm",
                className="mb-2 w-auto align-items-start",
                dropdown_props={
                    "placeholder": "Select date...",
                    "options": date_options,
                    "value": "",
                },
            ),
        ).dropdown

        # pull down to select data prod.
        dp_select = dp_select_container.child(
            LabeledDropdown(
                label_text="Data Prod",
                size="sm",
                className="mb-2 w-auto align-items-start",
                dropdown_props={
                    "placeholder": "Select data prod...",
                    "value": "",
                },
            ),
        ).dropdown

        # data prod assoc select
        dpa_select = dp_select_container.child(
            LabeledDropdown(
                label_text="Assoc. Data Prod",
                size="sm",
                className="mb-2 w-auto align-items-start",
                dropdown_props={
                    "placeholder": "Select a data product...",
                    "value": "",
                },
            ),
        ).dropdown

        dpa_as_dp_btn = dp_select_container.child(
            dbc.InputGroup,
            className="mb-2 w-auto align-items-start",
        ).child(
            dbc.Button,
            "Set current Assoc. DP as DP",
            size="sm",
            id="dpa-as-dp-btn",
        )

        # LMT Shift Report link for current obsnum
        lmtmc_link_container = dp_select_container.child(
            dbc.InputGroup,
            className="mb-2 w-auto align-items-center",
        )
        lmtmc_link = lmtmc_link_container.child(
            html.A,
            [
                html.I(className="fas fa-external-link-alt me-1"),
                "LMT Shift Report",
            ],
            href="#",
            target="_blank",
            className="btn btn-sm btn-outline-info",
            style={"display": "none"},
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

        # Cache monitor callback to poll file resolver status
        @app.callback(
            Output(cache_monitor.status_store.id, "data"),
            Input(cache_monitor.interval.id, "n_intervals"),
        )
        def update_cache_monitor(_n):
            """Poll file resolver for download status."""
            from ..data_prod.file_resolver import get_file_resolver

            try:
                resolver = get_file_resolver()
                return resolver.get_download_status()
            except Exception:
                return {"active_downloads": {}, "cache_stats": {}}

        @app.callback(
            [
                Output(dp_select.id, "options"),
                Output(dp_select.id, "valid"),
                Output(dp_select.id, "invalid"),
                Output(dp_select_feedback.id, "type"),
                Output(dp_select_feedback.id, "children"),
                Output(header.loading.id, "children"),
                Output(obs_date_input.id, "options"),
            ],
            [
                Input(header.timer.n_calls_store.id, "data"),
                Input(data_prod_type_input.id, "value"),
                Input(master_input.id, "value"),
                Input(min_obsnum_input.id, "value"),
                Input(max_obsnum_input.id, "value"),
                Input(obs_date_input.id, "value"),
            ],
        )
        def update_dp_select(
            _n_calls,
            data_prod_type_filter,
            master_filter,
            min_obsnum_filter,
            max_obsnum_filter,
            obs_date_filter,
        ):
            # Pass filters to collector for database-level filtering
            dps, collector_info = collect_data_prods(
                data_prod_type=data_prod_type_filter if data_prod_type_filter else None,
                master=master_filter if master_filter else None,
                min_obsnum=min_obsnum_filter,
                max_obsnum=max_obsnum_filter,
                obs_date=obs_date_filter if obs_date_filter else None,
            )

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
            fb_content = (
                f"{collector_info.message or ''} (Showing {len(dps)} data products)"
            )

            # Update date options dynamically
            date_options = [{"label": "All", "value": ""}]
            try:
                if tolteca_db_url:
                    with adapter.get_session() as session:
                        result = session.execute(
                            sa.text("""
                                SELECT DISTINCT 
                                    COALESCE(
                                        DATE(json_extract(meta, '$.obs_datetime')),
                                        DATE(created_at)
                                    ) as date 
                                FROM data_prod 
                                WHERE date IS NOT NULL
                                ORDER BY date DESC
                            """)
                        )
                        for row in result:
                            if row[0]:
                                date_options.append(
                                    {
                                        "label": row[0],
                                        "value": row[0],
                                    }
                                )
            except Exception as e:
                logger.warning(f"Could not load dates from database: {e}")

            return (
                options,
                fb_type == "valid",
                fb_type == "invalid",
                fb_type,
                fb_content,
                "",
                date_options,
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
                dpa_filepath = dpa["filepath"]

                # Handle both legacy file paths and new tolteca_db:// URIs
                if dpa_filepath.startswith("tolteca_db://"):
                    # Keep the full tolteca_db:// URI as filename
                    # The collector/store knows how to resolve these URIs
                    dpa_filename = dpa_filepath
                else:
                    # Legacy file path resolution
                    dpa_path = _resolve_path(
                        Path(dpa_filepath),
                        Path(dp.index_filepath).parent,
                    )
                    dpa_filename = dpa_path.name

                # validate - try to load the associated data product
                try:
                    dpa_dp = load_data_prod(dpa_filename)
                    options.append(
                        {
                            "label": dpa_dp.make_display_label(prefix=f"{dpa_type} - "),
                            "value": dpa_dp.index_filename,
                            "disabled": False,
                        },
                    )
                except Exception:  # noqa: BLE001
                    # If loading fails, add a disabled entry with the raw filename
                    options.append(
                        {
                            "label": f"{dpa_type} - {dpa_filename} (error loading)",
                            "value": dpa_filename,
                            "disabled": True,
                        },
                    )
            return options, options[0]["value"]

        # LMTMC API base URL
        LMTMC_API_BASE_URL = "http://187.248.54.232/cgi-bin/lmtmc/mc_sql.cgi"

        @app.callback(
            [
                Output(lmtmc_link.id, "href"),
                Output(lmtmc_link.id, "style"),
                Output(lmtmc_link.id, "children"),
            ],
            [
                Input(dp_select.id, "value"),
            ],
            prevent_initial_call=True,
        )
        def update_lmtmc_link(index_filename):
            """Update LMT Shift Report link based on selected data product."""
            icon = html.I(className="fas fa-external-link-alt me-1")
            label = "LMT Shift Report"
            hidden = [icon, label]
            if not index_filename:
                return "#", {"display": "none"}, hidden
            try:
                dp = load_data_prod(index_filename)
                obsnum = dp.index.get("meta", {}).get("obsnum")
                if obsnum:
                    url = f"{LMTMC_API_BASE_URL}?-obsNum={obsnum}&-format=html"
                    return url, {"display": "inline-block"}, [icon, f"{label} ({obsnum})"]
            except Exception:  # noqa: BLE001
                pass
            return "#", {"display": "none"}, hidden

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
    if p is None:
        return None
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
            # Filter out None values from roach set (for interfaces without valid data)
            nw = {
                d["meta"]["roach"]
                for d in self.index["data_items"]
                if "roach" in d["meta"] and d["meta"]["roach"] is not None
            }
            # Show {} for empty set (tel-only), otherwise show the set
            nw_str = "{}" if not nw else str(nw)
            return f"{prefix}{self.name} - {dk}{nw_str}"
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
            resolved_path = self._resolve_path(d["filepath"])
            d["filepath"] = (
                resolved_path.as_posix() if resolved_path is not None else None
            )
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
            options = [
                {"label": p["filepath"], "value": p["filepath"]} for p in tel_files
            ]
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
            from tolteca_web.data_prod.file_resolver import is_file_resolvable
            files_by_obsnum = {}
            for d in raw_kids_items:
                obsnum = d["meta"]["obsnum"]
                # skip files that cannot be resolved (local or remote)
                if not is_file_resolvable(d["filepath"]):
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
def collect_data_prods(
    data_prod_type=None,
    master=None,
    min_obsnum=None,
    max_obsnum=None,
    obs_date=None,
):
    """Collect data products with optional filters.

    Parameters
    ----------
    data_prod_type : str, optional
        Type of data product to filter by (e.g., 'dp_raw_obs')
    master : int, optional
        Master flag to filter by (0 or 1)
    min_obsnum : int, optional
        Minimum observation number to include
    max_obsnum : int, optional
        Maximum observation number to include
    obs_date : str, optional
        Observation date to filter by (YYYY-MM-DD format)
    """
    dpc = data_prod_collector
    store = dpc.data_prod_index_store
    with collect_data_prods_lock:
        # Pass filters to collector for database-level filtering
        info = dpc.collect(
            n_items=100,
            n_updates=2,
            data_prod_type=data_prod_type if data_prod_type else None,
            master=master,
            min_obsnum=min_obsnum,
            max_obsnum=max_obsnum,
            obs_date=obs_date,
        )
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

    # Get the tolteca_db URL for new collector
    tolteca_db_url = os.environ.get("TOLTECA_DB_URL", None)

    if tolteca_db_url is not None:
        # Use new tolteca_db collector that queries DataProd table
        data_prod_collector.proxy_init(
            ToltecaDBDataProdCollector(
                db_url=tolteca_db_url,
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
            "tolteca_db_url": os.environ.get(
                "TOLTECA_DB_URL", None
            ),  # For ObsQuery/ToltecaDBAdapter
        },
        "post_init": _post_init,
    }
