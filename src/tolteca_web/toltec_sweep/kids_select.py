"""A VNASweep viewer."""

import json
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from dash_component_template import ComponentTemplate
from tollan.utils.fmt import pformat_yaml

from ..common import CollapseContent, LabeledChecklist, LabeledDropdown

__all__ = [
    "ObsnumNetworkArraySelect",
]


class ObsnumNetworkArraySelect(ComponentTemplate):
    """A template to select obsnum and networks."""

    class Meta:  # noqa: D106
        component_cls = html.Div

    master_select: dcc.Dropdown
    obsnum_select: dcc.Dropdown
    nw_select: dcc.Checklist
    array_select: dcc.Checklist
    selected: dcc.Store

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        container = self
        self.selected = self.child(dcc.Store)
        obsnum_select_container = container.child(dbc.Form).child(
            dbc.Row,
            className="gx-2 gy-2",
        )
        master_select = self.master_select = obsnum_select_container.child(
            LabeledDropdown(
                label_text="Select Master",
                className="mt-3 w-auto mr-3 align-items-start",
                size="sm",
            ),
        ).dropdown

        master_select.options = [
            {"label": name, "value": name.lower()} for name in ["ICS", "TCS"]
        ]

        self.obsnum_select = obsnum_select_container.child(
            LabeledDropdown(
                label_text="Select ObsNum",
                className="mt-3 w-auto mr-3 align-items-start",
                size="sm",
            ),
        ).dropdown

        subset_select_container = container.child(dbc.Form).child(
            dbc.Row,
            className="gx-2 gy-2",
        )

        self.nw_select = subset_select_container.child(
            LabeledChecklist(
                className="w-auto align-items-baseline",
                label_text="Select Network",
                checklist_props={"options": make_network_options()},
                multi=True,
            ),
        ).checklist
        self.array_select = subset_select_container.child(
            LabeledChecklist(
                className="w-auto align-items-baseline",
                label_text="Select by Array",
                checklist_props={"options": make_array_options()},
                multi=False,
            ),
        ).checklist
        self._debug_info = (
            subset_select_container.child(
                dbc.InputGroup,
                className="w-auto align-items-baseline",
            )
            .child(
                CollapseContent(
                    button_text="Details ...",
                ),
            )
            .content
        )

    def setup_layout(self, app):
        """Set up the data prod viewr layout."""
        obsnum_select = self.obsnum_select
        master_select = self.master_select
        nw_select = self.nw_select
        array_select = self.array_select
        obsnum_select_feedback = obsnum_select.parent.feedback

        super().setup_layout(app)

        @app.callback(
            [
                Output(nw_select.id, "options"),
                Output(master_select.id, "value"),
                Output(obsnum_select.id, "valid"),
                Output(obsnum_select.id, "invalid"),
                Output(obsnum_select_feedback.id, "type"),
                Output(obsnum_select_feedback.id, "children"),
            ],
            [
                Input(obsnum_select.id, "value"),
            ],
        )
        def update_network_options(obsnum_value):
            if obsnum_value is None:
                return make_network_options(enabled=set()), dash.no_update
            data_items = json.loads(obsnum_value)
            nw_enabled = {d["meta"]["roach"] for d in data_items}
            # TODO: assume master is the same...
            master = data_items[0]["meta"]["master"].lower()
            # check data item accessble
            nw_invalid = {
                d["meta"]["roach"]
                for d in data_items
                if not Path(d["filepath"]).exists()
            }
            if nw_invalid:
                fb_type = "invalid"
                fb_content = f"Files missing for networks {nw_invalid}."
            else:
                fb_type = "valid"
                fb_content = ""
            options = make_network_options(enabled=nw_enabled - nw_invalid)
            return (
                options,
                master,
                fb_type == "valid",
                fb_type == "invalid",
                fb_type,
                fb_content,
            )

        def update_network_value_for_options(network_options, network_value):
            enabled = {o["value"] for o in network_options if not o["disabled"]}
            network_value = network_value or []
            return list(set(network_value).intersection(enabled))

        @app.callback(
            Output(array_select.id, "options"),
            [
                Input(nw_select.id, "options"),
            ],
        )
        def update_array_options(network_options):
            return make_array_options(network_options=network_options)

        @app.callback(
            Output(nw_select.id, "value"),
            [
                Input(nw_select.id, "options"),
                Input(array_select.id, "value"),
            ],
        )
        def update_network_select_value_with_array(
            network_select_options,
            array_select_value,
        ):
            value = get_networks_for_array(array_select_value)
            return update_network_value_for_options(network_select_options, value)

        @app.callback(
            Output(self.selected.id, "data"),
            [
                Input(nw_select.id, "value"),
                Input(obsnum_select.id, "value"),
            ],
        )
        def update_selected_data(nw_value, obsnum_value):
            nw_value = nw_value or []
            data_items = json.loads(obsnum_value or "[]")
            data_items = [d for d in data_items if d["meta"]["roach"] in nw_value]
            data_items.sort(key=lambda d: d["meta"]["roach"])
            return data_items

        @app.callback(
            Output(self._debug_info.id, "children"),
            [
                Input(self.selected.id, "data"),
            ],
        )
        def update_debug_info(data):
            return html.Pre(pformat_yaml(data))


def make_network_options(enabled=None, disabled=None):
    """Return the options dict for select TolTEC detector networks."""
    if enabled is None:
        enabled = set(range(13))
    if disabled is None:
        disabled = set()
    if len(enabled.intersection(disabled)) > 0:
        raise ValueError("conflict in enabled and disabled kwargs.")
    result = []
    for i in range(13):
        d = {
            "label": i,
            "value": i,
            "disabled": (i not in enabled) or (i in disabled),
        }
        result.append(d)
    return result


_array_option_specs = {
    "a1100": {
        "label": "1.1mm",
        "nws": [0, 1, 2, 3, 4, 5, 6],
    },
    "a1400": {
        "label": "1.4mm",
        "nws": [7, 8, 9, 10],
    },
    "a2000": {
        "label": "2.0mm",
        "nws": [11, 12],
    },
    "ALL": {
        "label": "All",
        "nws": list(range(13)),
    },
    "NONE": {
        "label": "None",
        "nws": [],
    },
}


def make_array_options(enabled=None, disabled=None, network_options=None):
    """Return the options dict for select TolTEC arrays."""
    if enabled is None:
        enabled = set(_array_option_specs.keys())
    if disabled is None:
        disabled = set()
    if network_options is not None:
        nw_disabled = {int(n["value"]) for n in network_options if n["disabled"]}
        for k, a in _array_option_specs.items():
            if a["nws"] and set(a["nws"]).issubset(nw_disabled):
                disabled.add(k)
                enabled.discard(k)
    if len(enabled.intersection(disabled)) > 0:
        raise ValueError("conflict in enabled and disabled kwargs.")

    result = []
    for k, a in _array_option_specs.items():
        d = {
            "label": a["label"],
            "value": k,
            "disabled": (k not in enabled) and (k in disabled),
        }
        result.append(d)
    return result


def get_networks_for_array(array_select_values):
    """Return list of networks from array select."""
    if not array_select_values:
        return None
    if not isinstance(array_select_values, list):
        array_select_values = [array_select_values]
    checked = set()
    for k in array_select_values:
        checked = checked.union(set(_array_option_specs[k]["nws"]))
    return list(checked)
