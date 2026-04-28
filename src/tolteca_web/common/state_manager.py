"""Bidirectional state ↔ component synchronizer (base layer).

``state_store`` is the single source of truth, written by:

* External sources (URL parse, API, programmatic callbacks).
* Widget interactions (Step D).

``state_store`` fans out to per-field stores and bound widgets (Step B).
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from dash import Input, Output, State, dcc
from dash_component_template import Template
from pydantic import BaseModel

import dash_mantine_components as dmc

__all__ = ["ComponentStateManager"]


# ── Internal binding record ───────────────────────────────────────────────────


@dataclasses.dataclass
class _Binding:
    field: str
    node: object  # Template child node; node() called lazily in setup_callbacks
    prop: str


# ── Base state manager ───────────────────────────────────────────────────────


class ComponentStateManager(Template):
    """Bidirectional state ↔ widget synchronizer driven by a Pydantic model.

    ``state_store`` is the single source of truth.  External sources and
    widget interactions both write to it (with ``allow_duplicate=True``).
    Per-field stores let downstream callbacks react only to the fields
    they care about.

    All sync callbacks are **clientside** — zero server round-trips.

    Parameters
    ----------
    model : type[BaseModel]
        Pydantic model class declaring field names, types, and defaults.
    show_debug : bool, optional
        Show a collapsible debug panel with the current state JSON.

    Usage
    -----
    Define the state model and embed the manager::

        class PageState(BaseModel):
            dataset: str = "A"
            limit: int = 10

        mgr = ComponentStateManager(PageState)
        panel.child(mgr)

    Bind widgets (bidirectional)::

        mgr.bind("dataset", select_node)           # prop defaults to "value"
        mgr.bind("limit", number_input_node)

    Write state from an external source::

        @app.callback(
            Output(mgr.state_store(), "data", allow_duplicate=True),
            Input(some_trigger, "n_clicks"),
            State(mgr.state_store(), "data"),
            prevent_initial_call=True,
        )
        def push_state(n, cur):
            return {**cur, "dataset": "C"}

    Subscribe to individual field changes::

        Input(mgr.store("dataset"), "data")
    """

    def __init__(self, model: type[BaseModel], *, show_debug: bool = False) -> None:
        super().__init__()
        self._model_cls = model
        self._defaults: dict[str, Any] = model().model_dump()
        self._bindings: list[_Binding] = []

        self._state_store = self.child[dcc.Store](data=self._defaults.copy())

        # One dcc.Store per model field — targeted subscriptions.
        # Initialized WITHOUT data so that Step B's first run always
        # transitions them from None → default value, triggering downstream
        # server-side callbacks on initial page load.
        self._field_stores: dict[str, object] = {
            field: self.child[dcc.Store]()
            for field in self._defaults
        }

        if show_debug:
            from .collapse_content import CollapseContent

            panel = self.child(CollapseContent("State"))
            self._debug_code: object | None = panel.content.child[dmc.Code](
                block=True, children="{}"
            )
        else:
            self._debug_code = None

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def model_cls(self) -> type[BaseModel]:
        """The Pydantic model class."""
        return self._model_cls

    @property
    def defaults(self) -> dict[str, Any]:
        """Default values from the model (read-only copy)."""
        return dict(self._defaults)

    @property
    def state_store(self) -> object:
        """Central ``dcc.Store`` — the single source of truth.

        Written by external sources and widget interactions (Step D).
        Use ``allow_duplicate=True`` when targeting from custom callbacks.
        """
        return self._state_store

    def store(self, field: str) -> object:
        """Return the per-field ``dcc.Store`` node for *field*.

        Use as ``Input`` to react only when that specific field changes.

        Parameters
        ----------
        field : str
            A field name declared in the model.
        """
        if field not in self._field_stores:
            msg = f"{field!r} is not a field of {self._model_cls.__name__!r}"
            raise KeyError(msg)
        return self._field_stores[field]

    def bind(self, field: str, node: object, *, prop: str = "value") -> None:
        """Bind a component prop to a model field (bidirectional).

        * **Read** (B2): ``state_store`` change → widget prop updated.
        * **Write** (D): widget prop change → ``state_store`` patched.

        Parameters
        ----------
        field : str
            Model field name.
        node : Template child node
            The component (a ``dash_component_template`` lazy node).
        prop : str, optional
            Component property to sync, by default ``"value"``.
        """
        if field not in self._field_stores:
            msg = f"{field!r} is not a field of {self._model_cls.__name__!r}"
            raise KeyError(msg)
        self._bindings.append(_Binding(field, node, prop))

    # ── Callbacks ─────────────────────────────────────────────────────────

    def setup_callbacks(self, app) -> None:
        """Register state ↔ widget sync callbacks (all clientside).

        B.  state_store → field stores + bound component props
            Single multi-output clientside callback.
            Per-value ``===`` guards return ``no_update`` for unchanged
            outputs.

        D.  widget → state_store  (per binding)
            Patches the single changed field.  ``allow_duplicate=True``.

        E.  Debug panel (optional).

        Subclasses should call ``super().setup_callbacks(app)`` and add
        their own steps (e.g. URL sync).
        """
        self._setup_step_b(app)
        self._setup_step_d(app)
        self._setup_step_e(app)

    # ── Step B: state_store → field stores + bound components ─────────

    def _setup_step_b(self, app) -> None:
        fields = list(self._defaults.keys())
        field_to_idx = {f: i for i, f in enumerate(fields)}

        b_outputs: list[Output] = []
        b_states: list[State] = []
        for field, store_node in self._field_stores.items():
            b_outputs.append(Output(store_node(), "data"))
            b_states.append(State(store_node(), "data"))
        for b in self._bindings:
            b_outputs.append(Output(b.node(), b.prop))
            b_states.append(State(b.node(), b.prop))

        if not b_outputs:
            return

        n = len(b_outputs)
        params = ["state"] + [f"c{i}" for i in range(n)]

        # Compute each field value once
        field_lines: list[str] = []
        for i, field in enumerate(fields):
            fk_js = json.dumps(field)
            def_js = json.dumps(self._defaults[field])
            field_lines.append(
                f"var f{i}=state[{fk_js}]!==undefined"
                f"?state[{fk_js}]:{def_js};"
            )

        # Build return array (field stores then bound components)
        ret: list[str] = []
        idx = 0
        for i in range(len(fields)):
            ret.append(f"f{i}===c{idx}?nu:f{i}")
            idx += 1
        for b in self._bindings:
            fi = field_to_idx[b.field]
            ret.append(f"f{fi}===c{idx}?nu:f{fi}")
            idx += 1

        nu_arr = ",".join(["nu"] * n)
        js_b = (
            f"function({','.join(params)}){{"
            f"var nu=window.dash_clientside.no_update;"
            f"if(!state)return[{nu_arr}];"
            f"{''.join(field_lines)}"
            f"return[{','.join(ret)}];}}"
        )
        app.clientside_callback(
            js_b,
            *b_outputs,
            Input(self._state_store(), "data"),
            *b_states,
        )

    # ── Step D: widget → state_store ──────────────────────────────────

    def _setup_step_d(self, app) -> None:
        for b in self._bindings:
            field_js = json.dumps(b.field)
            app.clientside_callback(
                f"""function(val, state) {{
                    if (!state) return window.dash_clientside.no_update;
                    if (val === state[{field_js}])
                        return window.dash_clientside.no_update;
                    var s = Object.assign({{}}, state);
                    s[{field_js}] = val;
                    return s;
                }}""",
                Output(self._state_store(), "data", allow_duplicate=True),
                Input(b.node(), b.prop),
                State(self._state_store(), "data"),
                prevent_initial_call=True,
            )

    # ── Step E: Debug panel ───────────────────────────────────────────

    def _setup_step_e(self, app) -> None:
        if self._debug_code is not None:
            app.clientside_callback(
                "function(d){return d?JSON.stringify(d,null,2):'{}';}",
                Output(self._debug_code(), "children"),
                Input(self._state_store(), "data"),
            )
