"""URL params ↔ component-store synchronizer.

Extends :class:`~.state_manager.ComponentStateManager` with:

* **Step A**: URL → ``state_store`` (server-side Pydantic validation)
* **Step C**: ``state_store`` → URL (clientside)
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs

from dash import Input, Output, State, dcc, html, no_update
from pydantic import BaseModel, ValidationError

import dash_mantine_components as dmc

from .state_manager import ComponentStateManager

__all__ = ["UrlStateManager"]


class UrlStateManager(ComponentStateManager):
    """URL params ↔ component stores.

    Inherits bidirectional state ↔ widget sync from
    :class:`ComponentStateManager` and adds:

    * **URL → state** (Step A): server-side Pydantic validation of URL
      params into ``state_store``.  Invalid params fall back to defaults;
      error messages are shown in an alert.
    * **state → URL** (Step C): clientside encoding of ``state_store``
      to individual URL params.

    Parameters
    ----------
    model : type[BaseModel]
        Pydantic model class declaring field names, types, and defaults.
    show_debug : bool, optional
        Show a collapsible debug panel with the current state JSON.

    Usage
    -----
    ::

        class PageState(BaseModel):
            dataset: str = "A"
            limit: int = 10
            active: bool = False

        mgr = UrlStateManager(PageState, show_debug=True)
        panel.child(mgr)

        mgr.bind("dataset", select_node)
        mgr.bind("limit", number_input_node)
        mgr.bind("active", switch_node, prop="checked")

    Navigate to ``/?dataset=C&limit=50`` — widgets update instantly.
    Change a widget — URL updates instantly.
    """

    def __init__(self, model: type[BaseModel], *, show_debug: bool = False) -> None:
        super().__init__(model, show_debug=show_debug)

        self._location = self.child[dcc.Location](refresh=False)
        self._errors_store = self.child[dcc.Store](data="")

        # Validation alert — wrapper div controls visibility
        self._alert_wrapper = self.child[html.Div](
            style={"display": "none"},
        )
        self._alert = self._alert_wrapper.child[dmc.Alert](
            children="",
            title="Invalid URL parameters",
            color="red",
            variant="light",
            withCloseButton=True,
            mb="sm",
        )

    # ── Public API (URL-specific) ─────────────────────────────────────────

    @property
    def location(self) -> object:
        """The ``dcc.Location`` node."""
        return self._location

    @property
    def errors_store(self) -> object:
        """``dcc.Store`` holding the Pydantic validation error string (or empty)."""
        return self._errors_store

    @property
    def alert(self) -> object:
        """``dmc.Alert`` node shown when URL params fail validation."""
        return self._alert

    # ── Callbacks ─────────────────────────────────────────────────────────

    def setup_callbacks(self, app) -> None:
        """Register all URL-sync callbacks.

        Architecture:

        A.   URL → state_store + errors_store  (**server-side**)
        A′.  errors_store → alert  (clientside)
        B.   state_store → field stores + bound components  (clientside, via super)
        D.   widget → state_store  (clientside, via super)
        C.   state_store → URL  (clientside)
        E.   Debug panel  (clientside, via super)
        """
        self._setup_step_a(app)
        self._setup_step_a_prime(app)
        super().setup_callbacks(app)  # Steps B, D, E
        self._setup_step_c(app)

    # ── Step A: URL → state_store (server-side Pydantic validation) ───

    def _setup_step_a(self, app) -> None:
        model_cls = self._model_cls
        defaults = self._defaults

        @app.callback(
            Output(self._state_store(), "data", allow_duplicate=True),
            Output(self._errors_store(), "data"),
            Input(self._location(), "search"),
            State(self._state_store(), "data"),
            prevent_initial_call="initial_duplicate",
        )
        def _url_to_state(
            search: str | None, cur_state: dict[str, Any] | None
        ) -> tuple[Any, ...]:
            raw_qs = parse_qs(
                (search or "").lstrip("?"), keep_blank_values=False
            )
            # Flatten: parse_qs returns lists; take first value per key
            raw: dict[str, str] = {k: v[0] for k, v in raw_qs.items()}

            # Pre-coerce: try JSON-parse equivalent for bools/numbers.
            # For fields whose default is a str, keep the raw string to avoid
            # json.loads("18595") returning int 18595 for a str field.
            coerced: dict[str, Any] = {}
            for k, v in raw.items():
                if k not in defaults:
                    continue
                if isinstance(defaults[k], str):
                    coerced[k] = v  # keep as string; Pydantic handles validation
                else:
                    try:
                        coerced[k] = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        coerced[k] = v

            # Validate through Pydantic
            merged = {**defaults, **coerced}
            error_fields: set[str] = set()
            error_message = ""
            try:
                validated = model_cls(**merged).model_dump()
            except ValidationError as exc:
                validated = dict(defaults)
                errs = exc.errors(include_url=False)
                lines = [f"{exc.error_count()} validation error(s) for {model_cls.__name__}"]
                for err in errs:
                    loc = ".".join(str(l) for l in err["loc"])
                    field_name = err["loc"][0] if err["loc"] else "unknown"
                    error_fields.add(str(field_name))
                    lines.append(f"{loc}\n  {err['msg']} [type={err['type']}]")
                error_message = "\n".join(lines)
                # Keep individually-valid fields
                for k, v in merged.items():
                    if k not in error_fields and k in defaults:
                        try:
                            model_cls(**{**defaults, k: v})
                            validated[k] = v
                        except ValidationError:
                            pass

            # Deep-compare guard: break A↔C loop
            if validated == cur_state and not error_message:
                return no_update, no_update

            return validated, error_message

    # ── Step A′: errors_store → alert ─────────────────────────────────

    def _setup_step_a_prime(self, app) -> None:
        app.clientside_callback(
            """function(msg) {
                if (!msg) return [{display: 'none'}, ''];
                return [
                    {display: 'block'},
                    {
                        props: {style: {whiteSpace: 'pre-wrap', margin: 0}, children: msg},
                        type: 'Pre',
                        namespace: 'dash_html_components',
                    },
                ];
            }""",
            Output(self._alert_wrapper(), "style"),
            Output(self._alert(), "children"),
            Input(self._errors_store(), "data"),
        )

    # ── Step C: state_store → URL ─────────────────────────────────────

    def _setup_step_c(self, app) -> None:
        field_keys = list(self._defaults.keys())
        enc_lines: list[str] = []
        for fk in field_keys:
            fk_js = json.dumps(fk)
            def_js = json.dumps(self._defaults[fk])
            enc_lines.append(
                f"var _k={fk_js},_fv=state[_k],"
                f"_sv=typeof _fv==='string'?_fv:JSON.stringify(_fv);"
                f"if(qs.has(_k)||JSON.stringify(_fv)!==JSON.stringify({def_js}))"
                f"qs.set(_k,_sv);"
            )
        app.clientside_callback(
            f"""function(state, errors, cur) {{
                if (!state) return window.dash_clientside.no_update;
                if (errors) return window.dash_clientside.no_update;
                var qs = new URLSearchParams((cur || '').replace(/^\\?/, ''));
                {''.join(enc_lines)}
                var s = qs.toString() ? '?' + qs.toString() : '';
                // Normalize cur for encoding-safe comparison (%2C vs , etc.)
                var curNorm = new URLSearchParams((cur || '').replace(/^\\?/, '')).toString();
                curNorm = curNorm ? '?' + curNorm : '';
                if (s === curNorm) return window.dash_clientside.no_update;
                // Use replaceState so URL reflects state without adding a
                // browser history entry (only user-initiated navigation should
                // push entries; widget-driven URL sync should not).
                window.history.replaceState(null, '', window.location.pathname + s);
                return window.dash_clientside.no_update;
            }}""",
            Output(self._location(), "search"),
            Input(self._state_store(), "data"),
            State(self._errors_store(), "data"),
            State(self._location(), "search"),
            prevent_initial_call=True,
        )



