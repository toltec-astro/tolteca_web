#! /usr/bin/env python


import copy

import dash_bootstrap_components as dbc
from astropy.utils.console import human_time
from dash import Input, Output, State, dcc, html
from dash_component_template import ComponentTemplate

from .collapsecontent import CollapseContent


class IntervalTimer(ComponentTemplate):
    class Meta:
        component_cls = html.Div

    _INTERVAL_PAUSE = -1
    min_interval = 500

    def __init__(
        self, *args, interval_options=None, interval_option_value=None, **kwargs
    ):
        # the intervals are in milliseconds.
        super().__init__(*args, **kwargs)

        if len(interval_options) == 0:
            interval_options = [self.min_interval]
        if interval_option_value is None:  # default
            interval_option_value = interval_options[0]
        if interval_option_value not in interval_options:
            raise ValueError("invalid interval option value")
        # if self.min_interval < 100:
        #     raise ValueError(
        #             'min interval should not be less than 500 ms.'
        #             )
        if self.min_interval > min(interval_options):
            raise ValueError(
                f"interval options cannot be less than "
                f"min interval {self.min_interval}"
            )
        for v in interval_options:
            if v % self.min_interval != 0:
                raise ValueError("interval options has to be multiples of min interval")
        self.interval_option_value = interval_option_value
        self.interval_options = copy.copy(interval_options)
        # add pause option
        self.interval_options.append(self._INTERVAL_PAUSE)

        self._timer = self.child(dcc.Interval, interval=self.min_interval)
        self._n_calls_store = self.child(dcc.Store, data=0)

    def setup_layout(self, app):
        container = self

        def make_interval_label(v):
            if v == self._INTERVAL_PAUSE:
                return "∞"
            if v >= 1000:
                return human_time(v / 1000)
            return f"{v / 1000.:.1f}s"

        controls_container = container
        button_icon_id = f"{controls_container.id}-button_icon0"
        button_icon = html.I(className="fas fa-hourglass-start", id=button_icon_id)

        controls_form_collapse = controls_container.child(
            CollapseContent(
                # button_text=fa('fas fa-cog')
                button_text=button_icon,
                className="d-flex",
            )
        )
        controls_form_container = controls_form_collapse.content
        # controls_form_collapse._button.style = {
        #         'color': '#555'
        #         }
        (
            controls_form_container,
            interval_progress_container,
        ) = controls_form_collapse.content.grid(2, 1)
        controls_form = controls_form_container.child(dbc.Form)
        # controls_form = controls_form_container
        # controls_form.className = 'd-flex'
        # controls_form_container.parent = controls_container
        interval_select_container = controls_form.child(dbc.Row)
        # interval_select_container, interval_progress_container = \
        #     controls_form.grid(2, 1)
        interval_progress = interval_progress_container.child(
            dbc.Progress,
            style={
                "height": "0.15em",
                "background-color": "rgba(0, 0, 0, 0)",
            },
            # className="mb-2",
            bar_style={
                "transition-duration": f"{self.min_interval * 1.1}ms",
            },
        )
        interval_select = interval_select_container.child(
            dbc.RadioItems,
            options=[
                {
                    "label": make_interval_label(v),
                    "value": v,
                }
                for v in self.interval_options
            ],
            value=self.interval_option_value,
            # inline=True,
            persistence=True,
            labelClassName=(
                "bs4-compat btn btn-sm btn-light " "form-check-label rounded-0 py-0"
            ),
            labelCheckedClassName="active",
            labelStyle={"height": "1.5em", "margin-top": "0.5em"},
            custom=False,
            inputClassName="d-none",
            className="d-flex form-check-compact",
        )

        super().setup_layout(app)

        app.clientside_callback(
            """
            function(
                    n, interval_option_value, min_interval
                    ) {
                if (interval_option_value <= 0) {
                    return window.dash_clientside.no_update
                }
                r = (n * min_interval) % interval_option_value
                // console.log(
                //     "progress:", n,
                //     interval_option_value, min_interval, r,
                //     100 * r / interval_option_value)
                return 100 * r / interval_option_value
                }
            """,
            Output(interval_progress.id, "value"),
            [
                Input(self._timer.id, "n_intervals"),
                Input(interval_select.id, "value"),
                Input(self._timer.id, "interval"),
            ],
            prevent_initial_call=True,
        )

        app.clientside_callback(
            """
            function(
                    n, interval_option_value, min_interval, n_calls
                    ) {
                // console.log(
                //    n, min_interval, interval_option_value)
                if (interval_option_value <= 0) {
                    return window.dash_clientside.no_update
                }
                if ((n * min_interval) % interval_option_value !== 0) {
                    // console.log(
                    // "no update", n, min_interval, interval_option_value)
                    return window.dash_clientside.no_update
                }
                // console.log(
                //     "n_calls: ", n_calls, "->", n_calls + 1
                //     )
                return n_calls + 1
                }
            """,
            Output(self._n_calls_store.id, "data"),
            [
                Input(self._timer.id, "n_intervals"),
                Input(interval_select.id, "value"),
                Input(self._timer.id, "interval"),
            ],
            [State(self._n_calls_store.id, "data")],
            prevent_initial_call=True,
        )

        app.clientside_callback(
            """
            function(n_calls) {
                if (n_calls % 2 === 0) {
                    return "fas fa-hourglass-start"
                }
                return "fas fa-hourglass-end"
                }
            """,
            Output(button_icon.id, "className"),
            [
                Input(self._n_calls_store.id, "data"),
            ],
            prevent_initial_call=True,
        )

    def register_callback(interval, outputs, inputs, states, callback):
        """This will set up the exec of the :"""
        pass

    @property
    def inputs(self):
        return [Input(self._n_calls_store.id, "data")]
