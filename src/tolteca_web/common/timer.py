"""Live-update interval timer template."""

from __future__ import annotations

from dash import Input, Output, State, dcc
from dash_component_template import Template
from dash_iconify import DashIconify

import dash_mantine_components as dmc

__all__ = ["IntervalTimer"]

_MIN_INTERVAL_MS = 500


def _fmt_interval(v: int) -> str:
    """Format interval milliseconds as a human-readable label."""
    if v < 0:
        return "∞"
    if v >= 60_000:
        return f"{v // 60_000}m"
    if v >= 1_000:
        return f"{v / 1_000:.0f}s"
    return f"{v}ms"


class IntervalTimer(Template):
    """A configurable live-update timer with pause support.

    The underlying `dcc.Interval` runs at `min_interval` (500 ms). A
    clientside callback only increments `n_calls_store` when the elapsed
    time is a multiple of the selected interval, so callers can fire at
    coarser rates without multiple server callbacks.

    A thin `dmc.Progress` bar counts down to the next tick, and a
    `dmc.SegmentedControl` lets the user choose the update rate.

    Parameters
    ----------
    interval_options : list[int]
        Allowed intervals in milliseconds. Each must be a multiple of
        `min_interval` (500 ms).
    interval_option_value : int, optional
        Initial interval; defaults to the first option.

    Attributes
    ----------
    n_calls_store : dcc.Store node
        Use ``Input(timer.n_calls_store(), "data")`` in your callback.

    Examples
    --------
    >>> timer = IntervalTimer(interval_options=[5_000, 30_000, 60_000])
    """

    _PAUSE = -1

    def __init__(
        self,
        interval_options: list[int] | None = None,
        interval_option_value: int | None = None,
    ) -> None:
        super().__init__()
        options = list(interval_options or [5_000, 30_000, 60_000])
        if not options:
            options = [_MIN_INTERVAL_MS]
        if interval_option_value is None:
            interval_option_value = options[0]
        if interval_option_value not in options:
            msg = f"interval_option_value {interval_option_value} not in options"
            raise ValueError(msg)
        if _MIN_INTERVAL_MS > min(options):
            msg = f"Options cannot be less than min_interval {_MIN_INTERVAL_MS} ms"
            raise ValueError(msg)
        for v in options:
            if v % _MIN_INTERVAL_MS != 0:
                msg = f"Option {v} is not a multiple of min_interval {_MIN_INTERVAL_MS}"
                raise ValueError(msg)

        self._interval_options = options + [self._PAUSE]
        self._interval_option_value = interval_option_value

        # Internal dcc components
        self._interval = self.child[dcc.Interval](interval=_MIN_INTERVAL_MS)
        self.n_calls_store = self.child[dcc.Store](data=0)

        # Build visible controls
        icon_container = self.child[dmc.Group](gap="xs", wrap="nowrap")

        self._icon = icon_container.child[dmc.ActionIcon](
            variant="subtle",
            size="sm",
        )
        self._icon_glyph = self._icon.child[DashIconify](
            icon="mdi:timer-sand", width=18
        )

        self._controls = icon_container.child[dmc.Collapse](opened=False)
        controls_stack = self._controls.child[dmc.Stack](gap=2)

        self._interval_select = controls_stack.child[dmc.SegmentedControl](
            data=[
                {"label": _fmt_interval(v), "value": str(v)}
                for v in self._interval_options
            ],
            value=str(self._interval_option_value),
            size="xs",
            persistence=True,
        )
        self._progress = controls_stack.child[dmc.Progress](
            value=0,
            size=3,
            style={"transition": f"width {_MIN_INTERVAL_MS * 1.1}ms linear"},
        )

    def setup_callbacks(self, app) -> None:
        """Register clientside callbacks for progress bar, n_calls, and icon."""
        # Toggle controls visibility
        app.clientside_callback(
            "function(n, o) { return n ? !o : o; }",
            Output(self._controls(), "opened"),
            Input(self._icon(), "n_clicks"),
            State(self._controls(), "opened"),
        )

        # Progress bar countdown
        app.clientside_callback(
            """
            function(n, sel_value, min_iv) {
                var iv = parseInt(sel_value);
                if (iv <= 0 || n <= 0) return 0;
                return 100 * ((n * min_iv) % iv) / iv;
            }
            """,
            Output(self._progress(), "value"),
            Input(self._interval(), "n_intervals"),
            Input(self._interval_select(), "value"),
            Input(self._interval(), "interval"),
            prevent_initial_call=True,
        )

        # Increment logical tick counter
        app.clientside_callback(
            """
            function(n, sel_value, min_iv, n_calls) {
                var iv = parseInt(sel_value);
                if (iv <= 0 || n <= 0) return window.dash_clientside.no_update;
                if ((n * min_iv) % iv !== 0) return window.dash_clientside.no_update;
                return (n_calls || 0) + 1;
            }
            """,
            Output(self.n_calls_store(), "data"),
            Input(self._interval(), "n_intervals"),
            Input(self._interval_select(), "value"),
            Input(self._interval(), "interval"),
            State(self.n_calls_store(), "data"),
            prevent_initial_call=True,
        )

        # Alternate icon on each logical tick
        app.clientside_callback(
            """
            function(n_calls) {
                return n_calls % 2 === 0 ? "mdi:timer-sand" : "mdi:timer-sand-complete";
            }
            """,
            Output(self._icon_glyph(), "icon"),
            Input(self.n_calls_store(), "data"),
            prevent_initial_call=True,
        )
