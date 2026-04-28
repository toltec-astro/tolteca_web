"""Cache/download progress monitor template."""

from __future__ import annotations

from dash import Input, Output, dcc
from dash_component_template import Template

import dash_mantine_components as dmc

__all__ = ["CacheMonitor"]


class CacheMonitor(Template):
    """Displays active download progress bars from a status store.

    The caller writes a dict (or list of dicts) to `status_store.data`
    with the schema::

        {
          "active": [
            {
              "filename": str,
              "progress": float,   # 0–100
              "speed_bps": float,  # bytes/s (optional)
              "eta_s": float,      # seconds remaining (optional)
            },
            ...
          ],
          "stats": str,            # optional free-text stats line
        }

    Attributes
    ----------
    status_store : dcc.Store node — write progress data here

    Parameters
    ----------
    poll_interval_ms : int, optional
        How often to poll (not used internally; caller drives via Interval).
        Default 500 ms.

    Examples
    --------
    >>> monitor = CacheMonitor()
    """

    def __init__(self, poll_interval_ms: int = 500) -> None:
        super().__init__()
        self.status_store = self.child[dcc.Store](data={})
        self._container = self.child[dmc.Stack](gap="xs")
        self._downloads_col = self._container.child[dmc.Stack](gap=4)
        self._stats_text = self._container.child[dmc.Text](
            size="xs", c="dimmed", children="Idle"
        )

    def setup_callbacks(self, app) -> None:
        """Update progress bars when status_store changes."""

        @app.callback(
            Output(self._downloads_col(), "children"),
            Output(self._stats_text(), "children"),
            Input(self.status_store(), "data"),
        )
        def _update(data: dict | None) -> tuple:
            if not data:
                return [], "Idle"

            active = data.get("active", [])
            stats = data.get("stats", "")
            children = []
            for item in active:
                prog = float(item.get("progress", 0))
                speed = item.get("speed_bps")
                eta = item.get("eta_s")
                label = item.get("filename", "?")

                # Format speed
                speed_str = ""
                if speed is not None:
                    if speed >= 1_048_576:
                        speed_str = f" {speed / 1_048_576:.1f} MB/s"
                    elif speed >= 1_024:
                        speed_str = f" {speed / 1_024:.1f} KB/s"
                    else:
                        speed_str = f" {speed:.0f} B/s"
                eta_str = f" ETA {eta:.0f}s" if eta is not None else ""

                children.append(
                    dmc.Stack(
                        gap=2,
                        children=[
                            dmc.Text(
                                f"{label}{speed_str}{eta_str}", size="xs"
                            ),
                            dmc.Progress(value=prog, size="sm"),
                        ],
                    )
                )
            return children, stats or "Idle"
