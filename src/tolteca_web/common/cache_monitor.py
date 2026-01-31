"""Cache download progress monitor widget."""

from __future__ import annotations

import time

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html
from dash_component_template import ComponentTemplate


def _format_speed(bytes_per_sec: float) -> str:
    """Format download speed in human-readable form."""
    if bytes_per_sec >= 1024 * 1024:
        return f"{bytes_per_sec / (1024 * 1024):.1f} MB/s"
    if bytes_per_sec >= 1024:
        return f"{bytes_per_sec / 1024:.1f} KB/s"
    return f"{bytes_per_sec:.0f} B/s"


def _format_eta(seconds: float) -> str:
    """Format ETA in human-readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


class CacheMonitorWidget(ComponentTemplate):
    """A widget to monitor file cache download progress.

    Displays active downloads with progress bars and cache statistics.
    Uses dcc.Interval for polling the progress store.
    """

    class Meta:  # noqa: D106
        component_cls = html.Div

    def __init__(
        self,
        *args,
        poll_interval_ms: int = 500,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._poll_interval_ms = poll_interval_ms

        self._interval = self.child(
            dcc.Interval,
            interval=self._poll_interval_ms,
            n_intervals=0,
        )
        self._status_store = self.child(dcc.Store, data={})

        # Container that can be hidden when remote is disabled
        self._container = self.child(html.Div, className="small d-flex align-items-center")
        self._container.child(html.I, className="fas fa-download me-2 text-muted")

        self._active_downloads_container = self._container.child(
            html.Div,
            className="d-flex align-items-center flex-wrap",
        )
        self._cache_stats_container = self._container.child(
            html.Span,
            className="text-muted ms-2",
        )

    def setup_layout(self, app):
        """Set up callbacks for the cache monitor widget."""
        super().setup_layout(app)

        @app.callback(
            Output(self._container.id, "style"),
            Output(self._active_downloads_container.id, "children"),
            Output(self._cache_stats_container.id, "children"),
            Input(self._status_store.id, "data"),
        )
        def update_display(status_data):
            """Update download display from status store."""
            hidden_style = {"display": "none"}
            visible_style = {}

            if not status_data:
                return hidden_style, "", ""

            # Hide when remote is not enabled
            if not status_data.get("remote_enabled", False):
                return hidden_style, "", ""

            active = status_data.get("active_downloads", {})
            stats = status_data.get("cache_stats", {})

            downloads_children = []
            if active:
                for file_id, info in active.items():
                    status = info.get("status", "unknown")
                    filename = info.get("filename", file_id.split("/")[-1])
                    total = info.get("total_bytes", 0)
                    current = info.get("current_bytes", 0)

                    if status == "downloading" and total > 0:
                        pct = int(100 * current / total)
                        size_mb = total / (1024 * 1024)

                        # Calculate speed and ETA
                        start_time = info.get("start_time", 0)
                        elapsed = time.time() - start_time if start_time else 0
                        speed = current / elapsed if elapsed > 0 else 0
                        remaining = total - current
                        eta = remaining / speed if speed > 0 else 0

                        speed_text = _format_speed(speed)
                        eta_text = _format_eta(eta) if eta > 0 else "--"

                        downloads_children.append(
                            html.Div(
                                [
                                    html.Span(
                                        f"{filename}",
                                        className="me-2",
                                        style={"maxWidth": "150px", "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"},
                                    ),
                                    dbc.Progress(
                                        value=pct,
                                        style={"height": "0.6em", "width": "100px"},
                                        className="me-2",
                                    ),
                                    html.Span(
                                        f"{pct}% | {speed_text} | ETA: {eta_text}",
                                        className="text-muted",
                                        style={"fontSize": "0.85em"},
                                    ),
                                ],
                                className="d-flex align-items-center me-3",
                            )
                        )
                    elif status == "completed":
                        downloads_children.append(
                            html.Div(
                                f"✓ {filename}",
                                className="text-success mb-1",
                            )
                        )
                    elif status == "error":
                        error = info.get("error", "Unknown error")
                        downloads_children.append(
                            html.Div(
                                f"✗ {filename}: {error}",
                                className="text-danger mb-1",
                            )
                        )
            else:
                downloads_children = [html.Span("Idle", className="text-muted")]

            stats_text = []
            if stats:
                if "total_files" in stats:
                    stats_text.append(f"Cached: {stats['total_files']} files")
                if "total_bytes" in stats:
                    size_mb = stats["total_bytes"] / (1024 * 1024)
                    stats_text.append(f"({size_mb:.1f} MB)")
            stats_children = html.Span(" ".join(stats_text)) if stats_text else ""

            return visible_style, downloads_children, stats_children

    @property
    def interval(self):
        """The polling interval component."""
        return self._interval

    @property
    def status_store(self):
        """The status store component (to be updated by parent)."""
        return self._status_store
