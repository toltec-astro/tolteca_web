"""Common widget templates for tolteca_web.

All templates are built on `dash_component_template.Template` and use
`dash-mantine-components` for UI, replacing legacy `dash-bootstrap-components`.

Available templates
-------------------
CollapseContent
    Toggle button + collapsible panel (clientside, zero server round-trips).
DownloadButton
    `dmc.Button` with download icon pre-wired to `dcc.Download`.
IntervalTimer
    Configurable live-update timer with pause, progress bar and rate selector.
LiveUpdateSection
    Section layout: title + `IntervalTimer` + `dmc.LoadingOverlay` + banner.
Pager
    Pagination widget backed by `dmc.Pagination`; exposes `page_store`.
CacheMonitor
    Per-file download progress bars driven by a `dcc.Store`.
UrlStateManager
    Bidirectional URL params ↔ Pydantic model ↔ per-field ``dcc.Store``
    synchronizer.  Pass a ``BaseModel`` subclass; call ``bind()`` to wire
    components; use ``store(field)`` for targeted subscriptions.
ComponentStateManager
    Unidirectional state → widget base class.  External callbacks write to
    ``state_store``; bound widgets update automatically.  No URL sync.

Plot helpers
------------
SurfacePlot
    2-D heatmap/scatter with histogram sidebar and range slider.
"""

from __future__ import annotations

from .cache_monitor import CacheMonitor
from .collapse_content import CollapseContent
from .component_state_manager import ComponentStateManager
from .download_button import DownloadButton
from .live_update_section import LiveUpdateSection
from .pager import Pager
from .timer import IntervalTimer
from .url_state_manager import UrlStateManager

__all__ = [
    "CacheMonitor",
    "CollapseContent",
    "ComponentStateManager",
    "DownloadButton",
    "IntervalTimer",
    "LiveUpdateSection",
    "Pager",
    "UrlStateManager",
]
