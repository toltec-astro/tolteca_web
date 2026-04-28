"""Live-update section template."""

from __future__ import annotations

from dash import dcc
from dash_component_template import Template

import dash_mantine_components as dmc

from .timer import IntervalTimer

__all__ = ["LiveUpdateSection"]


class LiveUpdateSection(Template):
    """A layout section with a title, live-update timer and loading indicator.

    Composes `IntervalTimer` and `dmc.LoadingOverlay` into a two-row layout:
    - Top row:  title | timer icon/controls
    - Content area: a ``pos="relative"`` Box; add your content via ``.content``
    - Banner row: alert/status message area

    ``dmc.LoadingOverlay`` in the current DMC version is a sibling overlay
    (not a wrapper); it lives inside ``self.content`` and is toggled via its
    ``visible`` prop.

    Parameters
    ----------
    title : str or Dash component
        Section title shown in the top row.
    interval_options : list[int]
        Allowed timer intervals in milliseconds.
    interval_option_value : int, optional
        Default timer interval.

    Attributes
    ----------
    timer : IntervalTimer
        Use ``Input(section.timer.n_calls_store(), "data")`` to trigger updates.
    content : dmc.Box node
        Add your content here via ``.content.child[...]()``.
    loading_overlay : dmc.LoadingOverlay node
        Toggle ``visible`` prop to show/hide the spinner.
    banner : dmc.Group node
        Alert/status message area below the content box.

    Examples
    --------
    >>> sec = LiveUpdateSection("My Data", interval_options=[5_000, 30_000])
    >>> label = sec.content.child[dmc.Text](children="...", id="my-label")
    """

    def __init__(
        self,
        title: str | object = "Live Update",
        interval_options: list[int] | None = None,
        interval_option_value: int | None = None,
    ) -> None:
        super().__init__()

        outer = self.child[dmc.Stack](gap="xs")

        # Top row: title + timer
        top_row = outer.child[dmc.Group](gap="xs", align="center")
        if isinstance(title, str):
            top_row.child[dmc.Text](children=title, fw=500)
        else:
            top_row.child(title)

        self.timer = IntervalTimer(
            interval_options=interval_options,
            interval_option_value=interval_option_value,
        )
        top_row.child(self.timer)

        # Content area: Box with pos="relative" so the overlay works correctly
        self.content = outer.child[dmc.Box](pos="relative")
        self.loading_overlay = self.content.child[dmc.LoadingOverlay](
            visible=False,
            loaderProps={"type": "dots"},
            overlayProps={"radius": "sm", "blur": 1},
            zIndex=100,
        )

        # Banner row for alerts
        self.banner = outer.child[dmc.Group](gap="xs")
