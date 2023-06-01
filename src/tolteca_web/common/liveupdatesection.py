#! /usr/bin/env python


from copy import copy

from dash import dcc, html
from dash_component_template import ComponentTemplate

from .timer import IntervalTimer


class LiveUpdateSection(ComponentTemplate):
    class Meta:
        component_cls = html.Div

    def __init__(
        self, title_component, interval_options, interval_option_value, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.title_component = title_component
        self.interval_options = interval_options
        self.interval_option_value = interval_option_value

        container, banner_container = self.grid(2, 1)
        container.className = "d-flex align-items-center"
        title = copy(self.title_component)
        title.className = "me-2 my-0"
        container.child(title)
        self._timer = container.child(
            IntervalTimer(
                interval_options=self.interval_options,
                interval_option_value=self.interval_option_value,
            )
        )
        self._loading = container.child(dcc.Loading, parent_className="ms-4")
        banner_container.className = "d-flex"
        self._banner = banner_container

    def setup_layout(self, app):
        super().setup_layout(app)

    @property
    def timer(self):
        return self._timer

    @property
    def loading(self):
        return self._loading

    @property
    def banner(self):
        return self._banner
