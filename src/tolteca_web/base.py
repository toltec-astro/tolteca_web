import dash_bootstrap_components as dbc
from dash_component_template import ComponentTemplate
from .common import ComponentStateManager


class ViewerBase(ComponentTemplate):
    """A base class for viewers."""

    class Meta:  # noqa: D106
        component_cls = dbc.Container

    _state_manager: ComponentStateManager

    def __init__(self, manager_kw=None, **kwargs):
        super().__init__(**kwargs)
        self._state_manager = self.child(ComponentStateManager(**(manager_kw or {})))

    @property
    def state_manager(self):
        """Return the state manager."""
        return self._state_manager

    def setup_layout(self, app):
        """Set up the data prod viewr layout."""
        # this moves the component state manager to the last in the root
        self.children = sorted(
            self.children,
            key=lambda c: isinstance(c, ComponentStateManager),
        )
        super().setup_layout(app)
