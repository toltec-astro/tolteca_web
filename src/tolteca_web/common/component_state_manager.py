"""The TolTEC data product viewer."""

from urllib.parse import parse_qs
from dash import html
from dash_component_template import ComponentTemplate
from dash import dcc, Input, Output, State
import dash

from .collapsecontent import CollapseContent
from tollan.utils.fmt import pformat_yaml

from collections.abc import Callable


class ComponentStateManager(ComponentTemplate):
    """A class that allow additional management of components."""

    class Meta:  # noqa: D106
        component_cls = html.Div

    _data: dcc.Store
    _url: dcc.Location
    _components: dict[str, "ManagedComponent"]
    _mapper_funcs: dict[str, Callable]
    _show_info: bool

    def __init__(self, show_info=True, mapper_funcs=None, **kwargs):
        super().__init__(**kwargs)
        self._data = self.child(dcc.Store)
        self._url = self.child(dcc.Location)
        self._components = {}
        self._mapper_funcs = mapper_funcs or {}
        self._show_info = show_info

    def setup_layout(self, app):  # noqa: C901
        """Set up the manager layout."""
        # generate info view
        container = self
        if self._show_info:
            info_container = container.child(
                CollapseContent("manager info ..."),
            ).content
            info = info_container.child(html.Pre)

        super().setup_layout(app)

        if self._show_info:
            # collate all info in to the debug view
            @app.callback(
                Output(info.id, "children"),
                [
                    Input(self.url.id, "search"),
                    Input(self.data.id, "data"),
                    State(self.url.id, "href"),
                ]
                + [Input(c.id, "data") for c in self.components.values()],
            )
            def collect_info(query_str, data, href, *component_data):
                data = {
                    "href": href,
                    "query_str": query_str,
                    "data": data,
                    "comonents": {
                        c.key: d
                        for (c, d) in zip(
                            self.components.values(),
                            component_data,
                            strict=True,
                        )
                    },
                }
                return pformat_yaml(data)

        # connect url to data store
        @app.callback(
            Output(self.data.id, "data", allow_duplicate=True),
            [
                Input(self.url.id, "search"),
            ],
            prevent_initial_call=True,
        )
        def update_data_from_url(query_str):
            if not query_str:
                return dash.no_update
            data = parse_qs(query_str[1:])
            if data is None:
                return dash.no_update
            return data

        # connect data store to managed components stores
        if self.components:

            @app.callback(
                [Output(c.id, "data") for c in self.components.values()],
                [
                    Input(self.data.id, "data"),
                ],
            )
            def map_data(data):
                if data is None:
                    return dash.no_update
                cdata = []
                for k, c in self.components.items():
                    if k not in data:
                        cdata.append(dash.no_update)
                    else:
                        cdata.append(c.mapper_func(data[k]))
                return cdata

    @property
    def url(self):
        """The url state."""
        return self._url

    @property
    def data(self):
        """The manager data."""
        return self._data

    @property
    def components(self):
        """The managed components."""
        return self._components

    def register(self, key, component, props, mapper_func=None):
        """Add the component to the manager."""
        mc = self.child(
            self.ManagedComponent(
                self,
                key=key,
                component=component,
                props=props,
                mapper_func=mapper_func or self._mapper_funcs.get(key, None),
            ),
        )
        self._components[mc.key] = mc

    class ManagedComponent(ComponentTemplate):
        """An adapter component that connect the component with the manager."""

        class Meta:  # noqa: D106
            component_cls = dcc.Store

        _manager: "ComponentStateManager"
        _key: str
        _component: ComponentTemplate
        _props: list[str]
        _mapper_func: Callable

        def __init__(  # noqa: PLR0913
            self,
            manager,
            key,
            component,
            props,
            mapper_func,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self._manager = manager
            self._key = key
            self._component = component
            self._props = props
            self._mapper_func = mapper_func or (lambda d: {p: d for p in self._props})

        def setup_layout(self, app):
            """Set up the data prod viewr layout."""
            super().setup_layout(app)

            if self._props:

                @app.callback(
                    [Output(self._component.id, p) for p in self._props],
                    [
                        Input(self.id, "data"),
                    ],
                    # prevent_initial_call=True,
                )
                def update_props(data):
                    return [data.get(p, dash.no_update) for p in self._props]

        @property
        def key(self):
            """The key used to identify this item."""
            return self._key

        @property
        def mapper_func(self):
            """The function that maps manager data to component data."""
            return self._mapper_func
