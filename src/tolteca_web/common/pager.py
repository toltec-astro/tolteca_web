import json

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from dash_component_template import ComponentTemplate
from tollan.utils.fmt import pformat_yaml

from .collapsecontent import CollapseContent
from .labeledselect import LabeledChecklist, LabeledDropdown

__all__ = [
    "ChecklistPager",
]


def _to_typed(s):
    """Return a typed object from string `s` if possible."""
    if not isinstance(s, str):
        raise TypeError("input object has to be string.")
    if "." not in s:
        try:
            return int(s)
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        return s


def _get_items_range(page_id, n_items_per_page, n_items):
    start = page_id * n_items_per_page
    stop = start + n_items_per_page
    # cutoff at n_items if it is possible
    if n_items is not None and n_items > 0 and stop > n_items:
        stop = n_items
    return start, stop


def _get_n_pages(n_items_per_page, n_items):
    if n_items == 0:
        return 1
    return n_items // n_items_per_page + (n_items % n_items_per_page > 0)


class ChecklistPager(ComponentTemplate):
    """A check list pager."""

    class Meta:  # noqa: D106
        component_cls = html.Div

    def __init__(self, title_text, n_items_per_page_options, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._title_text = title_text
        self._n_items_per_page_options = n_items_per_page_options

        self._n_items_store = self.child(dcc.Store, data=0)
        self._current_page_store = self.child(dcc.Store)

    @property
    def n_items_store(self):
        """The pager data store."""
        return self._n_items_store

    @property
    def current_page_store(self):
        """The pager data store."""
        return self._current_page_store

    def setup_layout(self, app):  # noqa: D102

        container = self
        checklist_container = container.child(dbc.Form).child(
            dbc.Row,
            className="gx-2 gy-2",
        )
        page_select = checklist_container.child(
            LabeledChecklist(
                label_text=self._title_text,
                className="w-auto me-3 align-items-baseline",
                size="sm",
                multi=False,
            ),
        ).checklist

        def _make_default_page_select_options():
            return [
                {
                    "label": "All",
                    "value": json.dumps(
                        {
                            "page_id": 0,
                            "start": None,
                            "stop": None,
                        },
                    ),
                },
            ]

        page_select.options = _make_default_page_select_options()
        page_select.value = page_select.options[0]["value"]

        n_items_per_page_drp = checklist_container.child(
            LabeledDropdown(
                label_text="items per page",
                label_first=False,
                className="w-auto me-2 align-items-baseline",
                size="sm",
            ),
        ).dropdown
        n_items_per_page_drp.options = [
            {"label": v, "value": v} for v in self._n_items_per_page_options
        ]
        n_items_per_page_drp.value = self._n_items_per_page_options[0]

        debug_info = (
            checklist_container.child(
                dbc.InputGroup,
                className="w-auto align-items-baseline",
            )
            .child(
                CollapseContent(
                    button_text="Details ...",
                ),
            )
            .content
        )

        settings_store = checklist_container.child(dcc.Store)

        super().setup_layout(app)

        @app.callback(
            [
                Output(page_select.id, "options"),
                Output(page_select.id, "value"),
                Output(settings_store.id, "data"),
            ],
            [
                Input(self.n_items_store.id, "data"),
                Input(n_items_per_page_drp.id, "value"),
            ],
        )
        def update_page_select(n_items, n_items_per_page):
            n_items = int(n_items)
            n_items_per_page = int(n_items_per_page)
            n_pages = _get_n_pages(n_items_per_page, n_items)
            options = []
            for i in range(n_pages):
                start, stop = _get_items_range(i, n_items_per_page, n_items)
                options.append(
                    {
                        "label": f"{start}",
                        "value": json.dumps(
                            {
                                "page_id": i,
                                "start": start,
                                "stop": stop,
                            },
                        ),
                    },
                )
            # option for all
            options.extend(_make_default_page_select_options())
            return (
                options,
                options[0]["value"],
                {
                    "n_items": n_items,
                    "n_pages": n_pages,
                    "n_items_per_page": n_items_per_page,
                },
            )

        @app.callback(
            Output(debug_info.id, "children"),
            [
                Input(settings_store.id, "data"),
                Input(self.current_page_store.id, "data"),
            ],
        )
        def update_details(settings_data, page_data):
            return html.Pre(pformat_yaml(settings_data | page_data))

        @app.callback(
            Output(self.current_page_store.id, "data"),
            [
                Input(page_select.id, "value"),
                State(settings_store.id, "data"),
            ],
        )
        def update_current_page(page_select_value, settings):
            if not settings or not page_select_value:
                return None
            page_info = json.loads(page_select_value)
            return page_info | settings
