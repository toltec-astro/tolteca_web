"""Pagination widget template."""

from __future__ import annotations

from dash import Input, Output, State, dcc
from dash_component_template import Template

import dash_mantine_components as dmc

__all__ = ["Pager"]

_DEFAULT_PAGE_SIZES = [10, 25, 50, 100]


def _n_pages(n_items: int, per_page: int) -> int:
    if per_page <= 0 or n_items <= 0:
        return 0
    return max(1, -(-n_items // per_page))  # ceiling division


class Pager(Template):
    """Pagination widget backed by `dmc.Pagination`.

    Exposes a `page_store` whose ``data`` dict has the shape::

        {
            "page":   1,           # 1-based
            "start":  0,           # slice start (0-based)
            "stop":   10,          # slice stop (exclusive)
            "n_items": 100,
            "n_pages": 10,
            "per_page": 10,
        }

    Set ``n_items_store.data`` to the total item count from your callback;
    the widget recalculates pages automatically.

    Parameters
    ----------
    per_page_options : list[int], optional
        Allowed page-size choices; defaults to [10, 25, 50, 100].
    default_per_page : int, optional
        Initial page size; defaults to first option.

    Attributes
    ----------
    n_items_store : dcc.Store node — write total item count here
    page_store    : dcc.Store node — read current page info from here

    Examples
    --------
    >>> pager = Pager(per_page_options=[10, 25, 50])
    """

    def __init__(
        self,
        per_page_options: list[int] | None = None,
        default_per_page: int | None = None,
    ) -> None:
        super().__init__()
        per_page_options = per_page_options or _DEFAULT_PAGE_SIZES
        default_per_page = default_per_page or per_page_options[0]

        self.n_items_store = self.child[dcc.Store](data=0)
        self.page_store = self.child[dcc.Store](data={})

        row = self.child[dmc.Group](gap="xs", align="center")

        self._pagination = row.child[dmc.Pagination](total=1, value=1, siblings=1)
        self._per_page = row.child[dmc.Select](
            data=[{"label": str(n), "value": str(n)} for n in per_page_options],
            value=str(default_per_page),
            w=80,
            size="xs",
        )
        row.child[dmc.Text](children="/ page", size="xs", c="dimmed")

    def setup_callbacks(self, app) -> None:
        """Reset to page 1 and update total when n_items changes."""

        @app.callback(
            Output(self._pagination(), "total"),
            Output(self._pagination(), "value"),
            Input(self.n_items_store(), "data"),
            Input(self._per_page(), "value"),
        )
        def _update_total(n_items: int | None, per_page_str: str) -> tuple:
            n = int(n_items or 0)
            pp = int(per_page_str or 10)
            return _n_pages(n, pp), 1

        @app.callback(
            Output(self.page_store(), "data"),
            Input(self._pagination(), "value"),
            State(self.n_items_store(), "data"),
            State(self._per_page(), "value"),
        )
        def _update_page(page: int, n_items: int | None, per_page_str: str) -> dict:
            n = int(n_items or 0)
            pp = int(per_page_str or 10)
            page = int(page or 1)
            start = (page - 1) * pp
            stop = min(start + pp, n)
            return {
                "page": page,
                "start": start,
                "stop": stop,
                "n_items": n,
                "n_pages": _n_pages(n, pp),
                "per_page": pp,
            }
