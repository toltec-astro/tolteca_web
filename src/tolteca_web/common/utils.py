#! /usr/bin/env python

import copy
import json

import dash
from dash import Input, Output, State, html
from plotly.subplots import make_subplots as _make_subplots

__all__ = [
    "PatternMatchingId",
    "update_class_name",
    "remove_class_name",
    "fa",
    "to_dependency",
    "parse_prop_id",
    "parse_triggered_prop_ids",
    "make_subplots",
]


class PatternMatchingId(object):
    """A helper class to create pattern matching ids."""

    def __init__(self, auto_index=True, **base):
        id = dict()
        if auto_index:
            id.update({"index": -1})
        if base is not None:
            id.update(base)
        self._id = id
        self._iter_index_inst = self._iter_index()
        # when auto index is false, the pmid does not return index
        # in the id.
        self._auto_index = auto_index

    def __call__(self, **kwargs):
        # make sure kwargs only contains keys in base
        k1 = set(kwargs.keys())
        k0 = self._id.keys()
        if k1 > k0:
            raise ValueError(f"invalid keys {k1 - k0}")
        id = copy.copy(self._id)
        id.update(kwargs)
        if "index" not in kwargs:
            if self._auto_index:
                id["index"] = self.make_id()
        return id

    def make_id(self):
        """Return an unique id."""
        return next(self._iter_index_inst)

    @staticmethod
    def _iter_index():
        h = 0
        while True:
            yield h
            h += 1


class ClassName(object):
    """A helper class to manage the ``className`` property."""

    def __init__(self, className):
        if isinstance(className, ClassName):
            self._data = copy.copy(className._data)
        elif not className:
            self._data = set()
        else:
            self._data = set(className.split(" "))

    def update(self, other):
        other = self.__class__(other)
        self._data.update(other._data)

    def remove(self, other):
        other = self.__class__(other)
        self._data -= other._data

    def __str__(self):
        return " ".join(self._data)


def update_class_name(className, *args):
    """Add classes to `className`."""
    cn = ClassName(className)
    for arg in args:
        cn.update(arg)
    return str(cn)


def remove_class_name(className, *args):
    """Remove classes from `className`."""
    cn = ClassName(className)
    for arg in args:
        cn.remove(arg)
    return str(cn)


def fa(icon, **kwargs):
    """Return a font-awesome icon."""
    className = update_class_name(icon, kwargs.pop("className", None))
    return html.I(className=className, **kwargs)


def to_dependency(type_, dep):
    """Convert a dependency object to another `type_`.

    Parameters
    ----------
    type_ : str
        The type to convert to, choosing from "state", "input", or "output".

    """
    dispatch = {"state": State, "input": Input, "output": Output}
    return dispatch[type_](dep.component_id, dep.component_property)


def parse_prop_id(prop_id):
    """Return a parsed `prop_id`."""
    d, v = prop_id.rsplit(".", 1)
    if d == "":
        return None
    if "{" in d:
        d = json.loads(d)
    return {"id": d, "prop": v}


def parse_triggered_prop_ids():
    """Return a parsed triggered `prop_id`."""
    return [parse_prop_id(d["prop_id"]) for d in dash.callback_context.triggered]


def make_subplots(nrows, ncols, fig_layout=None, **kwargs):
    """Return a sensible multi-panel figure with predefined layout."""
    _fig_layout = {
        "uirevision": True,
        "xaxis_autorange": True,
        "yaxis_autorange": True,
        "showlegend": True,
    }
    if fig_layout is not None:
        _fig_layout.update(fig_layout)
    fig = _make_subplots(nrows, ncols, **kwargs)
    fig.update_layout(**_fig_layout)
    return fig
