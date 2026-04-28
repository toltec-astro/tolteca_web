"""Plot utilities."""

from __future__ import annotations

__all__ = ["make_empty_figure"]


def make_empty_figure() -> dict:
    """Return a minimal empty Plotly figure dict with hidden axes.

    Examples
    --------
    >>> fig = make_empty_figure()
    >>> fig["data"]
    []
    """
    return {
        "data": [],
        "layout": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
        },
    }
