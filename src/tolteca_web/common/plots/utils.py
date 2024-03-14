import itertools

import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots as _make_subplots


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
    fig = _make_subplots(rows=nrows, cols=ncols, **kwargs)
    xaxes = _fig_layout.pop("xaxis", {})
    yaxes = _fig_layout.pop("yaxis", {})
    fig.update_layout(**_fig_layout)
    for i in range(nrows):
        for j in range(ncols):
            fig.update_xaxes(col=j + 1, row=i + 1, **xaxes)
            fig.update_yaxes(col=j + 1, row=i + 1, **yaxes)
    return fig


def make_empty_figure():
    """Return an empty figure."""
    return {
        "layout": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            # "annotations": [
            #     {
            #         "text": "No matching data found",
            #         "xref": "paper",
            #         "yref": "paper",
            #         "showarrow": False,
            #         "font": {
            #             "size": 28
            #             }
            #         }
            #     ]
        },
    }


class ColorPalette:
    """A class to manage colors."""

    name: str
    colors: list

    def __init__(self, name="Dark24"):
        colors = getattr(px.colors.qualitative, name, None)
        if colors is None:
            raise ValueError("invalid color sequence name.")
        self.name = name
        self.colors = colors

    def get_scaled(self, scale):
        """Return scaled colors."""
        colors = self.colors
        if scale >= 1:
            return colors
        return [
            "#{:02x}{:02x}{:02x}".format(
                *(
                    np.array(
                        px.colors.find_intermediate_color(
                            np.array(px.colors.hex_to_rgb(c)) / 255.0,
                            (1, 1, 1),
                            scale,
                        ),
                    )
                    * 255.0
                ).astype(int),
            )
            for c in colors
        ]

    def cycle(self, scale=1):
        """Return color cycle."""
        return itertools.cycle(self.get_scaled(scale))

    def cycles(self, *scales):
        """Return color cycles."""
        return itertools.cycle(
            zip(
                *(self.get_scaled(scale) for scale in scales),
                strict=False,
            ),
        )

    def cycle_alternated(self, *scales):
        """Return color cycles."""
        return itertools.cycle(
            itertools.chain.from_iterable(
                zip(
                    *(self.get_scaled(scale) for scale in scales),
                    strict=False,
                ),
            ),
        )
