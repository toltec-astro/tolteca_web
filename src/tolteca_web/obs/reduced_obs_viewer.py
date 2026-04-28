"""ReducedObsViewerPage — KIDs reduction results viewer.

Extends the SweepViewer concept with KIDs reduction overlays:

- Same URL params as SweepViewer (``quartets``, ``selected``, ``plot_mode``)
- KIDs summary bar: N_detected, N_bad_chans, quality_score per cell
- S21 subplot: vertical dashed lines at detected resonance frequencies
- Channel colour coding from SweepCheck bitmask (green/amber/red)
"""

from __future__ import annotations

from urllib.parse import urlencode

from dash import Input, Output, dcc, html, no_update
from dash_component_template import Template
import dash_mantine_components as dmc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

from tolteca_web.obs import ObsDataService

__all__ = ["ReducedObsViewerPage"]

# ── URL param parsing (mirrors sweep_viewer) ───────────────────────────────────


def _parse_cell_key(s: str) -> tuple[str, int, int, int, int] | None:
    """Parse ``"master-obsnum-subobsnum-scannum-nw"`` → 5-tuple or None."""
    parts = s.split("-")
    if len(parts) != 5:
        return None
    try:
        master, obsnum, subobsnum, scannum, nw = parts
        return master, int(obsnum), int(subobsnum), int(scannum), int(nw)
    except ValueError:
        return None


def _parse_quartets(s: str) -> list[tuple[str, int, int, int]]:
    out: list[tuple[str, int, int, int]] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        parts = tok.split("-")
        if len(parts) != 4:
            continue
        try:
            out.append((parts[0], int(parts[1]), int(parts[2]), int(parts[3])))
        except ValueError:
            pass
    return out


def _parse_selected(s: str) -> set[str]:
    return {t.strip() for t in (s or "").split(",") if t.strip()}


# ── Plotting helpers ───────────────────────────────────────────────────────────

_SWEEP_CHECK_COLORS = {
    "good": "#2f9e44",   # green
    "warn": "#e67700",   # amber
    "bad": "#e03131",    # red
}


def _build_kids_s21_figure(
    ds,
    f_tone_hz: np.ndarray | None,
    mask_chan_bad: np.ndarray | None,
    chan_start: int,
    chan_end: int,
    lo_center_hz: float,
) -> go.Figure:
    """Build S21 subplots with resonance overlays.

    Parameters
    ----------
    ds
        xarray Dataset with I, Q, tone_freq, lo_freq.
    f_tone_hz
        Detected tone offsets from LO centre (Hz), or None.
    mask_chan_bad
        Boolean array (n_chans,) — True = bad channel.
    chan_start, chan_end
        Channel range to plot.
    lo_center_hz
        LO centre frequency in Hz (for axis labels).

    Returns
    -------
    go.Figure
        Plotly figure with S21 (dB) traces and resonance markers.
    """
    n_chan = ds.sizes["chan"]
    lo_freq = ds.coords["lo_freq"].values  # (n_steps,)
    tone_freq = ds["tone_freq"].values      # (n_chan,) Hz offsets
    I = ds["I"].values                      # (n_chan, n_steps)
    Q = ds["Q"].values

    plot_chans = list(range(max(0, chan_start), min(n_chan, chan_end)))
    n_plot = len(plot_chans)
    if n_plot == 0:
        return go.Figure()

    n_cols = min(6, n_plot)
    n_rows = (n_plot + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.04, vertical_spacing=0.08,
    )

    sweep_offset_mhz = (lo_freq - lo_center_hz) / 1e6  # x-axis in MHz

    for idx, ch in enumerate(plot_chans):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        amp = np.sqrt(I[ch] ** 2 + Q[ch] ** 2)
        db = 20 * np.log10(np.maximum(amp, 1e-12))

        is_bad = bool(mask_chan_bad[ch]) if mask_chan_bad is not None else False
        line_color = _SWEEP_CHECK_COLORS["bad"] if is_bad else "#228be6"

        fig.add_trace(
            go.Scatter(
                x=sweep_offset_mhz,
                y=db,
                mode="lines",
                line={"color": line_color, "width": 1},
                name=f"ch{ch}",
                showlegend=False,
            ),
            row=row, col=col,
        )

        # Resonance overlay: vertical dashed line at tone offset
        ch_tone_hz = tone_freq[ch]  # offset from LO centre
        if f_tone_hz is not None and len(f_tone_hz) > 0:
            # Find closest detected resonance to this channel's tone
            dists = np.abs(f_tone_hz - ch_tone_hz)
            closest_idx = int(np.argmin(dists))
            if dists[closest_idx] < 1e6:  # within 1 MHz
                res_mhz = f_tone_hz[closest_idx] / 1e6
                fig.add_vline(
                    x=res_mhz,
                    line={"color": "#a61e4d", "width": 1.5, "dash": "dash"},
                    row=row, col=col,
                )

        # Subplot title (channel number)
        fig.update_xaxes(
            title_text=f"ch{ch}" + (" ✗" if is_bad else ""),
            title_font={"size": 9, "color": _SWEEP_CHECK_COLORS["bad"] if is_bad else "#495057"},
            tickfont={"size": 8},
            row=row, col=col,
        )
        fig.update_yaxes(
            tickfont={"size": 8},
            row=row, col=col,
        )

    subplot_height = 160
    fig.update_layout(
        height=max(320, n_rows * subplot_height),
        margin={"l": 30, "r": 10, "t": 20, "b": 20},
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa",
    )
    return fig


# ── Page ──────────────────────────────────────────────────────────────────────


class ReducedObsViewerPage(Template):
    """KIDs reduction results viewer page (``/reduced-obs``).

    Parameters
    ----------
    data_service
        :class:`~tolteca_web.obs.ObsDataService` providing catalog and zarr access.
    """

    def __init__(self, data_service: ObsDataService) -> None:
        super().__init__()
        self._svc = data_service

        provider = self.child[dmc.MantineProvider]()
        root = provider.child[dmc.Stack](gap=0, style={"minHeight": "100vh"})

        # ── URL location (read query params) ──────────────────────────────
        self._location = root.child[dcc.Location](id="reduced-obs-url", refresh=False)

        # ── Header ────────────────────────────────────────────────────────
        header_box = root.child[dmc.Paper](
            withBorder=True,
            shadow="none",
            style={"borderLeft": "none", "borderRight": "none", "borderTop": "none"},
        )
        hdr = header_box.child[dmc.Group](px="md", py="xs", justify="space-between")
        hdr.child[dmc.Title](children="TolTEC KIDs Reduction Viewer", order=4)
        _hdr_right = hdr.child[dmc.Group](gap="md")
        self._back_link = _hdr_right.child[dmc.Anchor](
            "← Sweep Viewer", href="/sweep", size="sm", c="dimmed",
        )
        self._diag_link = _hdr_right.child[dmc.Anchor](
            "Diagnostics →", href="/kids-diag", size="sm",
        )

        container = root.child[dmc.Container](fluid=True, px="md", pt="sm")

        # ── KIDs summary bar ──────────────────────────────────────────────
        summary_paper = container.child[dmc.Paper](
            withBorder=True, p="sm", mb="sm", radius="sm",
        )
        summary_hdr = summary_paper.child[dmc.Group](
            gap="xs", align="center", mb="xs"
        )
        summary_hdr.child[dmc.Text]("KIDs Reduction Summary", size="sm", fw=600)
        self._kids_summary_group = summary_paper.child[dmc.Group](
            gap="lg", align="center", children=[]
        )

        # ── Channel pager ─────────────────────────────────────────────────
        pager_row = container.child[dmc.Group](gap="md", align="center", mb="xs")
        self._chan_start = pager_row.child[dmc.NumberInput](
            label="Chan start", value=0, min=0, step=6, w=120, size="xs",
        )
        self._chan_end = pager_row.child[dmc.NumberInput](
            label="Chan end", value=24, min=1, step=6, w=120, size="xs",
        )
        pager_row.child[dmc.Text](
            "Pink dashed = detected resonance  ·  red label = bad channel",
            size="xs", c="dimmed",
        )

        # ── S21 plot ──────────────────────────────────────────────────────
        self._graph = container.child[dcc.Graph](
            figure={},
            style={"width": "100%"},
            config={"displayModeBar": False},
        )
        self._status = container.child[dmc.Text](
            children="Select a quartet and cell in the URL to view results.",
            size="xs", c="dimmed", mt="xs",
        )

    def setup_callbacks(self, app) -> None:
        """Register Dash callbacks."""

        svc = self._svc

        @app.callback(
            Output(self._back_link(), "href"),
            Output(self._diag_link(), "href"),
            Input(self._location(), "search"),
        )
        def _update_links(search: str) -> tuple[str, str]:
            s = search or ""
            return f"/sweep{s}", f"/kids-diag{s}"

        @app.callback(
            Output(self._kids_summary_group(), "children"),
            Output(self._graph(), "figure"),
            Output(self._status(), "children"),
            Input(self._location(), "search"),
            Input(self._chan_start(), "value"),
            Input(self._chan_end(), "value"),
        )
        def _update_view(
            search: str,
            chan_start: int | None,
            chan_end: int | None,
        ) -> tuple:
            from urllib.parse import parse_qs, urlparse

            qs = parse_qs((search or "").lstrip("?"))
            quartets_str = (qs.get("quartets", [""])[0] or "").strip()
            selected_str = (qs.get("selected", [""])[0] or "").strip()

            cs = int(chan_start) if chan_start is not None else 0
            ce = int(chan_end) if chan_end is not None else 24

            cells = sorted(_parse_selected(selected_str))
            if not cells:
                # Try to use first quartet + nw=0 as default
                quartets = _parse_quartets(quartets_str)
                if not quartets:
                    return [], {}, "No observation selected. Use the Sweep Viewer to select a cell."
                master, obsnum, subobsnum, scannum = quartets[0]
                cell_key = f"{master}-{obsnum}-{subobsnum}-{scannum}-0"
                cells = [cell_key]

            # Use first selected cell
            first_cell = cells[0]
            parsed = _parse_cell_key(first_cell)
            if not parsed:
                return [], {}, f"Cannot parse cell key: {first_cell}"

            master, obsnum, subobsnum, scannum, nw = parsed

            # Load sweep data
            try:
                ds = svc.get_obs_data(master, obsnum, subobsnum, scannum, nw)
                ds = ds.compute()
            except Exception as exc:
                return [], {}, f"Error loading sweep data: {exc}"

            lo_center_hz = float(ds.attrs.get("lo_center_freq_hz", 0.0))

            # Load KIDs results
            kf = svc.get_kids_find(master, obsnum, subobsnum, scannum, nw)
            sc = svc.get_sweep_check(master, obsnum, subobsnum, scannum, nw)

            f_tone_hz = kf.f_tone_hz if kf is not None else None
            mask_chan_bad = sc.mask_chan_bad if sc is not None else None

            # Build summary badges
            summary_items: list = []
            if kf is not None:
                n_det = len(kf.f_tone_hz)
                n_bad = int(sc.mask_chan_bad.sum()) if sc is not None else 0
                n_chan = ds.sizes["chan"]
                n_good = max(n_chan - n_bad, 0)
                quality = min(n_det / n_good, 1.0) if n_good > 0 else 0.0
                summary_items = [
                    dmc.Group(
                        [
                            dmc.Text("Cell:", size="xs", c="dimmed", fw=600),
                            dmc.Badge(first_cell, size="sm", variant="light", color="gray"),
                        ],
                        gap=4,
                    ),
                    dmc.Group(
                        [
                            dmc.Text("N detected:", size="xs", c="dimmed", fw=600),
                            dmc.Badge(
                                str(n_det), size="sm", variant="filled",
                                color="violet" if n_det > 0 else "gray",
                            ),
                        ],
                        gap=4,
                    ),
                    dmc.Group(
                        [
                            dmc.Text("N bad chans:", size="xs", c="dimmed", fw=600),
                            dmc.Badge(
                                str(n_bad), size="sm", variant="light",
                                color="red" if n_bad > 0 else "green",
                            ),
                        ],
                        gap=4,
                    ),
                    dmc.Group(
                        [
                            dmc.Text("Quality:", size="xs", c="dimmed", fw=600),
                            dmc.Badge(
                                f"{quality:.2f}", size="sm", variant="light",
                                color="green" if quality > 0.7 else ("orange" if quality > 0.3 else "red"),
                            ),
                        ],
                        gap=4,
                    ),
                ]
            else:
                summary_items = [
                    dmc.Text(
                        "No KIDs reduction data found. Run 'reduce_obs' Dagster asset first.",
                        size="sm", c="dimmed",
                    )
                ]

            # Build S21 figure
            try:
                fig = _build_kids_s21_figure(
                    ds=ds,
                    f_tone_hz=f_tone_hz,
                    mask_chan_bad=mask_chan_bad,
                    chan_start=cs,
                    chan_end=ce,
                    lo_center_hz=lo_center_hz,
                )
            except Exception as exc:
                return summary_items, {}, f"Error building plot: {exc}"

            n_det_str = str(len(f_tone_hz)) if f_tone_hz is not None else "?"
            status = (
                f"{first_cell}  ·  {ds.sizes['chan']} channels  ·  "
                f"{n_det_str} resonances detected"
            )
            return summary_items, fig, status
