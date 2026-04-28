"""TelViewerPage — telescope trajectory & header viewer.

Reads LMT telescope NetCDF files (``tel_*.nc``) from a configurable data
root directory and provides:

* Header panel: ObsNum/Source, Sky offsets, Radiometer tau, M2/M3 status
* Scan plots: Trajectory, Velocity, Error, Acceleration (2×2 grid)
* Timestream plots: Az & El vs time

URL state
---------
The page is driven by a single ``?quartet=<master>-<obsnum>-<subobsnum>-<scannum>``
query parameter (the quartet key), matching the convention used throughout the
v3 portal (sweep viewer, kids-diag viewer).  The portal's **"Tel →"** action
links pass this parameter when navigating here.

Data root
---------
Set at construction time via ``data_root``.  Defaults to the workspace
``run/data_lmt/tel/`` directory (real LMT data), with a fallback to
``tolteca_ref_data/data_lmt/tel/`` for the e2e test environment.

Architecture
------------
* ``dcc.Location`` reads the URL search string on every navigation.
* URL change → resolve tel file path → load into ``dcc.Store`` (JSON).
* Store change → render all plots + header values.
* Frame toggle and downsample control re-trigger the plot callback only.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
from dash import Input, Output, dcc, html, no_update
from dash_component_template import Template

import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go

__all__ = ["TelViewerPage"]


# ── unit helpers ──────────────────────────────────────────────────────────────

def _rad2deg(x: np.ndarray) -> np.ndarray:
    return np.rad2deg(x)

def _rad2arcmin(x: np.ndarray) -> np.ndarray:
    return np.rad2deg(x) * 60.0

def _rad2arcsec(x: np.ndarray) -> np.ndarray:
    return np.rad2deg(x) * 3600.0

def _downsample_mean(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    end = n * int(len(x) / n)
    if end == 0:
        return x
    return np.mean(x[:end].reshape(-1, n), axis=1)


# ── obsspec parsing ───────────────────────────────────────────────────────────


def _parse_obsspec(obsspec: str) -> tuple[str, int, int, int] | None:
    """Parse ``master-obsnum-subobsnum-scannum`` into (master, obsnum, subobsnum, scannum).

    Returns ``None`` if the string is malformed.
    """
    try:
        parts = obsspec.split("-")
        if len(parts) < 4:
            return None
        master = parts[0]
        obsnum = int(parts[1])
        subobsnum = int(parts[2])
        scannum = int(parts[3])
        return master, obsnum, subobsnum, scannum
    except (ValueError, IndexError):
        return None


def _find_tel_file(
    data_root: Path,
    obsnum: int,
    subobsnum: int = 0,
    scannum: int = 1,
) -> str | None:
    """Return path of the tel NC file matching obsnum/subobsnum/scannum.

    Naming convention: ``tel_toltec_{date}_{obsnum}_{subobsnum:02d}_{scannum:04d}.nc``

    Tries exact match first, then loose obsnum-only match.
    """
    exact = f"tel_*_{obsnum}_{subobsnum:02d}_{scannum:04d}.nc"
    matches = sorted(data_root.glob(exact))
    if matches:
        return str(matches[0])
    # Loose fallback: any file with this obsnum
    for f in sorted(data_root.glob(f"tel_*_{obsnum}_*.nc")):
        return str(f)
    return None


# ── NC file parsing ───────────────────────────────────────────────────────────


def _read_header(nc) -> dict:
    """Extract scalar/string header fields from an open netCDF4 Dataset."""

    def _float(key: str, default: float = 0.0) -> float:
        try:
            return float(nc.variables[key][0].data)
        except Exception:
            return default

    def _str(key: str) -> str:
        try:
            return b"".join(nc.variables[key][:]).decode().strip()
        except Exception:
            return "—"

    def _zernike() -> str:
        try:
            vals = nc.variables["Header.M1.ZernikeC"][:].data
            if hasattr(vals, "__len__"):
                return "[" + ", ".join(f"{v:.3f}" for v in vals[:4]) + "…]"
            return f"{float(vals):.3f}"
        except Exception:
            return "—"

    return {
        "obsnum": int(_float("Header.Dcs.ObsNum")),
        "subobsnum": int(_float("Header.Dcs.SubObsNum")),
        "scannum": int(_float("Header.Dcs.ScanNum")),
        "source_name": _str("Header.Source.SourceName"),
        "ra_deg": float(np.rad2deg(_float("Header.Source.Ra"))),
        "dec_deg": float(np.rad2deg(_float("Header.Source.Dec"))),
        "obs_pgm": _str("Header.Dcs.ObsPgm"),
        "obs_goal": _str("Header.Dcs.ObsGoal"),
        "project_id": _str("Header.Dcs.ProjectId"),
        "tau": _float("Header.Radiometer.Tau"),
        "az_req_deg": float(np.rad2deg(_float("Header.Sky.AzReq"))),
        "el_req_deg": float(np.rad2deg(_float("Header.Sky.ElReq"))),
        "az_off_arcsec": float(_rad2arcsec(np.array([_float("Header.Sky.AzOff")]))[0]),
        "el_off_arcsec": float(_rad2arcsec(np.array([_float("Header.Sky.ElOff")]))[0]),
        "crane_in_beam": bool(_float("Header.Telescope.CraneInBeam")),
        "zernike_c": _zernike(),
        "m2_z_req_mm": _float("Header.M2.ZReq"),
        "m2_alive": bool(_float("Header.M2.Alive")),
        "m3_alive": bool(_float("Header.M3.Alive")),
        "m3_fault": bool(_float("Header.M3.Fault")),
    }


def _read_plot_data(nc, downsample: int) -> dict:
    """Extract and downsample trajectory data from an open netCDF4 Dataset."""
    n = max(1, int(downsample))

    def _ds(key: str) -> np.ndarray:
        return _downsample_mean(nc.variables[key][:].data.astype(float), n)

    tel_time = _ds("Data.TelescopeBackend.TelTime")
    tel_az_act = _ds("Data.TelescopeBackend.TelAzAct")
    tel_el_act = _ds("Data.TelescopeBackend.TelElAct")
    tel_az_des = _ds("Data.TelescopeBackend.TelAzDes")
    tel_el_des = _ds("Data.TelescopeBackend.TelElDes")
    tel_az_map = _ds("Data.TelescopeBackend.TelAzMap")
    tel_el_map = _ds("Data.TelescopeBackend.TelElMap")
    obsnum = int(float(nc.variables["Header.Dcs.ObsNum"][0].data))

    return {
        "obsnum": obsnum,
        "tel_time": tel_time.tolist(),
        "tel_az_act": tel_az_act.tolist(),
        "tel_el_act": tel_el_act.tolist(),
        "tel_az_des": tel_az_des.tolist(),
        "tel_el_des": tel_el_des.tolist(),
        "tel_az_map": tel_az_map.tolist(),
        "tel_el_map": tel_el_map.tolist(),
    }


def _load_tel_file(path: str, downsample: int) -> tuple[dict, dict]:
    """Open a tel NC file and return (header, plot_data)."""
    import netCDF4  # type: ignore[import-untyped]
    nc = netCDF4.Dataset(path)
    try:
        header = _read_header(nc)
        plot_data = _read_plot_data(nc, downsample)
    finally:
        nc.close()
    return header, plot_data


# ── figure builders ───────────────────────────────────────────────────────────


def _xy_axis_style() -> dict:
    return dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor="black",
        linewidth=1,
        ticks="outside",
        tickfont=dict(size=11),
        title_font=dict(size=11),
        zeroline=False,
    )


def _empty_fig(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=320,
        plot_bgcolor="white",
        xaxis=_xy_axis_style(),
        yaxis=_xy_axis_style(),
        title=dict(text=title, font=dict(size=12)),
        margin=dict(l=10, r=10, b=30, t=36),
    )
    return fig


def _frame_coords(
    pd: dict, frame: str
) -> tuple[np.ndarray, np.ndarray, str, str]:
    if frame == "source":
        az = _rad2arcmin(np.array(pd["tel_az_map"]))
        el = _rad2arcmin(np.array(pd["tel_el_map"]))
        return az, el, "TelAzMap [arcmin]", "TelElMap [arcmin]"
    az = _rad2deg(np.array(pd["tel_az_act"]))
    el = _rad2deg(np.array(pd["tel_el_act"]))
    return az, el, "TelAzAct [deg]", "TelElAct [deg]"


def _vel_acc(
    pd: dict, frame: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frame == "source":
        az = np.array(pd["tel_az_map"])
        el = np.array(pd["tel_el_map"])
    else:
        az = np.array(pd["tel_az_act"])
        el = np.array(pd["tel_el_act"])
    t = np.array(pd["tel_time"])
    dt = np.ediff1d(t)
    dt = np.where(dt == 0, 1e-6, dt)
    az_vel = np.ediff1d(az) / dt
    el_vel = np.ediff1d(el) / dt
    vel = np.sqrt(az_vel**2 + el_vel**2)
    az_acc = np.ediff1d(az_vel) / dt[:-1]
    el_acc = np.ediff1d(el_vel) / dt[:-1]
    acc = np.sqrt(az_acc**2 + el_acc**2)
    minlen = min(len(az), len(vel), len(acc))
    return _rad2arcsec(vel[:minlen]), np.zeros(minlen), _rad2arcsec(acc[:minlen])


def _make_trajectory_fig(pd: dict, frame: str) -> go.Figure:
    az, el, xt, yt = _frame_coords(pd, frame)
    fig = go.Figure(go.Scatter(x=az.tolist(), y=el.tolist(), mode="lines",
                               line=dict(color="#228be6", width=0.8)))
    fig.update_layout(
        height=320, plot_bgcolor="white",
        xaxis={**_xy_axis_style(), "title": xt},
        yaxis={**_xy_axis_style(), "title": yt, "scaleanchor": "x"},
        title=dict(text=f"Trajectory ({frame}) — #{pd['obsnum']}", font=dict(size=12)),
        margin=dict(l=10, r=10, b=30, t=36),
    )
    return fig


def _make_error_fig(pd: dict, frame: str) -> go.Figure:
    az, el, xt, yt = _frame_coords(pd, frame)
    az_err = np.array(pd["tel_az_act"]) - np.array(pd["tel_az_des"])
    el_err = np.array(pd["tel_el_act"]) - np.array(pd["tel_el_des"])
    error_arcsec = _rad2arcsec(np.sqrt(az_err**2 + el_err**2))
    minlen = min(len(az), len(error_arcsec))
    fig = px.scatter(x=az[:minlen], y=el[:minlen], color=error_arcsec[:minlen],
                     color_continuous_scale="Viridis",
                     labels={"x": xt, "y": yt, "color": "Error [arcsec]"})
    fig.update_traces(marker=dict(size=3))
    fig.update_coloraxes(colorbar=dict(title="Error<br>[arcsec]", thickness=12, len=0.8))
    fig.update_layout(
        height=320, plot_bgcolor="white",
        xaxis=_xy_axis_style(), yaxis=_xy_axis_style(),
        title=dict(text=f"Error ({frame}) — #{pd['obsnum']}", font=dict(size=12)),
        margin=dict(l=10, r=10, b=30, t=36),
    )
    return fig


def _make_velocity_fig(pd: dict, frame: str) -> go.Figure:
    az, el, xt, yt = _frame_coords(pd, frame)
    vel_arcsec, _, _ = _vel_acc(pd, frame)
    minlen = min(len(az), len(vel_arcsec))
    fig = px.scatter(x=az[:minlen], y=el[:minlen], color=vel_arcsec[:minlen],
                     color_continuous_scale="plasma",
                     labels={"x": xt, "y": yt, "color": "Vel [arcsec/s]"})
    fig.update_traces(marker=dict(size=3))
    fig.update_coloraxes(colorbar=dict(title="Vel<br>[arcsec/s]", thickness=12, len=0.8))
    fig.update_layout(
        height=320, plot_bgcolor="white",
        xaxis=_xy_axis_style(), yaxis=_xy_axis_style(),
        title=dict(text=f"Velocity ({frame}) — #{pd['obsnum']}", font=dict(size=12)),
        margin=dict(l=10, r=10, b=30, t=36),
    )
    return fig


def _make_accel_fig(pd: dict, frame: str) -> go.Figure:
    az, el, xt, yt = _frame_coords(pd, frame)
    _, _, acc_arcsec = _vel_acc(pd, frame)
    minlen = min(len(az), len(acc_arcsec))
    fig = px.scatter(x=az[:minlen], y=el[:minlen], color=acc_arcsec[:minlen],
                     color_continuous_scale="thermal",
                     labels={"x": xt, "y": yt, "color": "Accel [arcsec/s²]"})
    fig.update_traces(marker=dict(size=3))
    fig.update_coloraxes(colorbar=dict(title="Accel<br>[arcsec/s²]", thickness=12, len=0.8))
    fig.update_layout(
        height=320, plot_bgcolor="white",
        xaxis=_xy_axis_style(), yaxis=_xy_axis_style(),
        title=dict(text=f"Acceleration ({frame}) — #{pd['obsnum']}", font=dict(size=12)),
        margin=dict(l=10, r=10, b=30, t=36),
    )
    return fig


def _make_az_ts_fig(pd: dict) -> go.Figure:
    t = np.array(pd["tel_time"]); t = t - t[0]
    az = _rad2deg(np.array(pd["tel_az_act"]))
    az_err = np.array(pd["tel_az_act"]) - np.array(pd["tel_az_des"])
    el_err = np.array(pd["tel_el_act"]) - np.array(pd["tel_el_des"])
    error_arcsec = _rad2arcsec(np.sqrt(az_err**2 + el_err**2))
    minlen = min(len(t), len(az), len(error_arcsec))
    fig = px.scatter(x=t[:minlen], y=az[:minlen], color=error_arcsec[:minlen],
                     color_continuous_scale="Viridis",
                     labels={"x": "Time [s]", "y": "TelAzAct [deg]", "color": "Error [arcsec]"})
    fig.update_traces(marker=dict(size=3))
    fig.update_coloraxes(colorbar=dict(title="Error<br>[arcsec]", thickness=12, len=0.8))
    fig.update_layout(
        height=260, plot_bgcolor="white",
        xaxis=_xy_axis_style(), yaxis=_xy_axis_style(),
        title=dict(text=f"Az vs Time — #{pd['obsnum']}", font=dict(size=12)),
        margin=dict(l=10, r=10, b=30, t=36),
    )
    return fig


def _make_el_ts_fig(pd: dict) -> go.Figure:
    t = np.array(pd["tel_time"]); t = t - t[0]
    el = _rad2deg(np.array(pd["tel_el_act"]))
    az_err = np.array(pd["tel_az_act"]) - np.array(pd["tel_az_des"])
    el_err = np.array(pd["tel_el_act"]) - np.array(pd["tel_el_des"])
    error_arcsec = _rad2arcsec(np.sqrt(az_err**2 + el_err**2))
    minlen = min(len(t), len(el), len(error_arcsec))
    fig = px.scatter(x=t[:minlen], y=el[:minlen], color=error_arcsec[:minlen],
                     color_continuous_scale="Viridis",
                     labels={"x": "Time [s]", "y": "TelElAct [deg]", "color": "Error [arcsec]"})
    fig.update_traces(marker=dict(size=3))
    fig.update_coloraxes(colorbar=dict(title="Error<br>[arcsec]", thickness=12, len=0.8))
    fig.update_layout(
        height=260, plot_bgcolor="white",
        xaxis=_xy_axis_style(), yaxis=_xy_axis_style(),
        title=dict(text=f"El vs Time — #{pd['obsnum']}", font=dict(size=12)),
        margin=dict(l=10, r=10, b=30, t=36),
    )
    return fig


# ── Layout helper ─────────────────────────────────────────────────────────────


def _hdr_entry(parent, label: str):
    """Labelled key-value row; returns the value Text component."""
    row = parent.child[dmc.Group](gap=4, wrap="nowrap", mb=2)
    row.child[dmc.Text](label + ":", size="xs", c="dimmed", fw=600, style={"minWidth": 90})
    v = row.child[dmc.Text]("—", size="xs", c="dark", style={"fontFamily": "monospace"})
    return v


# ── Page ──────────────────────────────────────────────────────────────────────


class TelViewerPage(Template):
    """Telescope trajectory & header viewer page.

    Driven by URL ``?obsspec=<master>-<obsnum>-<subobsnum>-<scannum>``.
    The portal's "Tel →" action links pass this parameter.

    Parameters
    ----------
    data_root
        Directory containing ``tel_*.nc`` files.  Falls back to
        ``run/data_lmt/tel/`` (real data), then
        ``tolteca_ref_data/data_lmt/tel/`` (ref data) when ``None``.
    """

    def __init__(self, data_root: Path | str | None = None) -> None:
        super().__init__()
        self._data_root = Path(data_root) if data_root else None

        provider = self.child[dmc.MantineProvider]()
        root = provider.child[dmc.Stack](gap=0, style={"minHeight": "100vh"})

        # URL location
        self._location = root.child[dcc.Location](id="tel-url", refresh=False)
        # Stores: plot data (large arrays) + header (small scalars)
        self._plot_store = root.child[dcc.Store](id="tel-plot-store", data=None)
        self._hdr_store = root.child[dcc.Store](id="tel-hdr-store", data=None)

        # ── Top bar ───────────────────────────────────────────────────────
        hdr_bar = root.child[dmc.Paper](
            withBorder=True, shadow="none",
            style={"borderLeft": "none", "borderRight": "none", "borderTop": "none"},
        )
        hdr_row = hdr_bar.child[dmc.Group](px="md", py="xs", justify="space-between")
        hdr_row.child[dmc.Title]("TolTEC Telescope Viewer", order=4)
        hdr_row.child[dmc.Anchor]("→ Portal", href="/", size="sm", c="blue")

        container = root.child[html.Div](style={"padding": "8px 16px", "width": "100%"})

        # ── Controls strip ─────────────────────────────────────────────────
        ctrl_paper = container.child[dmc.Paper](
            withBorder=True, p="sm", mb="sm", radius="sm",
        )
        ctrl_row = ctrl_paper.child[dmc.Group](gap="lg", align="flex-end", wrap="wrap")

        # Obsspec badge: shows current obsspec from URL (read-only)
        obs_col = ctrl_row.child[dmc.Stack](gap=2)
        obs_col.child[dmc.Text]("Observation", size="xs", c="dimmed", fw=600)
        self._obsspec_badge = obs_col.child[dmc.Text](
            "—",
            size="sm", fw=700, c="dark",
            style={"fontFamily": "monospace", "minWidth": 200},
        )

        # Tel file resolved path (small, gray)
        file_col = ctrl_row.child[dmc.Stack](gap=2)
        file_col.child[dmc.Text]("Tel file", size="xs", c="dimmed", fw=600)
        self._file_text = file_col.child[dmc.Text](
            "—", size="xs", c="dimmed",
            style={"fontFamily": "monospace", "maxWidth": 400,
                   "overflow": "hidden", "textOverflow": "ellipsis",
                   "whiteSpace": "nowrap"},
        )

        # Downsample
        ds_col = ctrl_row.child[dmc.Stack](gap=2)
        ds_col.child[dmc.Text]("Downsample", size="xs", c="dimmed", fw=600)
        self._downsample = ds_col.child[dmc.NumberInput](
            value=10, min=1, max=100, step=1, w=80, size="xs",
        )

        # Frame toggle
        frame_col = ctrl_row.child[dmc.Stack](gap=2)
        frame_col.child[dmc.Text]("Frame", size="xs", c="dimmed", fw=600)
        self._frame_ctrl = frame_col.child[dmc.SegmentedControl](
            data=[
                {"label": "Source", "value": "source"},
                {"label": "Telescope", "value": "telescope"},
            ],
            value="source", size="xs",
        )

        # Status (sample count, duration)
        self._status_text = ctrl_row.child[dmc.Text]("", size="xs", c="dimmed")

        # ── Main content ──────────────────────────────────────────────────
        content_row = container.child[dmc.Group](
            gap="sm", align="flex-start", wrap="nowrap",
        )

        # ── Header panel ──────────────────────────────────────────────────
        hdr_panel = content_row.child[dmc.Stack](
            gap=4, style={"minWidth": 240, "maxWidth": 240},
        )

        obs_box = hdr_panel.child[dmc.Paper](
            withBorder=True, p="xs", radius="sm",
            style={"background": "#e7f5ff", "borderColor": "#74c0fc"},
        )
        obs_box.child[dmc.Text]("Observation", size="xs", fw=700, c="blue", mb=4)
        self._v_obsnum  = _hdr_entry(obs_box, "ObsNum")
        self._v_source  = _hdr_entry(obs_box, "Source")
        self._v_ra      = _hdr_entry(obs_box, "RA")
        self._v_dec     = _hdr_entry(obs_box, "Dec")
        self._v_obspgm  = _hdr_entry(obs_box, "ObsPgm")
        self._v_obsgoal = _hdr_entry(obs_box, "Goal")
        self._v_proj    = _hdr_entry(obs_box, "Project")

        sky_box = hdr_panel.child[dmc.Paper](withBorder=True, p="xs", radius="sm")
        sky_box.child[dmc.Text]("Sky", size="xs", fw=700, c="teal", mb=4)
        self._v_azreq = _hdr_entry(sky_box, "Az req")
        self._v_elreq = _hdr_entry(sky_box, "El req")
        self._v_azoff = _hdr_entry(sky_box, "Az off")
        self._v_eloff = _hdr_entry(sky_box, "El off")

        rt_box = hdr_panel.child[dmc.Paper](withBorder=True, p="xs", radius="sm")
        rt_box.child[dmc.Text]("Radiometer / Tel", size="xs", fw=700, c="orange", mb=4)
        self._v_tau   = _hdr_entry(rt_box, "Tau")
        self._v_crane = _hdr_entry(rt_box, "Crane in beam")

        m_box = hdr_panel.child[dmc.Paper](withBorder=True, p="xs", radius="sm")
        m_box.child[dmc.Text]("M1 / M2 / M3", size="xs", fw=700, c="grape", mb=4)
        self._v_zernike = _hdr_entry(m_box, "M1 ZernikeC")
        self._v_m2z     = _hdr_entry(m_box, "M2 ZReq")
        self._v_m2alive = _hdr_entry(m_box, "M2 Alive")
        self._v_m3alive = _hdr_entry(m_box, "M3 Alive")
        self._v_m3fault = _hdr_entry(m_box, "M3 Fault")

        # ── Plots ─────────────────────────────────────────────────────────
        plots_col = content_row.child[dmc.Stack](gap="xs", style={"flex": "1", "minWidth": 0})

        row1 = plots_col.child[dmc.Group](gap="xs", wrap="nowrap", align="flex-start")
        self._traj_graph = row1.child[dcc.Graph](
            figure=_empty_fig("Trajectory"), style={"flex": "1"},
            config={"displayModeBar": False},
        )
        self._vel_graph = row1.child[dcc.Graph](
            figure=_empty_fig("Velocity"), style={"flex": "1"},
            config={"displayModeBar": False},
        )

        row2 = plots_col.child[dmc.Group](gap="xs", wrap="nowrap", align="flex-start")
        self._err_graph = row2.child[dcc.Graph](
            figure=_empty_fig("Error"), style={"flex": "1"},
            config={"displayModeBar": False},
        )
        self._acc_graph = row2.child[dcc.Graph](
            figure=_empty_fig("Acceleration"), style={"flex": "1"},
            config={"displayModeBar": False},
        )

        ts_row = plots_col.child[dmc.Group](gap="xs", wrap="nowrap", align="flex-start")
        self._az_ts_graph = ts_row.child[dcc.Graph](
            figure=_empty_fig("Az vs Time"), style={"flex": "1"},
            config={"displayModeBar": False},
        )
        self._el_ts_graph = ts_row.child[dcc.Graph](
            figure=_empty_fig("El vs Time"), style={"flex": "1"},
            config={"displayModeBar": False},
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def setup_callbacks(self, app) -> None:
        """Register all Dash callbacks for the tel viewer."""

        data_root = self._data_root

        # ── L1: URL obsspec → resolve file → load stores ───────────────────
        @app.callback(
            Output(self._plot_store(), "data"),
            Output(self._hdr_store(), "data"),
            Output(self._obsspec_badge(), "children"),
            Output(self._file_text(), "children"),
            Output(self._status_text(), "children"),
            Input(self._location(), "search"),
            Input(self._downsample(), "value"),
        )
        def _load_from_url(search: str | None, downsample: int | None):
            """Resolve obsspec URL param → find tel file → load NC data."""
            qs = parse_qs((search or "").lstrip("?"))
            quartet_raw = qs.get("quartet", [None])[0]

            if not quartet_raw or data_root is None:
                msg = "No file" if data_root is None else "No quartet in URL"
                return None, None, "—", msg, ""

            parsed = _parse_obsspec(quartet_raw)
            if parsed is None:
                return None, None, quartet_raw, "Invalid quartet", "Parse error"

            _, obsnum, subobsnum, scannum = parsed
            tel_path = _find_tel_file(data_root, obsnum, subobsnum, scannum)
            if tel_path is None:
                return (
                    None, None, quartet_raw,
                    f"tel file not found for obsnum={obsnum}",
                    "File not found",
                )

            ds = max(1, int(downsample or 10))
            try:
                hdr, pd = _load_tel_file(tel_path, ds)
            except Exception as exc:
                return None, None, obsspec_raw, Path(tel_path).name, f"Error: {exc}"

            n_pts = len(pd["tel_time"])
            dur = round(pd["tel_time"][-1] - pd["tel_time"][0], 1) if n_pts > 1 else 0
            status = f"{n_pts} samples · {dur}s"
            return pd, hdr, quartet_raw, Path(tel_path).name, status

        # ── L2: plot store + frame → all 6 figures ─────────────────────────
        @app.callback(
            Output(self._traj_graph(), "figure"),
            Output(self._vel_graph(), "figure"),
            Output(self._err_graph(), "figure"),
            Output(self._acc_graph(), "figure"),
            Output(self._az_ts_graph(), "figure"),
            Output(self._el_ts_graph(), "figure"),
            Input(self._plot_store(), "data"),
            Input(self._frame_ctrl(), "value"),
        )
        def _update_plots(pd: dict | None, frame: str):
            if pd is None:
                e = _empty_fig()
                return e, e, e, e, _empty_fig(), _empty_fig()
            pd = {k: np.array(v) if isinstance(v, list) else v for k, v in pd.items()}
            return (
                _make_trajectory_fig(pd, frame),
                _make_velocity_fig(pd, frame),
                _make_error_fig(pd, frame),
                _make_accel_fig(pd, frame),
                _make_az_ts_fig(pd),
                _make_el_ts_fig(pd),
            )

        # ── L3: header store → header panel values ─────────────────────────
        @app.callback(
            Output(self._v_obsnum(), "children"),
            Output(self._v_source(), "children"),
            Output(self._v_ra(), "children"),
            Output(self._v_dec(), "children"),
            Output(self._v_obspgm(), "children"),
            Output(self._v_obsgoal(), "children"),
            Output(self._v_proj(), "children"),
            Output(self._v_azreq(), "children"),
            Output(self._v_elreq(), "children"),
            Output(self._v_azoff(), "children"),
            Output(self._v_eloff(), "children"),
            Output(self._v_tau(), "children"),
            Output(self._v_crane(), "children"),
            Output(self._v_zernike(), "children"),
            Output(self._v_m2z(), "children"),
            Output(self._v_m2alive(), "children"),
            Output(self._v_m3alive(), "children"),
            Output(self._v_m3fault(), "children"),
            Input(self._hdr_store(), "data"),
        )
        def _update_header(hdr: dict | None):
            dash = "—"
            if hdr is None:
                return (dash,) * 18
            crane = "YES !!!" if hdr.get("crane_in_beam") else "No"
            return (
                str(hdr.get("obsnum", dash)),
                hdr.get("source_name") or dash,
                f"{hdr.get('ra_deg', 0):.5f}°",
                f"{hdr.get('dec_deg', 0):.5f}°",
                hdr.get("obs_pgm") or dash,
                hdr.get("obs_goal") or dash,
                hdr.get("project_id") or dash,
                f"{hdr.get('az_req_deg', 0):.3f}°",
                f"{hdr.get('el_req_deg', 0):.3f}°",
                f"{hdr.get('az_off_arcsec', 0):.2f}\"",
                f"{hdr.get('el_off_arcsec', 0):.2f}\"",
                f"{hdr.get('tau', 0):.3f}",
                crane,
                hdr.get("zernike_c") or dash,
                f"{hdr.get('m2_z_req_mm', 0):.3f} mm",
                "Yes" if hdr.get("m2_alive") else "No",
                "Yes" if hdr.get("m3_alive") else "No",
                "YES !!!" if hdr.get("m3_fault") else "No",
            )
