"""Lightweight accessor for zarr-format sweep datasets produced by ingest_obs.py.

Mirrors the ``get_chan_axis_data`` / ``get_sweep_axis_data`` API of
``ToltecKidsIOMapper`` but targets the simplified zarr schema used by this
web viewer, which stores:

* ``I``, ``Q`` — float32, dims ``(chan, sample)``
* ``tone_freq``  — float64, dim ``(chan,)``, Hz; tone offset from LO center
* coord ``lo_freq`` — float64, dim ``(sample,)``, Hz; LO per sweep step
* attr ``lo_center_freq_hz`` — scalar float; nominal LO center frequency

Usage
-----
::

    acc = ZarrSweepDataset(ds_c)          # ds_c already sliced by chan
    chan = acc.get_chan_axis_data()        # f_tone_hz, f_chan_hz per channel
    sweep = acc.get_sweep_axis_data()     # f_lo_hz, f_sweep_hz per step
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["ZarrSweepDataset"]


class ZarrSweepDataset:
    """Accessor for a zarr-opened sweep :class:`xarray.Dataset`.

    Parameters
    ----------
    ds
        Dataset opened (and optionally ``chan``-sliced) from a zarr store
        via :func:`~tolteca_web.obs.ObsDataService.get_obs_data`.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    @property
    def lo_center_hz(self) -> float:
        """Nominal LO center frequency in Hz."""
        return float(self._ds.attrs.get("lo_center_freq_hz", 0.0))

    def get_chan_axis_data(self) -> dict[str, np.ndarray]:
        """Channel-axis frequency arrays.

        Returns
        -------
        dict
            ``f_tone_hz``
                shape ``(n_chans,)`` — tone offset from LO center, Hz.
            ``f_chan_hz``
                shape ``(n_chans,)`` — absolute channel centre frequency, Hz.
        """
        f_tone = self._ds["tone_freq"].values.copy()
        f_chan = self.lo_center_hz + f_tone
        return {"f_tone_hz": f_tone, "f_chan_hz": f_chan}

    def get_sweep_axis_data(self) -> dict[str, float | np.ndarray]:
        """Sweep-axis (LO-scan) frequency arrays.

        Returns
        -------
        dict
            ``f_lo_hz``
                shape ``(n_samples,)`` — absolute LO frequency per step, Hz.
            ``f_sweep_hz``
                shape ``(n_samples,)`` — LO offset from LO center per step, Hz.
            ``lo_center_hz``
                scalar — LO center frequency, Hz.
        """
        f_lo = self._ds.coords["lo_freq"].values.copy()
        f_sweep = f_lo - self.lo_center_hz
        return {
            "f_lo_hz": f_lo,
            "f_sweep_hz": f_sweep,
            "lo_center_hz": self.lo_center_hz,
        }
