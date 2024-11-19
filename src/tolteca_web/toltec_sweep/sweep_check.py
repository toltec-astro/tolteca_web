from enum import IntFlag, auto

import astropy.units as u
import numpy as np
from pydantic import BaseModel, Field
from scipy.ndimage import median_filter
from tollan.utils.log import logger, timeit

from tolteca_kidsproc.kidsdata import MultiSweep


class WorkflowStep(BaseModel):
    """A base class for workflow steps."""

    @classmethod
    def get_or_create_workflow_context(cls, data, ns_key=None):
        """Get or create workflow context."""
        data.meta.setdefault("workflow_context", {})
        wf_ctx = data.meta["workflow_context"]
        ns_key = ns_key or cls.__name__
        if isinstance(ns_key, type(WorkflowStep)):
            ns_key = ns_key.__name__
        elif isinstance(ns_key, str):
            pass
        else:
            raise TypeError("invalid workflow context namespace key type.")
        wf_ctx.setdefault(ns_key, {})
        return wf_ctx[ns_key]


class SweepDataBitMask(IntFlag):
    """A bit mask for sweep data."""

    s21_small_range = auto()
    s21_spike = auto()
    s21_high_rms = auto()
    s21_low_rms = auto()


class Despike(WorkflowStep):
    """Despike config."""

    min_spike_height_frac: float = Field(
        default=0.1,
        description=(
            "The minimum range of spike, measured as fraction to the S21 range."
        ),
    )
    min_S21_range_db: float = Field(
        default=0.1,
        description="The minimum S21 range in dB to find spike.",
    )

    @staticmethod
    def calc_y(swp):
        """Return the proxy value to run the algorithms on."""
        S21 = swp.S21.to_value(u.dimensionless_unscaled)
        return 20.0 * np.log10(np.abs(S21))

    @classmethod
    def _find_spike_S21(cls, swp: MultiSweep, min_spike_height_frac, min_S21_range_db):
        y = cls.calc_y(swp)
        y_med = median_filter(y, size=(1, 5))
        y_range = np.max(y_med, axis=-1) - np.min(y_med, axis=-1)
        s_spike = spike_height_frac = (y - y_med) / y_range[:, np.newaxis]
        m_data_spike = np.abs(s_spike) >= min_spike_height_frac
        m_chan_small_range = y_range < min_S21_range_db
        m_data_small_range = m_chan_small_range[:, np.newaxis]
        bitmask = (SweepDataBitMask.s21_small_range * m_data_small_range) | (
            SweepDataBitMask.s21_spike * m_data_spike
        )
        bitmask_chan = SweepDataBitMask.s21_small_range * m_chan_small_range

        logger.debug(
            (
                "found low signal range channel:"
                f" {m_chan_small_range.sum()}/{m_chan_small_range.size}"
            ),
        )
        logger.debug(f"found spike {m_data_spike.sum()}/{m_data_spike.size}")
        # create spike mask, which is all spikes found in good channel.
        mask = m_data_spike & (~m_data_small_range)
        logger.debug(f"masked spike {mask.sum()}/{mask.size}")
        return locals()

    def find_spike_S21(self, swp: MultiSweep):
        """Find spike in S21."""
        ctx = self.get_or_create_workflow_context(swp)
        subctx = ctx["find_spike_S21"] = self._find_spike_S21(
            swp,
            min_spike_height_frac=self.min_spike_height_frac,
            min_S21_range_db=self.min_S21_range_db,
        )
        return subctx

    @classmethod
    def _despike(cls, swp: MultiSweep, min_spike_height_frac, min_S21_range_db):
        ctx_spike = cls._find_spike_S21(
            swp,
            min_spike_height_frac=min_spike_height_frac,
            min_S21_range_db=min_S21_range_db,
        )
        swp = ctx_spike["swp"]
        spike_mask = ctx_spike["mask"]
        goodmask = ~spike_mask
        fs_Hz = swp.frequency.to_value(u.Hz)
        S21_adu = swp.S21.to_value(u.dimensionless_unscaled).copy()
        for ci in range(fs_Hz.shape[0]):
            m = goodmask[ci]
            swp.S21[ci] = np.interp(fs_Hz[ci], fs_Hz[ci][m], S21_adu[ci][m]) << u.dimensionless_unscaled
        # make despiked y
        y_nospike = cls.calc_y(swp)
        return locals()

    def despike(self, swp: MultiSweep):
        """Apply spike mask on data."""
        ctx = self.get_or_create_workflow_context(swp)
        subctx = ctx["despike"] = self._despike(
            swp,
            min_spike_height_frac=self.min_spike_height_frac,
            min_S21_range_db=self.min_S21_range_db,
        )
        return subctx

    @timeit
    def run(self, swp: MultiSweep):
        """Return sweep data that have spikes identified and interpolated away."""
        ctx = self.get_or_create_workflow_context(swp)
        if ctx.get("is_completed", False):
            return swp
        # run
        ctx_spike = self.find_spike_S21(swp)
        self.despike(swp)
        ctx.update(
            {
                "is_completed": True,
                "config": self.model_dump(),
                "bitmask": ctx_spike["bitmask"],
                "bitmask_chan": ctx_spike["bitmask_chan"],
            },
        )
        return swp


class CheckSweep(WorkflowStep):
    """The sweep checker."""

    S21_rms_high_db: float = Field(
        default=0.2,
        description="Threshold to flag high rms data.",
    )

    S21_rms_low_db: float = Field(
        default=0.001,
        description="Threshold to flag low rms data.",
    )

    def run(self, swp: MultiSweep):
        """Run checking of sweep data."""
        ctx = self.get_or_create_workflow_context(swp)
        if ctx.get("is_completed", False):
            return swp
        ctx_noise = self.check_noise(swp)
        ctx.update(
            {
                "is_completed": True,
                "config": self.model_dump(),
                "bitmask": ctx_noise["bitmask"],
                "bitmask_chan": ctx_noise["bitmask_chan"],
            },
        )
        return swp

    def check_noise(self, swp: MultiSweep):
        """Check noise in sweep."""
        ctx = self.get_or_create_workflow_context(swp)
        subctx = ctx["check_noise"] = self._check_noise(
            swp,
            S21_rms_high_db=self.S21_rms_high_db,
            S21_rms_low_db=self.S21_rms_low_db,
        )
        return subctx

    @classmethod
    def _check_noise(cls, swp, S21_rms_high_db, S21_rms_low_db):
        # get bitmask from despike if exist
        ctx_despike_step = cls.get_or_create_workflow_context(swp, Despike)
        bitmask0 = ctx_despike_step.get("bitmask", None)
        bitmask_chan0 = ctx_despike_step.get("bitmask_chan", None)

        fs = swp.frequency.to_value(u.MHz)
        S21_adu = swp.S21.to_value(u.dimensionless_unscaled)
        S21_rms_adu = swp.uncertainty.quantity.to_value(u.dimensionless_unscaled)
        S21_rms_db = 20 * np.log10(np.e) * np.abs(S21_rms_adu / S21_adu)
        # per-channel mean and std value of S21_rms
        S21_rms_db_mean = np.mean(S21_rms_db, axis=1)
        S21_rms_db_std = np.std(S21_rms_db, axis=1)
        rms_db_med = np.median(S21_rms_db_mean)

        m_chan_high_rms = S21_rms_db_mean > S21_rms_high_db
        m_chan_low_rms = S21_rms_db_mean < S21_rms_low_db
        m_data_high_rms = m_chan_high_rms[:, np.newaxis]
        m_data_low_rms = m_chan_low_rms[:, np.newaxis]
        bitmask = (SweepDataBitMask.s21_high_rms * m_data_high_rms) | (
            SweepDataBitMask.s21_low_rms * m_data_low_rms
        )
        bitmask_chan = (SweepDataBitMask.s21_high_rms * m_chan_high_rms) | (
            SweepDataBitMask.s21_low_rms * m_chan_low_rms
        )

        if bitmask0 is not None:
            bitmask = bitmask | bitmask0

        if bitmask_chan0 is not None:
            bitmask_chan = bitmask_chan | bitmask_chan0

        logger.debug(
            f"found low rms channel: {m_chan_low_rms.sum()}/{m_chan_low_rms.size}",
        )
        logger.debug(
            (
                "found high rms channel:"
                f" {m_chan_high_rms.sum()}/{m_chan_high_rms.size}"
            ),
        )
        # create spike mask, which is all spikes found in good channel.
        m_chan_good = (~m_chan_low_rms) & (~m_chan_high_rms)
        logger.debug(f"found good channels {m_chan_good.sum()}/{m_chan_good.size}")
        return locals()
