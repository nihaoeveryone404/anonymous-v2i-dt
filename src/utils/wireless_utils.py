"""Shared wireless-domain conversions and calculations."""

from __future__ import annotations

import numpy as np


def noise_dbm(bandwidth_hz: float, noise_psd_dbm_hz: float = -174.0, noise_figure_db: float = 5.0) -> float:
    return float(noise_psd_dbm_hz + 10.0 * np.log10(bandwidth_hz) + noise_figure_db)


def dbm_to_mw(dbm):
    return np.power(10.0, np.asarray(dbm) / 10.0)


def sinr_and_rate(pr_dbm, bandwidth_hz: float, noise_figure_db: float = 5.0,
                  noise_psd_dbm_hz: float = -174.0, interference_coefficient: float = 1.0):
    power_mw = dbm_to_mw(pr_dbm)
    interference = interference_coefficient * (power_mw.sum(axis=-1, keepdims=True) - power_mw)
    noise_mw = dbm_to_mw(noise_dbm(bandwidth_hz, noise_psd_dbm_hz, noise_figure_db))
    sinr = power_mw / np.maximum(interference + noise_mw, 1e-12)
    rate = bandwidth_hz * np.log2(1.0 + np.maximum(sinr, 0.0))
    return sinr, rate
