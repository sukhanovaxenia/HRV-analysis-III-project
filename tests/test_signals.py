import math

import numpy as np
import pytest

from hrv.analysis.signals import (
    autocorrelation,
    fractal_indices,
    ratio,
    resample,
    spectrum,
    time_axis,
)


def test_autocorrelation_by_hand() -> None:
    assert autocorrelation(np.array([800.0, 900.0, 800.0])) == pytest.approx([1, -2 / 3, 1 / 6])


def test_time_axis_and_linear_interpolation() -> None:
    values = np.array([1000.0, 1100.0, 1200.0, 1300.0])
    assert time_axis(values) == pytest.approx([0, 1, 2.1, 3.3])
    grid, interpolated = resample(np.arange(4, dtype=float) * 2, values)
    assert grid.tolist() == [0, 3]
    assert interpolated == pytest.approx([1000, 1150])


def test_single_frequency_has_expected_band_power() -> None:
    # Один период на четыре интервала, частота 0.25 лежит в HF.
    bands = spectrum(np.array([800.0, 900.0, 800.0, 700.0]))
    assert bands.hf == pytest.approx((200 / 900) ** 2)
    assert bands.total == pytest.approx(bands.hf)
    assert bands.plot_power == pytest.approx([0, 2 * bands.hf])


def test_undefined_metrics() -> None:
    values = np.array([800.0, 800.0])
    assert np.isnan(autocorrelation(values)).all()
    assert math.isnan(ratio(0, 0))
    assert all(math.isnan(index) for index in fractal_indices(values))
