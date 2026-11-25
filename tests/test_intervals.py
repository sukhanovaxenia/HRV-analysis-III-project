import math

import numpy as np
import pytest

from hrv.analysis.intervals import IntervalSeries, time_metrics


def test_filters_use_last_accepted_interval() -> None:
    series = IntervalSeries.from_raw(np.array([250, 1000, 1200, 1080, 2100], dtype=float))
    assert series.values.tolist() == [1000, 1080]
    assert (series.outliers, series.abrupt_changes) == (2, 1)


def test_not_enough_accepted_intervals() -> None:
    with pytest.raises(ValueError, match="два интервала"):
        IntervalSeries.from_raw(np.array([250.0, 900.0]))


def test_time_metrics_by_hand() -> None:
    metrics = time_metrics(np.array([800.0, 850.0, 900.0]))
    assert metrics.mean == 850
    assert metrics.sdnn == 50
    assert metrics.rmssd == 50
    assert metrics.variation_percent == pytest.approx(50 / 850 * 100)
    assert metrics.mode_amplitude == pytest.approx(100 / 3)
    assert metrics.occupied_range == 150
    assert metrics.stress == pytest.approx((100 / 3) / (2 * 975 * 150) * 1_000_000)
    assert metrics.state == "сниженная вариабельность, напряжение"


def test_empty_histogram() -> None:
    metrics = time_metrics(np.array([1500.0, 1500.0]))
    assert metrics.occupied_range == 0
    assert math.isnan(metrics.stress)
