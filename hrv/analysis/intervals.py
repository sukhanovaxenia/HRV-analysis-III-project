"""Очистка кардиологических интервалов и расчёт временных показателей."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]  # объявляем новый тип, чтобы удобнее было
# использовать далее
MIN_INTERVAL_MS = 300
MAX_INTERVAL_MS = 2000
MAX_RELATIVE_CHANGE = 0.2
MIN_PAIR_SIZE = 2
HISTOGRAM_STEP_MS = 50
HISTOGRAM_EDGES = range(400, 1300, HISTOGRAM_STEP_MS)
REFERENCE_MODE_MS = 975.0
STRESS_SCALE = 1_000_000
LOW_SDNN_MS = 70
HIGH_SDNN_MS = 100
RELAXATION_THRESHOLD = 30
STRESS_THRESHOLD = 35


@dataclass  # используется dataclass для автоматического создания __init__, __repr__ и __eq__
class IntervalSeries:
    """Прошедшие фильтрацию интервалы и число исключений на каждом шаге."""

    values: FloatArray  # сразу создается init, где помещаются переменные
    outliers: int  # задаются также типы входных данных
    abrupt_changes: int

    @classmethod
    def from_raw(cls, raw: FloatArray) -> "IntervalSeries":
        """Отбирает интервалы по диапазону и скачку от последнего принятого значения.

        Args:
            raw: Интервалы в миллисекундах в порядке регистрации.

        Returns:
            Очищенный ряд со счётчиками для отчёта.

        Raises:
            ValueError: Если осталось меньше двух интервалов.
        """
        in_range = raw[(raw >= MIN_INTERVAL_MS) & (raw <= MAX_INTERVAL_MS)]
        accepted: list[float] = []
        for value in in_range:
            if not accepted or abs(value - accepted[-1]) / accepted[-1] < MAX_RELATIVE_CHANGE:
                accepted.append(float(value))
        if len(accepted) < MIN_PAIR_SIZE:
            raise ValueError("После фильтрации нужно хотя бы два интервала.")
        return cls(np.asarray(accepted), raw.size - in_range.size, in_range.size - len(accepted))


@dataclass
class TimeMetrics:
    """Временные и гистограммные показатели одного испытуемого."""

    mean: float
    sdnn: float
    rmssd: float
    mode_amplitude: float
    occupied_range: int
    stress: float

    @property
    def variation_percent(self) -> float:
        """Возвращает выборочное отклонение в процентах от среднего."""
        return self.sdnn / self.mean * 100

    @property
    def state(self) -> str:
        """Составляет подпись по порогам SDNN и индекса напряжения из задания."""
        if self.sdnn < LOW_SDNN_MS:
            variability = "сниженная вариабельность"
        elif self.sdnn < HIGH_SDNN_MS:
            variability = "нормальная вариабельность"
        else:
            variability = "повышенная вариабельность"
        if self.stress > STRESS_THRESHOLD:
            return variability + ", напряжение"
        if self.stress < RELAXATION_THRESHOLD:
            return variability + ", расслабление"
        return variability


def time_metrics(values: FloatArray) -> TimeMetrics:
    """Вычисляет статистики из очищенного ряда и гистограммы задания.

    Args:
        values: Не менее двух положительных интервалов в миллисекундах.

    Returns:
        Показатели с фиксированной модой из эталона. Пустая гистограмма даёт NaN
        для индекса напряжения.
    """
    counts, _ = np.histogram(values, bins=HISTOGRAM_EDGES)
    amplitude = float(counts.max() / values.size * 100)
    occupied_range = int(np.count_nonzero(counts)) * HISTOGRAM_STEP_MS
    stress = (
        amplitude / (2 * REFERENCE_MODE_MS * occupied_range) * STRESS_SCALE
        if occupied_range
        else float("nan")
    )
    return TimeMetrics(
        mean=float(values.mean()),
        sdnn=float(values.std(ddof=1)),
        rmssd=float(np.sqrt(np.mean(np.diff(values) ** 2))),
        mode_amplitude=amplitude,
        occupied_range=occupied_range,
        stress=stress,
    )
