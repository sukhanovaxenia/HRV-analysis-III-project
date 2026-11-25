"""Автокорреляция, спектр и временная шкала кардиоинтервалов."""

from dataclasses import dataclass

import nolds
import numpy as np
from scipy.interpolate import CubicSpline

from .intervals import FloatArray

MAX_ACF_LAG = 128
MS_PER_SECOND = 1000
RESAMPLE_SECONDS = 3
VLF_START = 0.015
LF_START = 0.04
HF_START = 0.15
HF_END = 0.4
MIN_FRACTAL_SAMPLES = 10


def autocorrelation(values: FloatArray) -> FloatArray:
    """Вычисляет нормированную автокорреляцию до сдвига 128 включительно.

    Исходный скрипт вычислял корреляцию для всех сдвигов, хотя использовал
    только сдвиги от 0 до 128. Теперь рассчитываются только нужные сдвиги.
    Для короткого ряда их число ограничивается длиной записи.
    Спектральный тест использует четыре интервала с известной частотой и проверяет
    мощность в диапазоне HF.

    Args:
        values: Очищенные интервалы.

    Returns:
        Доступные сдвиги от нуля. Для постоянного ряда значения не определены.
    """
    centered = values - values.mean()
    energy = float(centered @ centered)
    lag_count = min(values.size, MAX_ACF_LAG + 1)
    if energy == 0:
        return np.full(lag_count, np.nan)
    return np.array(
        [centered[: values.size - lag] @ centered[lag:] / energy for lag in range(lag_count)]
    )


def time_axis(values: FloatArray) -> FloatArray:
    """Возвращает время начала каждого интервала в секундах от начала записи."""
    return np.concatenate(([0.0], np.cumsum(values[:-1]))) / MS_PER_SECOND


def resample(timestamps: FloatArray, values: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Восстанавливает непрерывную кривую пульсограммы по интервалам методом кубических сплайнов.

    Args:
        timestamps: Время начала каждого интервала, в секундах.
        values: Интервалы в миллисекундах.

    Returns:
        Равномерная сетка с шагом три секунды и значения сплайна на ней.
    """
    grid = np.arange(0, timestamps[-1], RESAMPLE_SECONDS)
    return grid, np.asarray(CubicSpline(timestamps, values)(grid))


def ratio(numerator: float, denominator: float) -> float:
    """Возвращает отношение или NaN, если знаменатель равен нулю."""
    return numerator / denominator if denominator else float("nan")


@dataclass
class Spectrum:
    """Выделение участка спектра для графика и мощности по заданным диапазонам."""

    frequencies: FloatArray
    plot_power: FloatArray
    hf: float
    lf: float
    vlf: float
    ulf: float

    @property
    def total(self) -> float:
        """Суммирует мощности диапазонов по правилам эталонного отчёта."""
        return self.hf + self.lf + self.vlf + self.ulf


def spectrum(values: FloatArray) -> Spectrum:
    """Рассчитывает спектр в циклах на интервал.

    Args:
        values: Очищенные интервалы в порядке регистрации.

    Returns:
        Данные графика и мощности. Граница 0.015 входит в ULF и VLF по эталону.
    """
    normalized = (values - values.mean()) / values.max()
    frequencies = np.fft.rfftfreq(values.size)
    power = np.abs(np.fft.rfft(normalized)) ** 2
    plot_size = values.size // 2
    return Spectrum(
        frequencies=frequencies[:plot_size],
        plot_power=2 * power[:plot_size],
        hf=float(power[(frequencies >= HF_START) & (frequencies < HF_END)].sum()),
        lf=float(power[(frequencies >= LF_START) & (frequencies < HF_START)].sum()),
        vlf=float(power[(frequencies >= VLF_START) & (frequencies < LF_START)].sum()),
        ulf=float(power[frequencies <= VLF_START].sum()),
    )


def fractal_indices(values: FloatArray) -> tuple[float, float]:
    """Возвращает показатель Херста и DFA для отчёта.

    Для рядов короче десяти точек и постоянных рядов возвращает два NaN.
    Линейная аппроксимация воспроизводит расчёт в зависимостях от заданных диапазонов.
    """
    if values.size < MIN_FRACTAL_SAMPLES or np.ptp(values) == 0:
        return float("nan"), float("nan")
    return float(nolds.hurst_rs(values, fit="poly")), float(nolds.dfa(values, fit_exp="poly"))
