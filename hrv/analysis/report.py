"""Текстовый отчёт и сводная таблица HRV."""

from pathlib import Path

import numpy as np
import pandas as pd

from .intervals import REFERENCE_MODE_MS, TimeMetrics
from .signals import Spectrum, ratio

SUMMARY_COLUMNS = ["subject", "mean", "sdnn", "rmssd", "stress_index", "lf_hf", "hurst", "state"]


def print_time_metrics(metrics: TimeMetrics) -> None:
    """Выводит временные характеристики с точностью, соответствующей контрольному отчёту."""
    print("Среднее значение: ", np.round(metrics.mean, 2))
    print("Среднеквадратическое отклонение SDNN: ", np.round(metrics.sdnn, 2))
    print("Коэффициент вариации: ", np.round(metrics.variation_percent, 2), "%")
    print("Среднеквадратическая разностная характеристика RMSSD: ", np.round(metrics.rmssd, 2))
    print("Амплитуда моды: ", np.round(metrics.mode_amplitude, 4))
    print("Мода: ", REFERENCE_MODE_MS)
    print("Вариационный размах: ", metrics.occupied_range)
    print("Индекс напряжения регуляторных систем: ", np.round(metrics.stress, 2))


def print_spectrum(bands: Spectrum) -> None:
    """Выводит общую мощность, доли диапазонов и их отношения."""
    total = bands.total
    print(np.round(total, 2))
    print("Высокочастотный диапазон: ", np.round(ratio(bands.hf, total), 2))
    print("Низкочастотный диапазон: ", np.round(ratio(bands.lf, total), 2))
    print("Очень низкочастотный диапазон: ", np.round(ratio(bands.vlf, total), 2))
    print("Ультра низкочастотный компонент: ", np.round(ratio(bands.ulf, total), 2))
    print("Индекс централизации: ", np.round(ratio(bands.vlf + bands.lf, bands.hf), 2))
    print("Индекс вагосимпатического взаимодействия: ", np.round(ratio(bands.lf, bands.hf), 2))
    print("Индекс активации подкорковых нервных центров: ", np.round(ratio(bands.lf, bands.vlf), 2))


def write_summary(rows: list[dict[str, str | float]], path: Path) -> None:
    """Записывает CSV и дополнительно выводит итоговую таблицу в stdout.

    Args:
        rows: Сводные показатели по одной записи в каждом словаре.
        path: Путь к выходному CSV.
    """
    summary = pd.DataFrame.from_records(rows, columns=SUMMARY_COLUMNS)
    summary.to_csv(path, index=False)
    print(summary.to_string())
