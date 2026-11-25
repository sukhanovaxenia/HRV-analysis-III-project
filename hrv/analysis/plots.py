"""Сохранение графиков одного испытуемого в PNG."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from .intervals import HISTOGRAM_EDGES, REFERENCE_MODE_MS, FloatArray
from .signals import Spectrum, resample, time_axis

FIGURE_SIZE = (20, 8)
ACF_SIZE = (60, 30)
SCATTER_LIMIT = 300
AXIS_MIN_MS = 400
AXIS_MAX_MS = 1300
ELLIPSE_WIDTH_MS = 300
ELLIPSE_HEIGHT_MS = 110
ELLIPSE_ANGLE = 45


def _save_scatter(values: FloatArray, path: Path) -> None:
    """Сохраняет диаграмму пар соседних интервалов и эталонный эллипс."""
    segment = values[:SCATTER_LIMIT]
    plt.figure(figsize=FIGURE_SIZE)
    plt.scatter(segment[:-1], segment[1:])
    plt.xlim(AXIS_MIN_MS, AXIS_MAX_MS)
    plt.ylim(AXIS_MIN_MS, AXIS_MAX_MS)
    diagonal = range(AXIS_MIN_MS, AXIS_MAX_MS)
    plt.plot(diagonal, diagonal)
    ellipse = Ellipse(
        (REFERENCE_MODE_MS, REFERENCE_MODE_MS),
        width=ELLIPSE_WIDTH_MS,
        height=ELLIPSE_HEIGHT_MS,
        angle=ELLIPSE_ANGLE,
        edgecolor="red",
        facecolor="none",
        linewidth=3,
    )
    plt.gca().add_patch(ellipse)
    plt.savefig(path)
    plt.close()


def _save_time_plot(times: FloatArray, values: FloatArray, title: str, path: Path) -> None:
    """Сохраняет ряд интервалов на исходной или равномерной временной шкале."""
    plt.figure(figsize=FIGURE_SIZE)
    plt.scatter(times, values)
    plt.plot(times, values)
    plt.title(title)
    plt.savefig(path)
    plt.close()


def save_plots(
    output: Path, subject: str, values: FloatArray, acf: FloatArray, bands: Spectrum
) -> None:
    """Сохраняет шесть графиков для одной записи из данных.

    Args:
        output: Каталог для PNG.
        subject: Имя испытуемого для имени файла.
        values: Очищенные интервалы в миллисекундах.
        acf: Автокорреляция для доступных сдвигов.
        bands: Спектр того же ряда.
    """
    plt.figure(figsize=FIGURE_SIZE)
    plt.hist(values, bins=HISTOGRAM_EDGES)
    plt.savefig(output / f"hrv_hist_{subject}.png")
    plt.close()

    plt.figure(figsize=ACF_SIZE)
    plt.stem(range(len(acf)), acf)
    plt.savefig(output / f"hrv_acf_{subject}.png")
    plt.close()

    _save_scatter(values, output / f"hrv_scatter_{subject}.png")

    times = time_axis(values)
    _save_time_plot(times, values, "Кардиоритмограмма", output / f"hrv_rhythmogram_{subject}.png")
    grid, interpolated = resample(times, values)
    _save_time_plot(grid, interpolated, "Пульсограмма", output / f"hrv_pulsogram_{subject}.png")

    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(bands.frequencies, bands.plot_power)
    plt.savefig(output / f"hrv_spectrum_{subject}.png")
    plt.close()
