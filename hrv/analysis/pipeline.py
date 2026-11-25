"""Чтение записей, запуск анализа и сохранение результатов."""

# В итоговом скрипте пайплайна создание отдельного общего класса необязательно,
# так как классы из предыдущих этапов уже описывают очищенные интервалы и
# рассчитанные показатели.
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .intervals import FloatArray, IntervalSeries, time_metrics
from .plots import save_plots
from .report import print_spectrum, print_time_metrics, write_summary
from .signals import autocorrelation, fractal_indices, ratio, spectrum

logger = logging.getLogger(__name__)

DATA_DIR = Path("hrv/data")
OUTPUT_DIR = Path("output")
SUBJECTS = ["subject_01", "subject_02", "subject_03", "subject_04"]


def process_subject(subject: str, values: FloatArray) -> dict[str, str | float]:
    """Анализирует одну запись и сохраняет её графики.

    Args:
        subject: Имя испытуемого в таблице и названиях PNG.
        values: Интервалы после фильтрации, в миллисекундах.

    Returns:
        Строка сводной таблицы для этого испытуемого.
    """
    metrics = time_metrics(values)
    acf = autocorrelation(values)
    bands = spectrum(values)
    hurst, fluctuation = fractal_indices(values)

    print_time_metrics(metrics)
    print("Автокорреляция, сдвиг 1: ", np.round(acf[1], 2))
    print_spectrum(bands)
    print("Индекс Херста: ", np.round(hurst, 2))
    print("Показатель флуктуации a", np.round(fluctuation, 2))
    print("Состояние: ", metrics.state)
    save_plots(OUTPUT_DIR, subject, values, acf, bands)

    return {
        "subject": subject,
        "mean": float(np.round(metrics.mean, 2)),
        "sdnn": float(np.round(metrics.sdnn, 2)),
        "rmssd": float(np.round(metrics.rmssd, 2)),
        "stress_index": float(np.round(metrics.stress, 2)),
        "lf_hf": float(np.round(ratio(bands.lf, bands.hf), 2)),
        "hurst": float(np.round(hurst, 2)),
        "state": metrics.state,
    }


def main() -> None:
    """Обрабатывает каждый доступный CSV и записывает общую таблицу."""
    plt.switch_backend("Agg")
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = []
    for subject in SUBJECTS:
        input_file_path = DATA_DIR / f"{subject}.csv"
        logger.info("Чтение файла ./%s", input_file_path)
        try:
            table = pd.read_csv(input_file_path, header=None, usecols=[0])
            raw = table[0].to_numpy(dtype=float)
        except (OSError, ValueError) as error:
            logger.warning("Не удалось прочитать файл ./%s: %s", input_file_path, error)
            continue

        try:
            series = IntervalSeries.from_raw(raw)
        except ValueError as error:
            logger.warning("Файл невалиден ./%s: %s", input_file_path, error)
            continue

        print(f"===== Анализируем данные из таблицы ./{input_file_path} =====")
        print("Прочитано интервалов: ", len(raw))
        print("Удалено выбросов: ", series.outliers)
        print("Удалены внеочередные сокращениея сердца (экстрасистолы): ", series.abrupt_changes)
        row = process_subject(subject, series.values)
        rows.append(row)

    output_file_path = OUTPUT_DIR / "hrv_summary.csv"
    write_summary(rows, output_file_path)
    logger.info("Сводная таблица сохранена в %s (записей: %d)", output_file_path, len(rows))
