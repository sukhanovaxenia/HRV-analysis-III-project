import pandas as pd
import numpy as np
import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.interpolate import CubicSpline
import nolds

files = [
    "./hrv/data/subject_01.csv",
    "./hrv/data/subject_02.csv",
    "./hrv/data/subject_03.csv",
    "./hrv/data/subject_04.csv",
]

names = []
means = []
sdnns = []
rmssds = []
stress = []
lfhf = []
hurst = []
states = []

for f in files:
    print("=====", f, "=====")
    try:
        raw = pd.read_csv(f, header=None)[0]
    except:
        print("Не удалось прочитать файл", f)
        continue
    print("Прочитано интервалов: ", raw.size)

    clean = []
    for v in raw:
        if v >= 300 and v <= 2000:
            clean.append(v)
    print("Удалено выбросов: ", raw.size - len(clean))

    filtered = [clean[0]]
    for i in range(1, len(clean)):
        if abs(clean[i] - filtered[-1]) / filtered[-1] < 0.2:
            filtered.append(clean[i])
    print("Удалено экстрасистол: ", len(clean) - len(filtered))

    raw_data = pd.DataFrame(filtered)

    M = raw_data[0].mean()
    print("Среднее значение: ", np.round(M, 2))

    SDNN = 0
    for i in raw_data[0]:
        SDNN += math.pow(i - M, 2)

    SDNN = math.sqrt(SDNN / (raw_data[0].size - 1))
    print("Среднеквадратическое отклонение SDNN: ", np.round(SDNN, 2))

    CV = SDNN / M * 100
    print("Коэффициент вариации: ", np.round(CV, 2), "%")

    RMSSD = 0
    for i in range(raw_data[0].size - 1):
        RMSSD += math.pow(raw_data[0][i] - raw_data[0][i + 1], 2)

    RMSSD = math.sqrt(RMSSD / (raw_data[0].size - 1))
    print("среднеквадратическая разностная характеристика RMSSD: ", np.round(RMSSD, 2))

    plt.figure(figsize=(20, 8))
    data = plt.hist(raw_data[0], bins=range(400, 1300, 50))
    plt.savefig("output/hrv_hist_" + f[-14:-4] + ".png")
    plt.close()

    Am0 = max(data[0]) / raw_data[0].size * 100
    print("Амплитуда моды: ", np.round(Am0, 4))

    M0 = (950 + 1000) / 2.0
    print("Мода: ", M0)

    MxDMn = 0

    for interval in data[0]:
        if interval != 0:
            MxDMn += 50

    print("Вариационный размах: ", np.round(MxDMn, 2))

    IN = Am0 / (2 * M0 * MxDMn) * 1000000
    print("Индекс напряжения регуляторных систем: ", np.round(IN, 2))

    fig, ax = plt.subplots(1, figsize=(60, 30))
    x = raw_data[0] - raw_data[0].mean()
    acf = np.correlate(x, x, mode="full")[x.size - 1 :] / np.sum(x**2)
    ax.stem(range(129), acf[:129])
    plt.savefig("output/hrv_acf_" + f[-14:-4] + ".png")
    plt.close()
    print("Автокорреляция, сдвиг 1: ", np.round(acf[1], 2))

    x_data = raw_data.iloc[:299]
    y_data = raw_data.iloc[1:300]

    plt.figure(figsize=(20, 8))
    plt.xlim(400, 1300)
    plt.ylim(400, 1300)
    plt.scatter(x_data, y_data)

    x1 = range(400, 1300, 1)
    y1 = range(400, 1300, 1)
    plt.plot(x1, y1)

    ax = plt.gca()

    center = np.array([975, 975])

    ellipse = Ellipse(
        xy=center, width=300, height=110, edgecolor="r", fc="None", ls="solid", lw=3, angle=45
    )

    ax.add_patch(ellipse)
    plt.savefig("output/hrv_scatter_" + f[-14:-4] + ".png")
    plt.close()

    y = raw_data[0]

    x = []
    currentTimeStamp = 0

    for record in y:
        x.append(currentTimeStamp / 1000)
        currentTimeStamp += record

    plt.figure(figsize=(20, 8))
    plt.scatter(x, y)
    plt.plot(x, y)
    plt.title("Кардиоритмограмма")
    plt.savefig("output/hrv_rhythmogram_" + f[-14:-4] + ".png")
    plt.close()

    xnew = np.arange(0, x[len(x) - 1], 3)

    f2 = CubicSpline(x, y)
    ynew = f2(xnew)

    plt.figure(figsize=(20, 8))
    plt.plot(xnew, ynew)
    plt.scatter(xnew, ynew)
    plt.title("Пульсограмма")
    plt.savefig("output/hrv_pulsogram_" + f[-14:-4] + ".png")
    plt.close()

    fData = np.fft.fft((y - y.mean()) / y.max())
    frequancy = np.fft.fftfreq(y.shape[-1])

    cuttedData = frequancy[0 : len(frequancy) // 2]
    fDataFiltered = fData[0 : len(fData) // 2]

    fDataNormalised = 2 * np.abs(np.power(fDataFiltered, 2))

    plt.figure(figsize=(20, 8))
    plt.plot(cuttedData, fDataNormalised)
    plt.savefig("output/hrv_spectrum_" + f[-14:-4] + ".png")
    plt.close()

    data = np.abs(np.power(fData, 2))

    hf = 0
    for i in range(0, len(frequancy)):
        if frequancy[i] >= 0.15 and frequancy[i] < 0.4:
            hf += data[i]

    lf = 0
    for i in range(0, len(frequancy)):
        if frequancy[i] >= 0.04 and frequancy[i] < 0.15:
            lf += data[i]

    vlf = 0
    for i in range(0, len(frequancy)):
        if frequancy[i] >= 0.015 and frequancy[i] < 0.04:
            vlf += data[i]

    ulf = 0
    for i in range(0, len(frequancy)):
        if frequancy[i] >= 0 and frequancy[i] <= 0.015:
            ulf += data[i]

    all = hf + lf + vlf + ulf
    print(np.round(all, 2))

    print("Высокочастотный диапазон: ", np.round(hf / all, 2))
    print("Низкочастотный диапазон: ", np.round(lf / all, 2))
    print("Очень низкочастотный диапазон: ", np.round(vlf / all, 2))
    print("Ультра низкочастотный компонент: ", np.round(ulf / all, 2))

    print("Индекс централизации: ", np.round((vlf + lf) / hf, 2))
    print("Индекс вагосимпатического взаимодействия: ", np.round(lf / hf, 2))
    print("Индекс активации подкорковых нервных центров: ", np.round(lf / vlf, 2))

    herstIndex = nolds.hurst_rs(raw_data[0])
    print("Индекс Херста: ", np.round(herstIndex, 2))

    fluctuation = nolds.dfa(raw_data[0])

    print("Показатель флуктуации a", np.round(fluctuation, 2))

    if SDNN < 70:
        state = "сниженная вариабельность"
    elif SDNN < 100:
        state = "нормальная вариабельность"
    else:
        state = "повышенная вариабельность"

    if IN > 35:
        state = state + ", напряжение"
    elif IN < 30:
        state = state + ", расслабление"
    print("Состояние: ", state)

    names.append(f[-14:-4])
    means.append(np.round(M, 2))
    sdnns.append(np.round(SDNN, 2))
    rmssds.append(np.round(RMSSD, 2))
    stress.append(np.round(IN, 2))
    lfhf.append(np.round(lf / hf, 2))
    hurst.append(np.round(herstIndex, 2))
    states.append(state)

summary = pd.DataFrame(
    {
        "subject": names,
        "mean": means,
        "sdnn": sdnns,
        "rmssd": rmssds,
        "stress_index": stress,
        "lf_hf": lfhf,
        "hurst": hurst,
        "state": states,
    }
)
summary.to_csv("output/hrv_summary.csv", index=False)
print("=====  Сводка  =====")
print(summary.to_string())
