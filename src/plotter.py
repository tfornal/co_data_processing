from dateutil import tz
from functools import wraps
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate

from utc_converter import get_time_from_UTC

date = "20230215"
element = "C"
discharge_nr = 21
time_interval = [0, 100]


def load_df():
    path = (
        pathlib.Path(__file__).parent.parent.resolve()
        / "data"
        / "time_evolutions"
        / f"{element}"
        / f"{date}"
        / f"{element}-{date}-exp_{discharge_nr}-230215_AM113113_313.csv"
    )
    df = pd.read_csv(path, sep="\t")
    return df


df2 = load_df()
print(df2)


def plot():
    fig, ax = plt.subplots()
    from utc_converter import get_time_from_UTC

    time = list(map(get_time_from_UTC, df2["utc_timestamps"]))
    x = df2["time"]
    y = df2[f"{element}-intensity"]
    ax.plot(x, y, color="blue")
    x = list(range(len(time)))

    ### warunek ze jesli dla wielu minut, albo powyzej x sekund to dzielnik wiekszy
    ### czyli time interval / dt - sprawdzic ile pkt i dostsowowac warunki
    x_labels = [
        date.strftime("%H:%M:%S") if date.microsecond % 10_000_000 == 0 else ""
        for date in time
    ]
    ax.set_xticks(np.arange(len(x)), labels=x_labels)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_title = (
        f"Evolution of the C/O monitor signal intensity. \n {date}.{discharge_nr}"
    )
    plt.tight_layout()

    plt.show()


# def using_pd():
#     ax = df2.plot(
#         x="discharge_time",
#         y=f"{element}-intensity",
#         label=f"{element}",
#         linewidth=0.7,
#         color="blue",
#         title=f"Evolution of the C/O monitor signal intensity. \n {date}.{discharge_nr}",
#     )
#     ax.set_xlabel("Time [s]")
#     ax.set_ylabel("Intensity [a.u.]")
#     ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#     ax.legend()
#     plt.show()


# using_pd()
# plot()
