from datetime import datetime, timedelta
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import pandas as pd

from file_reader import (
    FilePathManager,
    DischargeFilesSelector,
    DischargeDataExtractor,
    BackgroundFilesSelector,
)
from intensity import Intensity


def generate_dates_list(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    current_date = start_date
    dates_list = []

    while current_date <= end_date:
        formatted_date = current_date.strftime("%Y%m%d")
        dates_list.append(formatted_date)
        current_date += timedelta(days=1)

    return dates_list


def plot_elements_comparison(bufor, date, discharge_nr, normalized, save_fig, plot):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()

    for idx, instance in enumerate(bufor):
        element = instance.element
        df = instance.df
        color = "blue" if element == "C" else "red"
        plot_single_element(
            ax1, ax2, df, element, date, discharge_nr, normalized, color
        )

    labels = [instance.element for instance in bufor]
    legend = ax2.legend(labels)
    for line in legend.get_lines():
        line.set_linewidth(1.5)

    configure_axes(ax1, ax2)
    save_or_show_plot(bufor[0], date, discharge_nr, normalized, save_fig, plot)

    plt.close()


def plot_single_element(ax1, ax2, df, element, date, discharge_nr, normalized, color):
    ax1.set_title(f"Comparison of Lyman-alpha intensities\n {date}.{discharge_nr:03}")
    intensity = df[f"QSO_{element}_{date}.{discharge_nr}"]

    if normalized:
        intensity /= intensity.max()

    ax1.plot(
        pd.to_datetime(df["time"], format="%H:%M:%S.%f", errors="coerce"),
        intensity,
        alpha=0,
    )

    ax2.plot(
        np.asarray(df["discharge_time"], float),
        intensity,
        label="discharge_time",
        linewidth=0.4,
        color=color,
    )
    labels = ["C", "O"]
    ax2.legend(labels)  # Możesz dostosować położenie legendy


def configure_axes(ax1, ax2):
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M:%S"))
    ax1.tick_params(axis="x", rotation=45)
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax1.set_ylabel("Intensity [a.u.]")
    ax1.set_xlabel("Local time")
    ax2.set_xlabel("Discharge time [s]")
    ax1.grid(which="major")
    plt.tight_layout(rect=None)


def save_or_show_plot(instance, date, discharge_nr, normalized, save_fig, plot):
    if not (save_fig or plot):
        return None

    image_type = "normalized" if normalized else "original"
    parent_path = instance.file_path_manager.images().parent.parent.parent
    path = parent_path / "_comparison" / date / image_type
    path.mkdir(parents=True, exist_ok=True)

    filename = (
        f"QSO_comparison_norm_{date}.{discharge_nr:03}.png"
        if normalized
        else f"QSO_comparison_{date}.{discharge_nr:03}.png"
    )

    if save_fig:
        plt.savefig(path / filename, dpi=200)
        print(
            f"QSO_comparison_{image_type}_{date}.{discharge_nr:03} - ntensity evolution saved!"
        )
    if plot:
        plt.show()


def main():
    time_interval = [0, 500]

    dates_list = generate_dates_list("20230101", "20230331")
    elements_list = ["C", "O"]
    discharges_list = [i for i in range(1, 100)]

    # dates_list = ["20230117"]
    # elements_list = ["C", "O"]  # , "O"]
    # discharges_list = [14]
    """
    Sprawdzic gdy wiele plików - np. w przypadku
    dates_list = ["20230117"]
    elements_list = ["C", "O"]  # , "O"]
    discharges_list = [14]

    wszystko wrzuca an jeden wykres - a moze rysowanie linii czasu?
    """
    for date in dates_list:
        for discharge in discharges_list:
            bufor = []

            for element in elements_list:
                try:
                    f = DischargeFilesSelector(element, date, discharge)
                    discharge_files = f.discharge_files

                    for file_name in discharge_files:
                        bufor.append(
                            Intensity(
                                element,
                                date,
                                discharge,
                                file_name,
                                time_interval,
                                save_df=True,
                                save_fig=True,
                                plot=False,
                            )
                        )
                except FileNotFoundError:
                    print(
                        f"{element}_{date}_{discharge} - No matching file found! Continue..."
                    )

            if len(bufor) >= 2:
                parameters = (bufor, date, discharge)
                plot_elements_comparison(
                    *parameters, normalized=False, save_fig=True, plot=False
                )
                plot_elements_comparison(
                    *parameters, normalized=True, save_fig=True, plot=False
                )


if __name__ == "__main__":
    main()
