from datetime import datetime, timedelta
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import pandas as pd

from file_reader import (
    FilePathManager,
    ExperimentalFilesSelector,
)
from utc_converter import get_time_from_UTC
from intensity import run_intensity_calculations


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


def get_triggers(date, discharge_nr):
    fpm = FilePathManager()
    pt = fpm.get_directory_for_program_triggers()
    return pt


def plot_elements_comparison(bufor, date, discharge_nr, normalized, save_fig, plot):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    time = []
    for idx, instance in enumerate(bufor):
        element, df = instance
        time.append(len(df))
        color = "blue" if element == "C" else "red"
        plot_single_element(
            ax1, ax2, df, element, date, discharge_nr, normalized, color
        )
    time = max(time)
    plot_triggers(ax1, ax2, time, date, discharge_nr)
    labels = [element for instance in bufor]
    legend = ax2.legend(labels)
    for line in legend.get_lines():
        line.set_linewidth(1.5)

    configure_axes(ax1, ax2)
    save_or_show_plot(bufor[0], date, discharge_nr, normalized, save_fig, plot)

    plt.close()


def plot_triggers(ax, ax2, time, date, discharge_nr):
    fpm = get_triggers(date, discharge_nr)
    f_path = fpm / f"{date}_triggers.csv"
    df2 = pd.read_csv(f_path, sep="\t")
    wiersz = df2.loc[df2["discharge_nr"] == discharge_nr]
    T1 = int(wiersz["T1"].to_numpy())
    T6 = int(wiersz["T6"].to_numpy())

    T1_human = get_time_from_UTC(T1)
    T6_human = get_time_from_UTC(T6)
    ax.axvline(
        x=pd.to_datetime(T1_human, format="%H:%M:%S.%f", errors="coerce"),
        color="black",
        linestyle="--",
        label="T1",
        linewidth=1,
    )
    # ax.axvline(
    #     x=pd.to_datetime(T6_human, format="%H:%M:%S.%f", errors="coerce"),
    #     color="black",
    #     linestyle="--",
    #     label="T6",
    #     linewidth=1,
    # )

    # delta = (T6 - T1) / 1e9
    # ax2.axvline(
    #     np.asarray(delta),
    #     color="black",
    #     linestyle="--",
    #     label="T6",
    #     linewidth=1,
    # )


def plot_single_element(ax1, ax2, df, element, date, discharge_nr, normalized, color):
    ax1.set_title(f"Comparison of Lyman-alpha intensities\n {date}.{discharge_nr:03}")
    intensity = df[f"QSO_{element}_{date}.{discharge_nr}"]

    if normalized:
        intensity /= intensity.max()
    # breakpoint()
    ax1.plot(
        pd.to_datetime(df["time"], format="%H:%M:%S.%f", errors="coerce"),
        intensity,
        color=color,
        linewidth=0.4,
    )
    ax2.plot(
        np.asarray(df["discharge_time"], float),
        intensity,
        label="discharge_time",
        alpha=0,
    )
    labels = ["C", "O"]
    ax1.legend(labels)


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
    # parent_path = instance.file_path_manager.images().parent.parent.parent
    # path = parent_path / "_comparison" / date / image_type
    # path.mkdir(parents=True, exist_ok=True)

    # filename = (
    #     f"QSO_comparison_norm_{date}.{discharge_nr:03}.png"
    #     if normalized
    #     else f"QSO_comparison_{date}.{discharge_nr:03}.png"
    # )

    # if save_fig:
    #     plt.savefig(path / filename, dpi=200)
    #     print(
    #         f"QSO_comparison_{image_type}_{date}.{discharge_nr:03} - intensity evolution saved!"
    #     )
    if plot:
        plt.show()


def main():
    time_interval = [0, 1000] # mapuje po WSZYSTKICH plikach do 110 a nie calosci wyladowania - TODO
    ### sprawic aby wybieranie przedzialu czasowego sprawialo ze wybiera odpowiednie pliki

    dates_list = generate_dates_list("20230101", "20230331")
    elements_list = ["C"]  # , "O"]
    discharges_list = [i for i in range(1, 100)]

    dates_list = ["20230215"]
    # elements_list = ["C", "O"]  # , "O"]  # , "O"]
    discharges_list = [32]  # 20230117.050 rowniez kiepsko

    for date in dates_list:
        for discharge in discharges_list:
            bufor = []

            for element in elements_list:
                try:
                    f = ExperimentalFilesSelector()
                    discharge_files = f.grab_discharge_files(element, date, discharge)
                    for file_name in discharge_files:
                        bufor.append(
                            run_intensity_calculations(
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
                    *parameters, normalized=False, save_fig=True, plot=True
                )
                plot_elements_comparison(
                    *parameters, normalized=True, save_fig=True, plot=True
                )


if __name__ == "__main__":
    main()
