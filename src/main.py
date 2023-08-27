from datetime import datetime, timedelta
import time

import matplotlib.pyplot as plt
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


# if __name__ == "__main__":
#     dates_list = generate_dates_list("20230101", "20230331")
#     elements_list = ["C", "O"]
#     discharges_list = [i for i in range(1, 100)]

#     dates_list = ["20230125"]
#     elements_list = ["C"]  # , "O"]
#     discharges_list = [14]

#     time_interval = [0, 500]  ### ponizej 4.5-5s time interval  czas time jest zly? TODO
#     for element in elements_list:
#         for date in dates_list:
#             for discharge in discharges_list:
#                 try:
#                     f = DischargeFilesSelector(element, date, discharge)
#                     discharge_files = f.discharge_files
#                     for file_name in discharge_files:
#                         Intensity(
#                             element,
#                             date,
#                             discharge,
#                             file_name,
#                             time_interval,
#                             save_df=True,
#                             save_fig=True,
#                             plot=True,
#                         )
#                 except FileNotFoundError:
#                     print(
#                         f"{element}_{date}_{discharge} - No matching file found! Continue..."
#                     )
#                     continue


### rysowanie C/O na jednym wykresie

if __name__ == "__main__":
    dates_list = generate_dates_list("20230101", "20230331")
    elements_list = ["C", "O"]
    discharges_list = [i for i in range(1, 100)]

    dates_list = ["20230125"]
    elements_list = ["C", "O"]
    discharges_list = [14]

    time_interval = [0, 500]

    def collect_elements_data():
        ...

    bufor = []
    for element in elements_list:
        for date in dates_list:
            for discharge in discharges_list:
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
                                save_df=False,
                                save_fig=False,
                                plot=False,
                            )
                        )
                except FileNotFoundError:
                    print(
                        f"{element}_{date}_{discharge} - No matching file found! Continue..."
                    )
                    continue

    def plot_all_elements(normalized=False, save_fig=False, plot=True):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()

        colors = ["blue", "red"]

        for idx, instance in enumerate(bufor):
            element = instance.element
            df = instance.df
            date = instance.date
            discharge_nr = instance.discharge_nr

            ax1.set_title(
                f"Comparison of Lyman-alpha intensities\n {date}.{discharge_nr:03}"
            )
            intensity = df[f"QSO_{element}_{date}.{discharge_nr}"]
            if not normalized:
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
                    color=colors[idx],
                )
            else:
                ax1.plot(
                    pd.to_datetime(df["time"], format="%H:%M:%S.%f", errors="coerce"),
                    intensity / intensity.max(),
                    alpha=0,
                )
                ax2.plot(
                    np.asarray(df["discharge_time"], float),
                    intensity / intensity.max(),
                    label="discharge_time",
                    linewidth=0.4,
                    color=colors[idx],
                )

            ax1.xaxis.set_major_formatter(
                plt.matplotlib.dates.DateFormatter("%H:%M:%S")
            )
            ax1.tick_params(axis="x", rotation=45)
            ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax1.set_ylabel("Intensity [a.u.]")
            ax1.set_xlabel("Local time")
            ax2.set_xlabel("Discharge time [s]")
            ax1.grid(which="major")
            plt.tight_layout(rect=None)

        def save_plot():
            path = (
                instance.file_path_manager.images().parent.parent.parent
                / "comparison"
                / date
            )
            path.mkdir(parents=True, exist_ok=True)
            if normalized:
                plt.savefig(
                    path
                    / f"QSO_comparison_norm_{date}.{discharge_nr:03}-{file_name.stem}.png",
                    dpi=200,
                )
            else:
                plt.savefig(
                    path
                    / f"QSO_comparison_{date}.{discharge_nr:03}-{file_name.stem}.png",
                    dpi=200,
                )

        if save_fig:
            save_plot()
        if plot:
            plt.show()
        plt.close()

    plot_all_elements(normalized=True, save_fig=True, plot=True)
