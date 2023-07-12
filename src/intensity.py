from dateutil import tz
from functools import wraps
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate


def timer(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        exec_time = time.time() - start
        print(f"Execution time is: {exec_time:.2f}s")
        return result

    return wrapper


def get_pixel_intens(binary_file, row, column):
    shift = 4096 + (row - 1) * 3072 + (column - 1) * 3
    binary_file.seek(shift)
    bytes_ = binary_file.read(3)
    return int.from_bytes(bytes_, "little")


def get_spectrum(binary_file, cols_number, row):
    return list(
        map(
            lambda column: get_pixel_intens(binary_file, row, column + 1),
            range(cols_number),
        )
    )


def integrate_spectrum(spectrum, range_):
    line = spectrum[range_[0] : range_[-1]]
    background = min(line[0], line[-1])
    ### removes the background level (do not mistakenly take as a noise level!)
    line -= background
    pixels = np.linspace(range_[0], range_[1] - 1, num=range_[1] - range_[0])
    integral = integrate.simps(line, pixels)

    return integral


def get_det_size(binary_file):
    binary_file.seek(0)
    bajty = binary_file.read(4)
    ncols = int.from_bytes(bajty, "little")

    binary_file.seek(4)
    bajty = binary_file.read(4)
    nrows = int.from_bytes(bajty, "little")
    return nrows, ncols


def generate_time_stamps(time_interval, dt):
    start, end = time_interval
    selected_time_stamps = [
        "{:.3f}".format(i) for i in np.arange(start, end + dt / 100, dt)
    ]
    # print(selected_time_stamps)
    return selected_time_stamps


def get_BGR(file_name):
    with open(file_name, "rb") as binary_file:
        _, cols_number = get_det_size(binary_file)
        spec_in_time = pd.DataFrame()
        spectrum = get_spectrum(binary_file, cols_number, row=1)
        spec_header = f"{file_name.stem}"
        spec_in_time[spec_header] = spectrum
        col_name = spec_header

    return col_name, spectrum


@timer
def get_all_spectra(file_name, lineRange, time_interval, dt):
    # wyznacza intensywnosci wybranych przerzialod czasowych (time interval)
    start_time = time.time()

    with open(file_name, "rb") as binary_file:
        rows_number, cols_number = get_det_size(binary_file)
        idx_start = int(min(time_interval) / dt)
        idx_end = int(max(time_interval) / dt)

        if idx_end < rows_number:
            spectra = list(
                map(
                    lambda row: get_spectrum(binary_file, cols_number, row),
                    range(idx_start, idx_end),
                )
            )
            spec_in_time = pd.DataFrame(spectra).T
            return spec_in_time

        spectra = list(
            map(
                lambda row: get_spectrum(binary_file, cols_number, row),
                range(idx_start, rows_number),
            )
        )
        spec_in_time = pd.DataFrame(spectra).T

    print("{:.2f}".format(time.time() - start_time))
    return spec_in_time


def get_utc_from_csv(file_name, element, date):
    data_file = (
        pathlib.Path(__file__).parent.parent.resolve()
        / "data_processing"
        / "discharge_numbers"
        / f"{element}"
        / f"{element}-{date}.csv"
    )
    with open(data_file, "r") as data:
        df = pd.read_csv(
            data,
            sep="\t",
            usecols=[
                "date",
                "discharge_nr",
                "file_name",
                "time",
                "type_of_data",
                "file_size",
                "utc_time",
                "frequency",
            ],
        )
        df = df.astype({"date": int})
        exp_info = df.loc[df["file_name"] == file_name.stem]

    return exp_info


def convert_frequency_to_dt(frequency):
    return 1 / frequency


def calc_utc_timestamps(utc_time, selected_time_stamps, dt):
    frames_nr = len(selected_time_stamps)
    ns_time_stamps = [int(i * dt * 1e9) for i in range(frames_nr)]
    ns_time_stamps.reverse()
    removed = [utc_time - i for i in ns_time_stamps]

    return removed


##### dodatkowo wyliczyc timestampy w odniesieniu do triggerow i wrzucic na wykresy!!!!!


def get_discharge_nr_from_csv(element, date, discharge_nr, time_interval, plotter):
    integral_line_range = {"C": [120, 990], "O": [190, 941]}
    range_ = integral_line_range[f"{element}"]
    cwd = pathlib.Path.cwd()
    file_path = cwd / "discharge_numbers" / f"{element}" / f"{element}-{date}.csv"

    df = pd.read_csv(file_path, sep="\t")
    if not discharge_nr == 0:
        df["discharge_nr"] = df["discharge_nr"].replace("-", "0").astype(int)
        selected_file_names = df.loc[df["discharge_nr"] == discharge_nr][
            "file_name"
        ].to_list()

    if not selected_file_names:
        print("No discharge!")
        return None

    directory = (
        pathlib.Path(__file__).parent.parent.resolve()
        / "__Experimental_data"
        / "data"
        / element
        / date
    )
    file_list = list(directory.glob("**/*"))
    discharge_files = [
        x
        for x in file_list
        if x.stat().st_size > 8000
        and x.stem in selected_file_names
        and "BGR" not in x.stem
    ]
    bgr_files = [x for x in file_list if "BGR" in x.stem in selected_file_names]
    for file_name in discharge_files:
        exp_info = get_utc_from_csv(file_name, element, date)
        utc_time = int(exp_info["utc_time"].iloc[0])
        discharge_nr = int(exp_info["discharge_nr"].iloc[0])
        frequency = int(exp_info["frequency"].iloc[0])
        dt = convert_frequency_to_dt(frequency)

        spectra = get_all_spectra(file_name, range_, time_interval, dt)
        ### takes last recorded noise signal before the discharge
        bgr_file_name, bgr_spec = get_BGR(bgr_files[-1])

        ### TODO co jesli nie ma plikow BGR???
        spectra_without_bgr = spectra.iloc[:, :].sub(bgr_spec, axis=0)
        selected_time_stamps = generate_time_stamps(time_interval, dt)[
            : spectra_without_bgr.shape[1]
        ]

        time_stamps = calc_utc_timestamps(utc_time, selected_time_stamps, dt)

        # intensity = list(map(lambda row: get_spectrum(binary_file, cols_number, row), range(idx_start, idx_end + 1)))
        # intensity = list(map(lambda range_: integrate_spectrum(spectra_without_bgr, range_), range_))############################ zmapowac ponizsza petle
        ##################### TODO DO ZMAPOWANIA
        intensity = []
        for i in spectra_without_bgr:
            integral = integrate_spectrum(np.array(spectra_without_bgr[i]), range_)
            intensity.append(integral)

        df2 = pd.DataFrame()
        df2["time"] = selected_time_stamps
        df2[f"{element}-intensity"] = intensity
        df2[f"{element}-intensity"] = df2[f"{element}-intensity"].round(1)
        df2["utc_timestamps"] = time_stamps
        # df2["BGR"] = bgr_spec
        df2 = df2.iloc[
            1:
        ]  ### usuwa p[ierwsza ramke w czasie 0s -> w celu usuniecia niefizycznych wartosci

        def save_file():
            destination = (
                pathlib.Path.cwd() / "time_evolutions" / f"{element}" / f"{date}"
            )
            destination.mkdir(parents=True, exist_ok=True)
            df2.to_csv(
                destination
                / f"{element}-{date}-exp_{discharge_nr}-{file_name.stem}.csv",
                sep="\t",
                index=False,
                header=True,
            )
            print("File successfully saved!")

        save_file()

        def plot():
            ax = df2.plot(
                x="time",
                y=f"{element}-intensity",
                label=f"{element}",
                linewidth=0.7,
                color="blue",
                title=f"Evolution of the C/O monitor signal intensity. \n {date}.{discharge_nr}",
            )
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Intensity [a.u.]")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.legend()
            plt.show()

        if plotter:
            plot()


import time
from intensity import get_discharge_nr_from_csv
import pandas as pd
import pathlib

start = time.time()

elements = ["C", "O"]  # , "O"]  # , "O"]
date = "20230215"


discharges = [15]  # , 18, 21]
# discharges = [0]
time_interval = [0, 415]


if __name__ == "__main__":
    for shot in discharges:
        for element in elements:
            get_discharge_nr_from_csv(element, date, shot, time_interval, plotter=True)


#### na osiach rowniey dodac cyasz lokalne z eksperymentow - czas rozpoczecia i zakonczenia, a takze ewentualnie dla sprawdzenia utc
