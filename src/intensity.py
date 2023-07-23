from functools import wraps
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from yaml import load, dump
from file_reader import Files
from utc_converter import get_time_from_UTC


### zrobic to w OOP!!
def timer(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        exec_time = time.time() - start
        print(f"Execution time: {exec_time:.2f}s")
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


def integrate_spectrum(spectrum, spectrum_range):
    line = spectrum[spectrum_range[0] : spectrum_range[-1]]
    background = min(line[0], line[-1])
    ### removes the background level (do not mistakenly take as a noise level!)
    line -= background
    pixels = np.linspace(
        spectrum_range[0],
        spectrum_range[1] - 1,
        num=spectrum_range[1] - spectrum_range[0],
    )
    integral = integrate.simps(line, pixels)

    return integral


def get_det_size(binary_file):
    binary_file.seek(0)
    bites = binary_file.read(4)
    ncols = int.from_bytes(bites, "little")

    binary_file.seek(4)
    bites = binary_file.read(4)
    nrows = int.from_bytes(bites, "little")
    return nrows, ncols


def validate_time_duration(
    time_interval,
    dt,
):
    ###
    pass


@timer
def generate_time_stamps(time_interval, dt):
    start, end = time_interval
    selected_time_stamps = [
        "{:.3f}".format(i)
        for i in np.arange(
            start, end + dt / 100, dt
        )  ### czy to 100 na pewno jest tu potrzebne?
    ]
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
    ### TODO test!!!!!!

    # time_interval = [0 for i in time_interval if i < 0 else i for i in time_interval]

    with open(file_name, "rb") as binary_file:
        rows_number, cols_number = get_det_size(binary_file)

        ### opisa dlaczego tak!!!!!!!!!!!!!!!! TODO
        aquisition_time = rows_number * dt
        if float(max(time_interval)) > aquisition_time:
            idx_end = rows_number - int(aquisition_time / dt)
        else:
            idx_end = rows_number - int(max(time_interval) / dt)
        idx_start = rows_number - int(min(time_interval) / dt)
        spectra = list(
            map(
                lambda row: get_spectrum(binary_file, cols_number, row),
                range(idx_end, idx_start),
            )
        )
        spec_in_time = pd.DataFrame(spectra).T
        spec_in_time = spec_in_time[spec_in_time.columns[::-1]]
        return spec_in_time
    return spec_in_time


def get_utc_from_csv(file_name, element, date):
    data_file = (
        pathlib.Path(__file__).parent.parent.resolve()
        / "data"
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


def check_if_negative(numbers_list):
    numbers_list = [num if num >= 0 else 0 for num in numbers_list]
    return numbers_list


def get_discharge_nr_from_csv(element, date, discharge_nr, time_interval, plotter):
    time_interval = check_if_negative(time_interval)

    integral_line_range = {"C": [120, 990], "O": [190, 941]}
    range_ = integral_line_range[f"{element}"]
    file_path = (
        pathlib.Path(__file__).parent.parent.resolve()
        / "data"
        / "discharge_numbers"
        / f"{element}"
        / f"{element}-{date}.csv"
    )
    df = pd.read_csv(file_path, sep="\t")
    if not discharge_nr == 0:
        df["discharge_nr"] = df["discharge_nr"].replace("-", "0").astype(int)
        selected_file_names = df.loc[df["discharge_nr"] == discharge_nr][
            "file_name"
        ].to_list()

    if not selected_file_names:
        print(f"{date}.{discharge_nr:03} -> No discharge!")
        return None

    directory = (
        pathlib.Path(__file__).parent.parent.resolve()
        / "data"
        / "exp_data"
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
        try:
            bgr_file_name, bgr_spec = get_BGR(bgr_files[-1])

            ### TODO co jesli nie ma plikow BGR???
            spectra_without_bgr = spectra.iloc[:, :].sub(bgr_spec, axis=0)
            selected_time_stamps = generate_time_stamps(time_interval, dt)[
                : spectra_without_bgr.shape[1]
            ]
        #### TODO background jesli go nie ma!!!
        except IndexError:
            spectra_without_bgr = spectra
            selected_time_stamps = generate_time_stamps(time_interval, dt)[
                : spectra_without_bgr.shape[1]
            ]
            print("BACKGROUND NOT REMOVED!!!!!!!!!!!!!!!!!!!!!!!! TODO")
        time_stamps = calc_utc_timestamps(utc_time, selected_time_stamps, dt)

        # intensity = list(map(lambda row: get_spectrum(binary_file, cols_number, row), range(idx_start, idx_end + 1)))
        # intensity = list(map(lambda range_: integrate_spectrum(spectra_without_bgr, range_), range_))############################ zmapowac ponizsza petle
        ##################### TODO DO ZMAPOWANIA
        intensity = []
        for i in spectra_without_bgr:
            integral = integrate_spectrum(np.array(spectra_without_bgr[i]), range_)
            intensity.append(integral)

        df2 = pd.DataFrame()
        df2["discharge_time"] = selected_time_stamps
        df2[f"QSO_{element}_{date}.{discharge_nr}"] = intensity
        df2[f"QSO_{element}_{date}.{discharge_nr}"] = df2[
            f"QSO_{element}_{date}.{discharge_nr}"
        ].round(1)
        df2["utc_timestamps"] = time_stamps
        df2 = df2.iloc[:-1]
        ### usuwa p[ierwsza ramke w czasie 0s -> w celu usuniecia niefizycznych wartosci

        time = list(map(get_time_from_UTC, df2["utc_timestamps"]))
        x_labels = [
            date.strftime("%H:%M:%S.%f")[:-3] if date.microsecond % 1_000 == 0 else ""
            for date in time
        ]
        df2["time"] = x_labels

        # breakpoint()
        def save_file():
            destination = (
                pathlib.Path(__file__).parent.parent.resolve()
                / "data"
                / "time_evolutions"
                / f"{element}"
                / f"{date}"
            )
            destination.mkdir(parents=True, exist_ok=True)
            df2.to_csv(
                destination
                / f"QSO_{element}_{date}.{discharge_nr:03}-{file_name.stem}-time_{min(time_interval)}_{max(time_interval)}s.csv",
                sep="\t",
                index=False,
                header=True,
            )
            print(
                f"QSO_{element}_{date}.{discharge_nr:03} - intensity evolution saved!"
            )

        def plot():
            fig, ax1 = plt.subplots()
            # plt.style.use("_mpl-gallery")
            ax1.set_title(
                f"Evolution of the C/O monitor signal intensity.\n {date}.{discharge_nr:03}"
            )
            ax2 = ax1.twiny()

            ax1.plot(
                pd.to_datetime(df2["time"], format="%H:%M:%S.%f", errors="coerce"),
                df2[f"QSO_{element}_{date}.{discharge_nr}"],
                alpha=0,
            )

            ax2.plot(
                np.asarray(df2["discharge_time"], float),
                df2[f"QSO_{element}_{date}.{discharge_nr}"],
                color="blue",
                label="discharge_time",
                linewidth=0.4,
            )
            ax1.tick_params(axis="x", rotation=45)
            ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax1.set_ylabel("Intensity [a.u.]")
            ax1.set_xlabel("Local time")
            ax2.set_xlabel("Discharge time [s]")
            destination = (
                pathlib.Path(__file__).parent.parent.resolve()
                / "data"
                / "time_evolutions"
                / f"{element}"
                / f"{date}"
                / "img"
            )
            ax1.grid(which="major")
            plt.tight_layout()  # Dostosowanie rozmiaru obszaru wykresu
            destination.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                destination
                / f"QSO_{element}_{date}.{discharge_nr:03}-{file_name.stem}-time_{min(time_interval)}_{max(time_interval)}s.png",
                dpi=200,
            )
            plt.show()

        save_file()
        if plotter:
            plot()
