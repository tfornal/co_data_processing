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


##### dodatkowo wyliczyc timestampy w odniesieniu do triggerow i wrzucic na wykresy!!!!!


def get_discharge_nr_from_csv(element, date, discharge_nr, time_interval, plotter):
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
        print("No discharge!")
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
        df2["discharge_time"] = selected_time_stamps
        df2[f"{element}-intensity"] = intensity
        df2[f"{element}-intensity"] = df2[f"{element}-intensity"].round(1)
        df2["utc_timestamps"] = time_stamps
        # df2["BGR"] = bgr_spec
        df2 = df2.iloc[1:]
        ### usuwa p[ierwsza ramke w czasie 0s -> w celu usuniecia niefizycznych wartosci
        
        from utc_converter import get_time_from_UTC

        time = list(map(get_time_from_UTC, df2["utc_timestamps"]))
        x_labels = [date.strftime("%H:%M:%S.%f")[:-3] if date.microsecond % 1_000 == 0 else ""
                for date in time]
        df2["time"] = x_labels
        # Wyodrębnienie ostatniej kolumny
        last_column = df2.pop(df2.columns[-1])

        # # Wstawienie ostatniej kolumny na początek
        df2.insert(0, last_column.name, last_column)
        print(df2)


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
                / f"{element}-{date}-exp_{discharge_nr}-{file_name.stem}.csv",
                sep="\t",
                index=False,
                header=True,
            )
            print("File successfully saved!")

        def plot():
            fig, ax = plt.subplots(figsize=(12, 5))
            # ax2 = ax1.twinx()
            from utc_converter import get_time_from_UTC

            time = list(map(get_time_from_UTC, df2["utc_timestamps"]))

            # def conv_2(time):
            #     return time.hour, time.minute, time.second, time.microsecond // 1000

            # time2 = list(map(conv_2, time1))
            # print(time2)

            # df2["godzina"] = time2
# 
            x = df2["time"]
            y = df2[f"{element}-intensity"]
            # x = df2["godzina"]

            ax.plot(x, y, color="blue")

            x = list(range(len(time)))
            x_labels = [date.strftime("%H:%M:%S") if date.microsecond % 5_000_000 == 0 else ""
                for date in time]
            ax.set_xticks(np.arange(len(x)), labels=x_labels)
            plt.setp(ax.get_xticklabels(), rotation=90)
            plt.tight_layout()
            # ax.set_xlabel("Wartości x1")
            # ax.set_ylabel("Wartości y", color="blue")
            # ax2 = ax.twiny()
            # ax2.set_xticks(df2["utc_timestamps"])
            # ax2.set_xlabel(time2)

            plt.show()

            # # Przykładowe dane
            # x1 = np.linspace(0, 10, 100)
            # x2 = np.linspace(0, 5, 50)
            # y = np.sin(x1)

            # fig, ax1 = plt.subplots()  # Tworzenie głównego wykresu

            # # Tworzenie pierwszej osi x i osi y
            # ax1.plot(x1, y, color='blue')
            # ax1.set_xlabel('Wartości x1')
            # ax1.set_ylabel('Wartości y', color='blue')

            # # Tworzenie drugiej osi x i przypisanie innych danych
            # ax2 = ax1.twinx()
            # ax2.plot([], [])  # Puste dane dla drugiej osi y, tylko dla utworzenia drugiej osi x
            # ax2.set_xlabel('Wartości x2')

            # # Ustawienie pozycji drugiego zestawu danych osi x
            # ax2.spines['bottom'].set_position(('outward', 60))

            # plt.show()

            # df2.plot(
            #     df2["time"].astype(float),
            #     df2[f"{element}-intensity"],
            # )
            # df2.plot(df2["utc_timestamps"], df2[f"{element}-intensity"])

            # ax = df2.plot(
            #     x="time",
            #     y=f"{element}-intensity",
            #     label=f"{element}",
            #     linewidth=0.7,
            #     color="blue",
            #     title=f"Evolution of the C/O monitor signal intensity. \n {date}.{discharge_nr}",
            # )
            # ax.set_xlabel("Time [s]")
            # ax.set_ylabel("Intensity [a.u.]")
            # ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            # ax.legend()
            # plt.show()

        if plotter:
            plot()
        save_file()
