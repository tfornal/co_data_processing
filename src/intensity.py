from functools import wraps
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate

# from yaml import load, dump
from file_reader import Files, FilePaths
from utc_converter import get_time_from_UTC


def timer(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        exec_time = time.time() - start
        print(f"Execution time: {exec_time:.2f}s")
        return result

    return wrapper


class Intensity:
    def __init__(self, element, date, discharge_nr, time_interval, plotter=True):
        self.element = element
        self.date = date
        self.discharge_nr = discharge_nr
        self.time_interval = self.check_if_negative(time_interval)

        self.fp = self._get_file_paths()
        self.discharge_nr_file_path = self._get_file_path()
        self.exp_data_file_path = self._get_exp_data_file_path()
        self.file_list = self.grab_file_list()

        self.df2 = self.final_calculation()
        self.df2 = self.make_df()
        if plotter:
            self.plot()

    def _get_file_paths(self):
        return FilePaths(self.element, self.date)

    def _get_file_path(self):
        return self.fp.discharge_nrs()

    def _get_exp_data_file_path(self):
        return self.fp.experimental_data()

    def get_pixel_intens(self, binary_file, row, column):
        shift = 4096 + (row - 1) * 3072 + (column - 1) * 3
        binary_file.seek(shift)
        bytes_ = binary_file.read(3)
        return int.from_bytes(bytes_, "little")

    def get_spectrum(self, binary_file, cols_number, row):
        return list(
            map(
                lambda column: self.get_pixel_intens(binary_file, row, column + 1),
                range(cols_number),
            )
        )

    def integrate_spectrum(self, spectrum, spectrum_range):
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

    def get_det_size(self, binary_file):
        binary_file.seek(0)
        bites = binary_file.read(4)
        ncols = int.from_bytes(bites, "little")

        binary_file.seek(4)
        bites = binary_file.read(4)
        nrows = int.from_bytes(bites, "little")
        return nrows, ncols

    def validate_time_duration(self):
        ### TODO
        pass

    def generate_time_stamps(self, time_interval):
        start, end = time_interval
        selected_time_stamps = [
            "{:.3f}".format(i)
            for i in np.arange(
                start, end + self.dt / 100, self.dt
            )  ### czy to 100 na pewno jest tu potrzebne?
        ]
        return selected_time_stamps

    def get_BGR(self, file_name):
        with open(file_name, "rb") as binary_file:
            _, cols_number = self.get_det_size(binary_file)
            spec_in_time = pd.DataFrame()
            spectrum = self.get_spectrum(binary_file, cols_number, row=1)
            spec_header = f"{file_name.stem}"
            spec_in_time[spec_header] = spectrum
            col_name = spec_header

        return col_name, spectrum

    def get_all_spectra(self, file_name):
        # wyznacza intensywnosci wybranych przerzialod czasowych (time interval)
        ### TODO test!!!!!!
        with open(file_name, "rb") as binary_file:
            rows_number, cols_number = self.get_det_size(binary_file)
            ### opisa dlaczego tak!!!!!!!!!!!!!!!! TODO
            aquisition_time = rows_number * self.dt
            if float(max(self.time_interval)) > aquisition_time:
                idx_end = rows_number - int(aquisition_time / self.dt)
            else:
                idx_end = rows_number - int(max(self.time_interval) / self.dt)
            idx_start = rows_number - int(min(self.time_interval) / self.dt)
            spectra = list(
                map(
                    lambda row: self.get_spectrum(binary_file, cols_number, row),
                    range(idx_end, idx_start),
                )
            )
            spec_in_time = pd.DataFrame(spectra).T
            spec_in_time = spec_in_time[spec_in_time.columns[::-1]]
            return spec_in_time
        return spec_in_time

    def get_utc_from_csv(self, file_name):
        with open(self.discharge_nr_file_path, "r") as data:
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

    def convert_frequency_to_dt(self, frequency):
        return 1 / frequency

    def calc_utc_timestamps(self, utc_time, selected_time_stamps):
        frames_nr = len(selected_time_stamps)
        ns_time_stamps = [int(i * self.dt * 1e9) for i in range(frames_nr)]
        ns_time_stamps.reverse()
        removed = [utc_time - i for i in ns_time_stamps]

        return removed

    def check_if_negative(self, numbers_list):
        numbers_list = [num if num >= 0 else 0 for num in numbers_list]
        return numbers_list

    def grab_file_list(self):
        return list(self.exp_data_file_path.glob("**/*"))

    def final_calculation(self):
        integral_line_range = {"C": [120, 990], "O": [190, 941]}
        range_ = integral_line_range[f"{self.element}"]

        df = pd.read_csv(self.discharge_nr_file_path, sep="\t")
        if not self.discharge_nr == 0:
            df["discharge_nr"] = df["discharge_nr"].replace("-", "0").astype(int)
            selected_file_names = df.loc[df["discharge_nr"] == self.discharge_nr][
                "file_name"
            ].to_list()

        if not selected_file_names:
            print(f"{self.date}.{self.discharge_nr:03} -> No discharge!")
            return None

        discharge_files = [
            x
            for x in self.file_list
            if x.stat().st_size > 8000
            and x.stem in selected_file_names
            and "BGR" not in x.stem
        ]
        bgr_files = [
            x for x in self.file_list if "BGR" in x.stem in selected_file_names
        ]
        # def grab_relative_discharge_files(self):
        #     for file_name in discharge_files:

        for file_name in discharge_files:
            exp_info_df = self.get_utc_from_csv(file_name)
            utc_time = int(exp_info_df["utc_time"].iloc[0])
            self.discharge_nr = int(exp_info_df["discharge_nr"].iloc[0])
            frequency = int(exp_info_df["frequency"].iloc[0])
            self.dt = self.convert_frequency_to_dt(frequency)
            breakpoint()
            ########### TODO  time_interval automatycznie dostoswany do rozmiarów pliku - using get_det_size
            spectra = self.get_all_spectra(file_name)
            ### takes last recorded noise signal before the discharge

            ### a moze zastosowac tutaj generator yield?
            try:
                bgr_file_name, bgr_spec = self.get_BGR(bgr_files[-1])

                spectra_without_bgr = spectra.iloc[:, :].sub(bgr_spec, axis=0)
                selected_time_stamps = self.generate_time_stamps(self.time_interval)[
                    : spectra_without_bgr.shape[1]
                ]
            #### TODO background jesli go nie ma!!!
            except IndexError:
                spectra_without_bgr = spectra
                selected_time_stamps = self.generate_time_stamps(self.time_interval)[
                    : spectra_without_bgr.shape[1]
                ]
                print("TODO BACKGROUND NOT REMOVED!!!!!!!!!!!!!!!!!!!!!!!!")
            time_stamps = self.calc_utc_timestamps(utc_time, selected_time_stamps)

            # intensity = list(map(lambda row: get_spectrum(binary_file, cols_number, row), range(idx_start, idx_end + 1)))
            # intensity = list(map(lambda range_: integrate_spectrum(spectra_without_bgr, range_), range_))############################ zmapowac ponizsza petle
            ##################### TODO DO ZMAPOWANIA
            intensity = [
                self.integrate_spectrum(np.array(spectra_without_bgr[i]), range_)
                for i in spectra_without_bgr
            ]

    def make_df(self):
        df = pd.DataFrame()
        df["discharge_time"] = selected_time_stamps
        df[f"QSO_{self.element}_{self.date}.{self.discharge_nr}"] = intensity
        df[f"QSO_{self.element}_{self.date}.{self.discharge_nr}"] = df[
            f"QSO_{self.element}_{self.date}.{self.discharge_nr}"
        ].round(1)
        df["utc_timestamps"] = time_stamps
        df = df.iloc[:-1]
        ### usuwa p[ierwsza ramke w czasie 0s -> w celu usuniecia niefizycznych wartosci

        time = list(map(get_time_from_UTC, df["utc_timestamps"]))
        # breakpoint()
        x_labels = [
            date.strftime("%H:%M:%S.%f")[:-3] if date.microsecond % 1_000 == 0 else ""
            for date in time
        ]
        df["time"] = x_labels
        return df

    def save_file(self):
        destination = (
            pathlib.Path(__file__).parent.parent.resolve()
            / "data"
            / "time_evolutions"
            / f"{self.element}"
            / f"{self.date}"
        )
        destination.mkdir(parents=True, exist_ok=True)
        self.df2.to_csv(
            destination
            / f"QSO_{self.element}_{self.date}.{self.discharge_nr:03}-{file_name.stem}-time_{min(self.time_interval )}_{max(self.time_interval )}s.csv",
            sep="\t",
            index=False,
            header=True,
        )
        print(
            f"QSO_{self.element}_{self.date}.{self.discharge_nr:03} - intensity evolution saved!"
        )

    def plot(self):
        fig, ax1 = plt.subplots()
        ax1.set_title(
            f"Evolution of the C/O monitor signal intensity.\n {self.date}.{self.discharge_nr:03}"
        )
        ax2 = ax1.twiny()
        ax1.plot(
            pd.to_datetime(self.df2["time"], format="%H:%M:%S.%f", errors="coerce"),
            self.df2[f"QSO_{self.element}_{self.date}.{self.discharge_nr}"],
            alpha=0,
        )

        ax2.plot(
            np.asarray(self.df2["discharge_time"], float),
            self.df2[f"QSO_{self.element}_{self.date}.{self.discharge_nr}"],
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
            / f"{self.element}"
            / f"{self.date}"
            / "img"
        )
        ax1.grid(which="major")
        plt.tight_layout()  # Dostosowanie rozmiaru obszaru wykresu
        destination.mkdir(parents=True, exist_ok=True)
        # plt.savefig(
        #     destination
        #     / f"QSO_{self.element}_{self.date}.{self.discharge_nr:03}-{file_name.stem}-time_{min(time_interval)}_{max(time_interval)}s.png",
        #     dpi=200,
        # )
        plt.show()
