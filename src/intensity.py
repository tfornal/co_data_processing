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


class DischargeNumbers:
    def __init__(self, element, date, file_name):
        self.element = element
        self.date = date
        self.file_name = file_name
        self.discharge_nr_file_path = self._get_specific_file_path()
        self.discharge_data = self.get_discharge_parameters()

    def _get_specific_file_path(self):
        return FilePaths(self.element, self.date).discharge_nrs()

    def get_discharge_parameters(self):
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
            discharge_data = df.loc[df["file_name"] == self.file_name.stem]
        return discharge_data


class ExperimentalFiles:
    def __init__(self, element, date, discharge_nr):
        self.element = element
        self.date = date
        self.discharge_nr = discharge_nr

        self.fp = self._get_file_path_object()
        self.exp_data_file_path = self._get_exp_data_file_path()
        self.file_list = self._grab_file_list()
        self.discharge_nr_file_path = self._get_specific_file_path()
        self.selected_file_names = self._select_file_names()
        self.bgr_files = self._grab_bgr_files()
        self.discharge_files = self._grab_discharge_files()

    def _get_file_path_object(self):
        return FilePaths(self.element, self.date)

    def _get_exp_data_file_path(self):
        return self.fp.experimental_data()

    def _grab_file_list(self):
        return list(self.exp_data_file_path.glob("**/*"))

    def _grab_bgr_files(self):
        bgr_files = [
            x
            for x in self._grab_file_list()
            if "BGR" in x.stem in self.selected_file_names
        ]
        return bgr_files

    def _get_specific_file_path(self):
        return self.fp.discharge_nrs()

    def _select_file_names(self):
        df = pd.read_csv(self.discharge_nr_file_path, sep="\t")
        if self.discharge_nr != 0:
            df["discharge_nr"] = df["discharge_nr"].replace("-", "0").astype(int)
            selected_file_names = df.loc[df["discharge_nr"] == self.discharge_nr][
                "file_name"
            ].to_list()

            return selected_file_names

    def _grab_discharge_files(self):
        discharge_files = [
            x
            for x in self.file_list
            if x.stat().st_size > 8000
            and x.stem in self.selected_file_names
            and "BGR" not in x.stem
        ]

        return discharge_files


class Intensity:
    def __init__(
        self, element, date, discharge_nr, file_name, time_interval, plotter=True
    ):
        self.element = element
        self.date = date
        self.discharge_nr = discharge_nr
        self.file_name = file_name

        self.exp_info_df = self._get_discharge_info()
        self.utc_time_of_saved_file = self._get_utc_time()
        self.frequency = self._get_frequency()
        self.bgr_files = self._get_bgr_files()

        self.time_interval = self.check_if_negative(time_interval)
        self.integral_range = self.select_integral_range()
        self.dt = self.convert_frequency_to_dt()
        self.spectra = self.get_all_spectra()
        self.spectra_without_bgr, self.selected_time_stamps = self.calculate_intensity()
        self.utc_time_stamps = self.convert_to_utc_time_stamps(
            self.utc_time_of_saved_file, self.selected_time_stamps
        )
        self.intensity = self.get_intensity()

        self.df = self.make_df(save=True)
        if plotter:
            self.plot_results(save=True)

    def get_intensity(self):
        intensity = [
            self.integrate_spectrum(
                np.array(self.spectra_without_bgr[i]), self.integral_range
            )
            for i in self.spectra_without_bgr
        ]
        return intensity

    def _get_discharge_info(self):
        exp_info_df = DischargeNumbers(
            self.element, self.date, self.file_name
        ).discharge_data
        return exp_info_df

    def _get_utc_time(self):
        return int(self.exp_info_df["utc_time"].iloc[0])

    def _get_frequency(self):
        return int(self.exp_info_df["frequency"].iloc[0])

    def _get_bgr_files(self):
        ef = ExperimentalFiles(self.element, self.date, self.discharge_nr)
        return ef.bgr_files

    def check_if_negative(self, numbers_list):
        numbers_list = [num if num >= 0 else 0 for num in numbers_list]
        return numbers_list

    def select_integral_range(self):
        ranges_dict = {"C": [120, 990], "O": [190, 941]}
        integral_range = ranges_dict[f"{self.element}"]
        return integral_range

    def convert_frequency_to_dt(self):
        return 1 / self.frequency

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

    def get_BGR(self, bgr_file_name):
        with open(bgr_file_name, "rb") as binary_file:
            _, cols_number = self.get_det_size(binary_file)
            spec_in_time = pd.DataFrame()
            spectrum = self.get_spectrum(binary_file, cols_number, row=1)
            spec_header = f"{bgr_file_name.stem}"
            spec_in_time[spec_header] = spectrum
            col_name = spec_header

        return col_name, spectrum

    def get_all_spectra(self):
        # wyznacza intensywnosci wybranych przerzialod czasowych (time interval)
        ### TODO test!!!!!!
        with open(self.file_name, "rb") as binary_file:
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

    def convert_to_utc_time_stamps(self, utc_time, timestamps_list):
        frames_nr = len(timestamps_list)
        ns_time_stamps = [int(i * self.dt * 1e9) for i in range(frames_nr)]
        ns_time_stamps.reverse()
        removed = [utc_time - i for i in ns_time_stamps]

        return removed

    def calculate_intensity(self):
        ########### TODO  time_interval automatycznie dostoswany do rozmiarów pliku - using get_det_size

        ### takes last recorded noise signal before the discharge
        # bgr_files = Files(element, date, discharge).bgr_files

        try:
            _, bgr_spec = self.get_BGR(self.bgr_files[-1])

            spectra_without_bgr = self.spectra.iloc[:, :].sub(bgr_spec, axis=0)
            selected_time_stamps = self.generate_time_stamps(self.time_interval)[
                : spectra_without_bgr.shape[1]
            ]
        #### TODO background jesli go nie ma!!!
        except IndexError:
            spectra_without_bgr = self.spectra
            selected_time_stamps = self.generate_time_stamps(self.time_interval)[
                : spectra_without_bgr.shape[1]
            ]
            print("TODO BACKGROUND NOT REMOVED!!!!!!!!!!!!!!!!!!!!!!!!")

        return spectra_without_bgr, selected_time_stamps

    def make_df(self, save=True):
        df = pd.DataFrame()
        df["discharge_time"] = self.selected_time_stamps
        df[f"QSO_{self.element}_{self.date}.{self.discharge_nr}"] = self.intensity
        df[f"QSO_{self.element}_{self.date}.{self.discharge_nr}"] = df[
            f"QSO_{self.element}_{self.date}.{self.discharge_nr}"
        ].round(1)
        df["utc_timestamps"] = self.utc_time_stamps
        df = df.iloc[:-1]
        ### usuwa p[ierwsza ramke w czasie 0s -> w celu usuniecia niefizycznych wartosci

        time = list(map(get_time_from_UTC, df["utc_timestamps"]))
        x_labels = [
            date.strftime("%H:%M:%S.%f")[:-3] if date.microsecond % 1_000 == 0 else ""
            for date in time
        ]
        df["time"] = x_labels

        def save_file():
            path = FilePaths(self.element, self.date).time_evolutions()
            path.mkdir(parents=True, exist_ok=True)
            df.to_csv(
                path
                / f"QSO_{self.element}_{self.date}.{self.discharge_nr:03}-{file_name.stem}-time_{min(self.time_interval )}_{max(self.time_interval )}s.csv",
                sep="\t",
                index=False,
                header=True,
            )
            print(
                f"QSO_{self.element}_{self.date}.{self.discharge_nr:03} - intensity evolution saved!"
            )

        if save:
            save_file()
        return df

    def plot_results(self, save=True):
        fig, ax1 = plt.subplots()
        ax1.set_title(
            f"{self.element} Lyman-alpha - intensity evolution.\n {self.date}.{self.discharge_nr:03}"
        )
        ax2 = ax1.twiny()
        ax1.plot(
            pd.to_datetime(self.df["time"], format="%H:%M:%S.%f", errors="coerce"),
            self.df[f"QSO_{self.element}_{self.date}.{self.discharge_nr}"],
            alpha=0,
        )

        ax2.plot(
            np.asarray(self.df["discharge_time"], float),
            self.df[f"QSO_{self.element}_{self.date}.{self.discharge_nr}"],
            color="blue",
            label="discharge_time",
            linewidth=0.4,
        )
        ax1.tick_params(axis="x", rotation=45)
        ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax1.set_ylabel("Intensity [a.u.]")
        ax1.set_xlabel("Local time")
        ax2.set_xlabel("Discharge time [s]")
        ax1.grid(which="major")
        plt.tight_layout()

        def save_fig():
            path = FilePaths(self.element, self.date).images()
            path.mkdir(parents=True, exist_ok=True)

            plt.savefig(
                path
                / f"QSO_{self.element}_{self.date}.{self.discharge_nr:03}-{self.file_name.stem}-time_{min(self.time_interval)}_{max(self.time_interval)}s.png",
                dpi=200,
            )

        if save:
            save_fig()

        plt.show()


dates_list = ["20230118"]  # "20230307"
elements_list = ["O"]  # , "O"]
discharges_list = [20]
time_interval = [-12, 6]  ### ponizej 5s czas time jest zly? 29h... TODO

if __name__ == "__main__":
    for element in elements_list:
        for date in dates_list:
            for discharge in discharges_list:
                try:
                    f = ExperimentalFiles(element, date, discharge)
                    discharge_files = f.discharge_files
                    # breakpoint()
                    for file_name in discharge_files:
                        Intensity(
                            element,
                            date,
                            discharge,
                            file_name,
                            time_interval,
                            plotter=True,
                        )
                except FileNotFoundError:
                    print("No matching file found! Continue...")
                    continue
