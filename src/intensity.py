from functools import wraps

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate

# from yaml import load, dump
from file_reader import (
    FilePathManager,
    DischargeDataExtractor,
    BackgroundFilesSelector,
)
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
    def __init__(
        self,
        element,
        date,
        discharge_nr,
        file_name,
        time_interval,
        save_df=True,
        save_fig=True,
        plot=False,
    ):
        self.element = element
        self.date = date
        self.discharge_nr = discharge_nr
        self.file_name = file_name

        self.file_path_manager = FilePathManager(self.element, self.date)
        self.exp_info_df = DischargeDataExtractor(
            self.element, self.date, self.file_name
        ).discharge_data

        self.utc_time_of_saved_file = self._get_utc_time()
        self.frequency = self._get_frequency()
        self.bgr_files = self._get_bgr_files()

        self.time_interval = self.check_if_negative(time_interval)
        self.integral_range = self.select_integral_range()
        self.dt = self.convert_frequency_to_dt()
        self.spectra = self.get_all_spectra()
        self.spectra_without_bgr, self.selected_time_stamps = self.calculate_intensity()

        # data = self.spectra.to_numpy()[:, :-560].T
        # # data = self.spectra_without_bgr.to_numpy()[:, :-560].T

        # time = self.selected_time_stamps[::50]
        # import matplotlib.pyplot as plt

        # # Stworzenie colormapy
        # plt.figure(figsize=(10, 8))
        # plt.imshow(
        #     data, cmap="jet", aspect="auto"
        # )  # 'viridis' to przykładowa mapa kolorów
        # plt.colorbar()  # Dodanie skali kolorów
        # plt.xlabel("Piksele")

        # plt.ylabel("Ramki czasowe")
        # plt.yticks(np.arange(len(time)), time)
        # plt.title("Colormap z macierzy danych")

        # plt.show()
        self.utc_time_stamps = self.convert_to_utc_time_stamps(
            self.utc_time_of_saved_file, self.selected_time_stamps
        )
        self.intensity = self.get_intensity()
        self.df = self.make_df(save_df)
        self.plot(plot, save_fig)

    def get_intensity(self):
        intensity = [
            self.integrate_spectrum(
                np.array(self.spectra_without_bgr[i]), self.integral_range
            )
            for i in self.spectra_without_bgr
        ]
        return intensity

    def _get_utc_time(self):
        return int(self.exp_info_df["utc_time"].iloc[0])

    def _get_frequency(self):
        return int(self.exp_info_df["frequency"].iloc[0])

    def _get_bgr_files(self):
        ef = BackgroundFilesSelector(self.element, self.date, self.discharge_nr)
        return ef.bgr_files

    @classmethod
    def check_if_negative(cls, numbers_list):
        numbers_list = [num if num >= 0 else 0 for num in numbers_list]
        return sorted(numbers_list)

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
        # Extract the specified range of the spectrum
        selected_spectrum = spectrum[spectrum_range[0] : spectrum_range[1] + 1]
        # Calculate the background level as the minimum value of the first and last data points in the range
        background = min(selected_spectrum[0], selected_spectrum[-1])
        # Subtract the background level from the spectrum (removing the background)
        spectrum_without_background = selected_spectrum - background
        # Create an array of pixel indices corresponding to the selected spectrum range
        pixels = np.arange(spectrum_range[0], spectrum_range[1] + 1)
        # Integrate the spectrum using Simpson's rule
        integral = integrate.simps(spectrum_without_background, pixels)
        return integral

    def get_det_size(self, binary_file):
        breakpoint()
        binary_file.seek(0)
        bites = binary_file.read(4)
        ncols = int.from_bytes(bites, "little")

        binary_file.seek(4)
        bites = binary_file.read(4)
        nrows = int.from_bytes(bites, "little")
        return nrows, ncols

    @classmethod
    def validate_time_duration(cls):
        ### TODO
        pass

    def generate_time_stamps(self, time_interval):
        start, end = time_interval
        selected_time_stamps = [
            "{:.3f}".format(i) for i in np.arange(start, end + self.dt, self.dt)
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
            ### opisac dlaczego tak!!!!!!!!!!!!!!!! TODO
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

    def make_df(self, save_df=True):
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

        def save():
            path = self.file_path_manager.time_evolutions()
            path.mkdir(parents=True, exist_ok=True)
            df.to_csv(
                path
                / f"QSO_{self.element}_{self.date}.{self.discharge_nr:03}-{self.file_name.stem}.csv",
                sep="\t",
                index=False,
                header=True,
            )
            print(
                f"QSO_{self.element}_{self.date}.{self.discharge_nr:03} - intensity evolution saved!"
            )

        if save_df:
            save()
        return df

    def plot(self, plot, save_fig):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()
        ax1.set_title(
            f"{self.element} Lyman-alpha - intensity evolution\n {self.date}.{self.discharge_nr:03}"
        )
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
        ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M:%S"))
        ax1.tick_params(axis="x", rotation=45)
        ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax1.set_ylabel("Intensity [a.u.]")
        ax1.set_xlabel("Local time")
        ax2.set_xlabel("Discharge time [s]")
        ax1.grid(which="major")
        plt.tight_layout()

        def save_plot():
            path = self.file_path_manager.images()
            path.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                path
                / f"QSO_{self.element}_{self.date}.{self.discharge_nr:03}-{self.file_name.stem}.png",
                dpi=200,
            )

        if save_fig:
            save_plot()
        if plot:
            plt.show()

        plt.close()
