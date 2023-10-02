from functools import wraps

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate

from file_reader import (
    FilePathManager,
    ExperimentalDataExtractor,
    ExperimentalFilesSelector,
    BackgroundFilesSelector,
)

from utc_converter import get_time_from_UTC

MAX_PIXEL_CAPACITY = 241_000
SATURATION_THRESHOLD = 1 # %

def timer(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        exec_time = time.time() - start
        print(f"Execution time: {exec_time:.2f}s")
        return result

    return wrapper

def get_intensity(spectra_without_bgr, integral_range):
    intensity = [
        integrate_spectrum(np.array(spectra_without_bgr[i]), integral_range)
        for i in spectra_without_bgr
    ]
    return intensity

def validate_detector_saturation(spectra_without_bgr):
    saturation = [is_saturated(np.array(spectra_without_bgr[i]))
    for i in spectra_without_bgr]
    return saturation

def get_utc_time(exp_info_df):
    return int(exp_info_df["new_time"].iloc[0])


def get_frequency(exp_info_df):
    return int(exp_info_df["frequency"].iloc[0])


def get_bgr_files(element, date, exp_nr, exp_info_df):
    ef = BackgroundFilesSelector()
    return ef.get_bgr_file_list(element, date, exp_nr)


def check_if_negative(numbers_list):
    numbers_list = [num if num >= 0 else 0 for num in numbers_list]
    return sorted(numbers_list)


def select_integral_range(element):
    ranges_dict = {"C": [120, 990], "O": [190, 941]}
    integral_range = ranges_dict[f"{element}"]
    return integral_range

def convert_frequency_to_dt(frequency):
    return 1 / frequency


def get_pixel_intens(binary_file_content, row, column):
    shift = 4096 + (row - 1) * 3072 + (column - 1) * 3
    binary_file_content.seek(shift)
    bytes_ = binary_file_content.read(3)
    return int.from_bytes(bytes_, "little")


def get_spectrum(binary_file_content, cols_number, row):
    return list(
        map(
            lambda column: get_pixel_intens(binary_file_content, row, column + 1),
            range(cols_number),
        )
    )

def is_saturated(spectrum):
    saturation_status_list = spectrum > MAX_PIXEL_CAPACITY
    saturated_pixel_count = np.count_nonzero(saturation_status_list)
    percentage = saturated_pixel_count/len(saturation_status_list) * 100
    if percentage > SATURATION_THRESHOLD:
        return True
    return False


def integrate_spectrum(spectrum, spectrum_range):
    ## TODO - background not removed -> procedure for saturation finding; 
    # Extract the specified range of the spectrum
    start_index = spectrum_range[0]  
    end_index = spectrum_range[1]

    selected_spectrum = spectrum[start_index : end_index + 1]
    # Calculate the background level as the minimum value of the first and last data points in the range
    background = min(selected_spectrum[0], selected_spectrum[-1])
    # Subtract the background level from the spectrum (removing the background)
    spectrum_without_background = selected_spectrum - background
    # Create an array of pixel indices corresponding to the selected spectrum range
    pixels = np.arange(start_index, end_index + 1)
    # Integrate the spectrum using Simpson's rule
    integral = integrate.simps(spectrum_without_background, pixels)
    
    return integral


def get_det_size(binary_file):
    binary_file.seek(0)
    bites = binary_file.read(4)
    ncols = int.from_bytes(bites, "little")

    binary_file.seek(4)
    bites = binary_file.read(4)
    nrows = int.from_bytes(bites, "little")
    return nrows, ncols


def validate_time_duration(cls):
    ### TODO
    ...


def generate_time_stamps(time_interval, dt):
    start, end = time_interval
    selected_time_stamps = ["{:.3f}".format(i) for i in np.arange(start, end + dt, dt)]
    return selected_time_stamps


def get_bgr(bgr_file_name):
    with open(bgr_file_name, "rb") as binary_file_content:
        _, cols_number = get_det_size(binary_file_content)
        spec_in_time = pd.DataFrame()
        spectrum = get_spectrum(binary_file_content, cols_number, row=1)
        spec_header = f"{bgr_file_name.stem}"
        spec_in_time[spec_header] = spectrum
        col_name = spec_header

    return col_name, spectrum


def get_all_spectra(file_name, time_interval, dt):
    # wyznacza intensywnosci wybranych przerzialod czasowych (time interval)
    ### TODO test!!!!!!
    with open(file_name, "rb") as binary_file:
        rows_number, cols_number = get_det_size(binary_file)
        ### opisac dlaczego tak!!!!!!!!!!!!!!!! TODO
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


def convert_to_utc_time_stamps(utc_time, timestamps_list, dt):
    frames_nr = len(timestamps_list)
    ns_time_stamps = [int(i * dt * 1e9) for i in range(frames_nr)]
    ns_time_stamps.reverse()
    removed = [utc_time - i for i in ns_time_stamps]
    return removed


def calculate_intensity(spectra, bgr_files, time_interval, dt):
    ########### TODO  time_interval automatycznie dostoswany do rozmiarów pliku - using get_det_size
    ### takes last recorded noise signal before the discharge
    # bgr_files = Files(element, date, discharge).bgr_files
    try:
        _, bgr_spec = get_bgr(bgr_files[-1])

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
        print("TODO BACKGROUND NOT REMOVED!!!!!!!!!!!!!!!!!!!!!!!!")
    return spectra_without_bgr, selected_time_stamps


def make_df(
    element,
    date,
    exp_nr,
    selected_time_stamps,
    intensity,
    saturation,
    utc_time_stamps,
    file_name,
    save_df=True,
):
    df = pd.DataFrame()
    df["discharge_time"] = selected_time_stamps
    df[f"QSO_{element}_{date}.{exp_nr}"] = intensity
    df[f"QSO_{element}_{date}.{exp_nr}"] = df[f"QSO_{element}_{date}.{exp_nr}"].round(1)
    df["utc_timestamps"] = utc_time_stamps
    df["saturation"] = saturation
    df = df.iloc[:-1] # excludes last timeframe to remove unphysical data 

    time = list(map(get_time_from_UTC, df["utc_timestamps"]))

    x_labels = [date.strftime("%H:%M:%S.%f")[:-4] for date in time]
    df["time"] = x_labels

    def save():
        fpm = FilePathManager()
        path = fpm.get_directory_for_time_evolutions(element, date)
        path.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            path / f"QSO_{element}_{date}.{exp_nr:03}-{file_name.stem}.csv",
            sep="\t",
            index=False,
            header=True,
        )
        print(f"QSO_{element}_{date}.{exp_nr:03} - intensity evolution saved!")

    if save_df:
        save()

    return df


def plotter(element, date, df, exp_nr, file_name, plot, save_fig):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    ax1.set_title(f"{element} Lyman-alpha - intensity evolution\n {date}.{exp_nr:03}")
    ax1.plot(
        pd.to_datetime(df["time"], format="%H:%M:%S.%f", errors="coerce"),
        df[f"QSO_{element}_{date}.{exp_nr}"],
        alpha=0,
    )

    #     # Tworzenie wykresu liniowego dla 'discharge_time'
    # plt.plot(df['discharge_time'], df['saturation'])

    # Ustalenie, które wartości mają być oznaczone jako True

    # Dodanie pionowych linii na wykresie
    # ax3.vlines(x=true_values, ymin=0, ymax=6E7, colors='r', linestyles='dashed')
    max_intensity = df[f"QSO_{element}_{date}.{exp_nr}"].max()
    ax2.fill_between(np.asarray(df["discharge_time"], float), 0, max_intensity, where=df["saturation"], color='red', alpha=0.4)


    ax2.plot(
        np.asarray(df["discharge_time"], float),
        df[f"QSO_{element}_{date}.{exp_nr}"],
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
        fpm = FilePathManager()
        path = fpm.get_directory_for_images(element, date)
        # path = file_path_manager.get_directory_for_images(element, date)
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            path / f"QSO_{element}_{date}.{exp_nr:03}-{file_name.stem}.png",
            dpi=200,
        )

    if save_fig:
        save_plot()
    if plot:
        plt.show()

    plt.close()


def run_intensity_calculations(
    element,
    date,
    exp_nr,
    file_name,
    time_interval,
    save_df=True,
    save_fig=True,
    plot=True,
):
    exp_info_df = ExperimentalDataExtractor()
    exp_info_df = exp_info_df.get_discharge_parameters(
        element,
        date,
        file_name,
    )
    utc_time_of_saved_file = get_utc_time(exp_info_df)
    frequency = get_frequency(exp_info_df)
    bgr_files = get_bgr_files(
        element,
        date,
        exp_nr,
        exp_info_df,
    )

    time_interval = check_if_negative(time_interval)
    integral_range = select_integral_range(element)
    dt = convert_frequency_to_dt(frequency)
    spectra = get_all_spectra(
        file_name,
        time_interval,
        dt,
    )
    spectra_without_bgr, selected_time_stamps = calculate_intensity(
        spectra,
        bgr_files,
        time_interval,
        dt,
    )
    utc_time_stamps = convert_to_utc_time_stamps(
        utc_time_of_saved_file,
        selected_time_stamps,
        dt,
    )
    intensity = get_intensity(
        spectra_without_bgr,
        integral_range,
    )
    saturation = validate_detector_saturation(spectra,
    )


    df = make_df(
        element,
        date,
        exp_nr,
        selected_time_stamps,
        intensity,
        saturation,
        utc_time_stamps,
        file_name,
        save_df,
    )
    plotter(
        element,
        date,
        df,
        exp_nr,
        file_name,
        plot,
        save_fig,
    )


def main():
    # TODO - checking whether the trigger informatin, assignment files (csv???) and discharge files do exists.
    # if not - raise warning! Or error. 
    time_interval = [0, 120.85]
    ### sprawic aby wybieranie przedzialu czasowego sprawialo ze wybiera odpowiednie pliki
    dates_list = ["20230118"]
    elements_list = ["C"]
    discharges_list = [7]  # 20230117.050 rowniez kiepsko
    # jesli discharge to 0 to wyrzuca blad - TODO
    for date in dates_list:
        for discharge in discharges_list:
            for element in elements_list:
                try:
                    f = ExperimentalFilesSelector()
                    discharge_files = f.grab_discharge_files(element, date, discharge)
                    for file_name in discharge_files:
                        run_intensity_calculations(
                            element,
                            date,
                            discharge,
                            file_name,
                            time_interval,
                        )
                except FileNotFoundError:
                    print(
                        f"{element}_{date}_{discharge} - No matching file found! Continue..."
                    )


if __name__ == "__main__":
    main()
