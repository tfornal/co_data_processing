from archive_uploader import establish_connection
from intensity import run_intensity_calculations
from file_reader import (
    FilePathManager,
    ExperimentalDataExtractor,
    ExperimentalFilesSelector,
    BackgroundFilesSelector,
)
import json
import pandas as pd

date = "20230119"
element = "C"
discharge = 23
time_interval = [0, 10000]
save_df = True
save_fig = True
plot = True

f = ExperimentalFilesSelector()
discharge_files = f.grab_discharge_files(element, date, discharge)


# def get_intensity(file_name):
#     return df


def grab_intensity_df_to_json(file_name):
    # df = get_intensity(file_name)
    _, df, spectra = run_intensity_calculations(
        element,
        date,
        discharge,
        file_name,
        time_interval,
        save_df,
        save_fig,
        plot,
    )
    dic = {
        "datatype": "float",
        "dimensions": list(df["utc_timestamps"]),
        "values": list(df["QSO_C_20230119.23"]),
    }

    intensity_json = json.dumps(dic)
    # with open("intensity.json", "w") as f:
    #     json.dump(intensity_json, f)
    return intensity_json


def grab_spectra_df_to_json(file_name):
    _, df, spectra = run_intensity_calculations(
        element,
        date,
        discharge,
        file_name,
        time_interval,
        save_df,
        save_fig,
        plot,
    )
    timestamps = df["utc_timestamps"].tolist()
    # [-1] - ostatnia ramka jest usunieta juiz na poziomie definiowania df (nie fizyczna);
    list_of_lists = [spectra[column].tolist() for column in spectra.columns[:-1]]
    slownik = {"values": list_of_lists, "dimensions": timestamps}
    data = json.dumps(slownik)
    # with open("spectra.json", "w") as f:
    #     json.dump(data, f)
    breakpoint()
    return data


for file_name in discharge_files:
    intensity_json = grab_intensity_df_to_json(file_name)
    spectra_json = grab_spectra_df_to_json(file_name)


############################# DALSZA CZESC Z W7X!!###########################33


"""
http://archive-webapi.ipp-hgw.mpg.de/#example 
powyzej link do typow danych 
1. multichannel lub profile dla kazdej z ramek - raw data
2 stream- signal intensities 
"""
import requests
import json
import pandas as pd
import numpy as np

url = "http://archive-webapi.ipp-hgw.mpg.de/Sandbox/raw/W7X/QW21/wwwwwww_DATASTREAM/"


def establish_connection():
    ...


def upload_intensity():
    data_file = "C:/Users/tofo/Desktop/Programs/arch_upl/test.csv"
    csv_data = pd.read_csv(data_file, sep="\t")
    df = csv_data[["utc_timestamps", "QSO_C_20221215.10"]]
    timestamps = csv_data["utc_timestamps"]

    dic = {
        "datatype": "float",
        "dimensions": list(df["utc_timestamps"]),
        "values": list(df["QSO_C_20221215.10"]),
    }

    data = json.dumps(dic)
    response = requests.post(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Content-length": "length"},
    )
    print(response.json())


def get_timestamps_list(liczba_ramek):
    df_timestamps = pd.read_csv("utc_time_stamps.csv", sep="\t")
    timestamps = list(df_timestamps.iloc[:liczba_ramek]["0"])

    return timestamps


def get_spectra_df(liczba_kolumn):
    cols = list(np.arange(liczba_kolumn + 1))
    spectra = pd.read_csv("spectra_example.csv", usecols=cols, sep="\t")
    return spectra


def write_parameters():
    dic = {
        "label": "parms",
        "values": [
            {
                "chanDescs": {
                    "[0]": {
                        "name": "Chan1",
                        "active": 1,
                        "physicalQuantity": {"type": "V"},
                    },
                    "[1]": {
                        "name": "Chan2",
                        "active": 1,
                        "physicalQuantity": {"type": "mV"},
                    },
                }
            }
        ],
        "dimensions": [1399090569999999999, 1399176969999999999],
    }
    print(dic)
    #### example
    #### http://archive-webapi.ipp-hgw.mpg.de/


write_parameters()


def upload_raw_profiles():
    # df_spectra = pd.read_csv("spectra_example.csv", sep = "\t")

    nr = 100

    timestamps = get_timestamps_list(nr)
    spectra = get_spectra_df(nr)
    list_spectra = [list(value) for key, value in spectra.iteritems()][1:]
    slownik = {"values": list_spectra, "dimensions": timestamps}
    data = json.dumps(slownik)
    response = requests.post(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Content-length": "length"},
    )
    print(response.json())


# upload_intensity()

# upload_raw_profiles()
