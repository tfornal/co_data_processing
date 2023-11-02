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
    csv_data = pd.read_csv(data_file, sep = "\t")
    df = csv_data[["utc_timestamps", "QSO_C_20221215.10"]]
    timestamps = csv_data["utc_timestamps"]
    
    dic={"datatype":"float", "dimensions": list(df["utc_timestamps"]), "values" : list(df["QSO_C_20221215.10"])}
    
    
    data = json.dumps(dic)
    response = requests.post(url,
        data=data,
        headers={"Content-Type": "application/json", "Content-length":"length"},
    )
    print(response.json())


def get_timestamps_list(liczba_ramek):
    df_timestamps = pd.read_csv("utc_time_stamps.csv", sep = "\t")
    timestamps = list(df_timestamps.iloc[:liczba_ramek]["0"])
    
    return timestamps


def get_spectra_df(liczba_kolumn):
    cols = list(np.arange(liczba_kolumn+1))
    spectra = pd.read_csv("spectra_example.csv", usecols = cols, sep = "\t")
    return spectra

def write_parameters():
    dic = {
             "label" : "parms",
             "values" : [{
                 "chanDescs" : {
                     "[0]" : { "name" : "Chan1", "active" : 1, "physicalQuantity": {"type": "V"} },
                     "[1]" : { "name" : "Chan2", "active" : 1, "physicalQuantity": {"type": "mV"} }
                 }
             }],
             "dimensions" : [1399090569999999999, 1399176969999999999]
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
    slownik = {"values" : list_spectra, "dimensions":timestamps}
    data = json.dumps(slownik)
    response = requests.post(url, 
        data=data,
        headers={"Content-Type": "application/json", "Content-length":"length"},
    )
    print(response.json())




upload_intensity()

upload_raw_profiles()

