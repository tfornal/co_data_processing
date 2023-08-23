from datetime import datetime, timedelta
import time
from file_reader import (
    FilePathManager,
    DischargeFilesSelector,
    DischargeDataExtractor,
    BackgroundFilesSelector,
)
from intensity import Intensity
import numpy as np


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


# dates_list = generate_dates_list("20230117", "20230331")

if __name__ == "__main__":
    dates_list = ["20230331"]
    elements_list = ["C"]
    discharges_list = [49]
    time_interval = [-12, 500]  ### ponizej 5s czas time jest zly? 29h... TODO
    for element in elements_list:
        for date in dates_list:
            for discharge in discharges_list:
                try:
                    f = DischargeFilesSelector(element, date, discharge)
                    discharge_files = f.discharge_files

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
