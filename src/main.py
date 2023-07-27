from datetime import datetime, timedelta
import time

from intensity import Intensity
from file_reader import (
    FilePathManager,
    DischargeFilesSelector,
    DischargeDataExtractor,
    BackgroundFilesSelector,
)


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


# start_date_str = "20221213"
# end_date_str = "20230405"
# dates_list = generate_dates_list(start_date_str, end_date_str)
# discharges_list = [i for i in range(1, 100)]


# elements_list = ["C", "O"]  # , "O"]
# # dates_list = ["20230118"]  # "20230307"
# # discharges = [20]
# time_interval = [-1, 1000]  ### ponizej 5s czas time jest zly? 29h...
# ### gdy mniej niz max dlugosc pliku - ucina poczatek widma - plik binarny tylem do przodu?
# ### przy czasie 0-3 s czasy sie sypia!!! TODO

# if __name__ == "__main__":
#     for element in elements_list:
#         for date in dates_list:
#             for discharge in discharges_list:
#                 try:
#                     f = DischargeFilesSelector(element, date, discharge)
#                     discharge_files = f.discharge_files
#                     # breakpoint()
#                     for file_name in discharge_files:
#                         Intensity(
#                             element,
#                             date,
#                             discharge,
#                             file_name,
#                             time_interval,
#                             plotter=True,
#                         )
#                 except FileNotFoundError:
#                     print("No matching file found! Continue...")
#                     continue


if __name__ == "__main__":
    dates_list = ["20230118"]
    elements_list = ["C"]
    discharges_list = [20]
    time_interval = [-12, 5]  ### ponizej 5s czas time jest zly? 29h... TODO

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
