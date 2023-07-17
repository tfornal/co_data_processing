import datetime
import calendar
import requests
import pandas as pd
import pathlib
import pytz


def get_time_from_UTC(time_in_ns):
    # Wartość czasu w nanosekundach

    # Tworzenie obiektu datetime na podstawie wartości czasu w sekundach i mikrosekundach
    time_in_seconds = time_in_ns // 1_000_000_000  # Zamiana na sekundy
    time_in_microseconds = (
        time_in_ns % 1_000_000_000
    ) // 1000  # Zamiana na mikrosekundy

    # breakpoint()
    time_in_milliseconds = time_in_ns % 1000  # Zamiana na milisekundy

    utc_time = datetime.datetime.utcfromtimestamp(time_in_seconds)
    utc_time_with_microseconds = utc_time.replace(microsecond=time_in_microseconds)

    # Wypisanie daty i godziny
    return utc_time_with_microseconds.time()


time_in_ns = 1676457072510000000
x = get_time_from_UTC(time_in_ns)
print(x)
