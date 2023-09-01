import datetime
import calendar
import requests
import pandas as pd
import pathlib
import pytz


def get_time_from_UTC(time_in_ns):
    time_in_seconds = time_in_ns // 1_000_000_000  # To seconds
    time_in_microseconds = (time_in_ns % 1_000_000_000) // 1000  # To microseconds
    time_in_milliseconds = time_in_ns % 1000  # To miliseconds

    utc_time = datetime.datetime.utcfromtimestamp(time_in_seconds)
    utc_time_with_microseconds = utc_time.replace(microsecond=time_in_microseconds)

    return utc_time_with_microseconds.time()


if __name__ == "__main__":
    time = get_time_from_UTC(1676457072510000000)
    print(time)
