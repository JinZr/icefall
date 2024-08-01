from pathlib import Path

import pyedflib
from pyedflib.highlevel import read_edf


def edf_contents(edf_file: str):
    if edf_file is Path:
        edf_file = str(edf_file)
    signals, signal_headers, header = read_edf(edf_file)
    return signals, signal_headers, header


def edf_start_time(edf_file: str):
    if edf_file is Path:
        edf_file = str(edf_file)
    f = pyedflib.EdfReader(edf_file)
    start_time = f.getStartdatetime()
    f.close()
    return start_time


def edf_duration(edf_file: str):
    if edf_file is Path:
        edf_file = str(edf_file)
    f = pyedflib.EdfReader(edf_file)
    duration = f.getFileDuration()
    f.close()
    return duration
