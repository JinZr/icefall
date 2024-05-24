from pathlib import Path

from pyedflib.highlevel import read_edf


def edf_contents(edf_file: str):
    if edf_file is Path:
        edf_file = str(edf_file)
    signals, signal_headers, header = read_edf(edf_file)
    return signals, signal_headers, header
