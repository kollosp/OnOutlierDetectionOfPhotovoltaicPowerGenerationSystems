import pandas as pd
import numpy as np
from typing import List

DATASET_FILEPATH = "datasets/database.csv"
PV_FILEPATH = "datasets/pv.csv"
LATITUDE_DEGREES= 51
LONGITUDE_DEGREES = 14

tssc = {
    "hour": 12,
    "day": 24*12,
}

def load_dataset(convert_index_to_time : bool = False):
    df = pd.read_csv(DATASET_FILEPATH, header=0, sep=";", index_col=0)
    ts = None
    if convert_index_to_time:
        ts = df.index.to_numpy()
        df.index = pd.to_datetime(df.index, unit='s')
    df.index.names = ['Datetime']
    df = df.rename(columns={"X1": "Production", "X2": "Demand", "RCE": "Price"})
    if ts is None:
        return df
    else:
        return df, ts

def load_pv(convert_index_to_time : bool = False):
    df = pd.read_csv(PV_FILEPATH,header=None, sep=",", index_col=0)
    ts = None
    if convert_index_to_time:
        ts = df.index.to_numpy()
        df.index = pd.to_datetime(df.index, unit='s')

    df.index.names = ['Datetime']
    df = df.rename(columns={1: "Production"})
    if ts is None:
        return df
    else:
        return df, ts

def timeseries_to_dataset(data: List[np.ndarray], window_size: int, ts=None, skip:int = None):
    """
    That function change list of timeseries (the same length) into an array of windows of specified size.
    ex1:
        A = [a1,a2,..,an] window_size = 2 ->
        [[a1,a2], [a2,a3], ..., [an-1,an]]
    ex2:
        A = [[a1,a2,...],[b1,b2,...]] window_size = 2 ->
        [[a1,b1, a2,b2], [a2,b2,a3,b3], ..., [an-1,bn-1,an,bn]]

    :param data: list of np.ndarray of the same length (may it be also 2D array)
    :param window_size: a number of observation selected as features
    :param ts: timestamps array (optional) if exists is appended as the first column to the dataset
    :param skip: parameter that allows to skip some windows (to adjust data obtained from different window_sizes)
    :return:
        [[a1, a2],
        [a2,a3],
        ...,
        [an-1,an]]

    The returned array represents observation and its features that is consists with the schema that is used in ML.

    """

    end = len(data[0]) - int(window_size)
    # override if passed
    if skip is not None:
        end = len(data[0]) - skip

    dss = [np.array([ts[i: i+window_size] for i in range(0, end)]) for ts in data]
    ret = np.hstack(dss)


    if ts is not None:
        ret = np.hstack([ts[:end].reshape(-1, 1), ret])

    return ret


"""
    Show dataset if the script was run directly instead of being loaded as package
"""
if __name__ == "__main__":
    data = load_dataset()
    print(data.head())