import numpy as np
from datetime import datetime as dt
from typing import List, Tuple



class Optimized:
    def __init__(self):
        pass

    @staticmethod
    def model_assign_interpolated(model_representation: np.ndarray, model_bins: np.ndarray,
                                  elevation: np.ndarray) -> np.ndarray:
        mr = np.append([0], model_representation)
        mr = np.append(mr, [0])
        diff = np.append(np.diff(elevation), [0])
        bins_no = (len(model_bins) - 1) * 2
        # print(model_bins)
        # print(mr)

        ret = np.zeros((len(elevation)))

        for j, e in enumerate(elevation):
            for i in range(len(model_bins) - 1):
                if model_bins[i] < e < model_bins[i + 1]:
                    factor = (e - model_bins[i]) / (model_bins[i + 1] - model_bins[i])

                    if diff[j] > 0:
                        c = mr[i]
                        n = mr[i + 1]
                        ret[j] = factor * (n - c) + c
                        # print("(P) factor={0:.3f}, n={1:.3f},c={2:0.3f}, ret[j]={3:.3f}".format(factor, n, c, ret[j]))
                    else:
                        c = mr[bins_no - i + 1]
                        n = mr[bins_no - i + 2]

                        ret[j] = n + factor * (c - n)
                        # print("(N) factor={0:.3f}, n={1:.3f},c={2:0.3f}, ret[j]={3:.3f}".format(factor, n,c, ret[j]))
        return ret

    @staticmethod
    def model_assign(model_representation: np.ndarray, model_bins: np.ndarray, elevation: np.ndarray,
                     ret_bins: bool = True, interpolation=False) -> np.ndarray:
        # assign elevation bins
        pred_bins = np.digitize(elevation, model_bins)
        pred = pred_bins.copy()
        pred_next = pred_bins.copy()
        diff = np.append(np.diff(elevation), [0])

        # assign afternoon elevation bins
        pred[diff < 0] = (len(model_bins) - 1) * 2 - pred[diff < 0]
        pred[elevation <= 0] = 0

        # assign model's data to elevation bins
        ret = np.array([model_representation[p] for p in pred])

        if interpolation:
            # if interpolation enabled then compute upper range
            pred_next[diff < 0] = (len(model_bins) - 1) * 2 - pred_next[diff < 0]
            pred_next = pred + 1  # store for future use
            pred_next[pred_next >= len(model_representation)] = len(model_representation)-1

            pred_next[elevation <= 0] = 0
            ret_next = np.array([model_representation[p] for p in pred_next])
            # ret = ret_next

            # dirty hack to overcome issue. pred_bins after np.digitize contains values that are equals to len(model_bins)
            # which causes index error.
            pred_bins[pred_bins >= len(model_bins)] = len(model_bins) -1
            upper_bounds = model_bins[pred_bins]
            lower_bounds = model_bins[pred_bins-1]
            ranges = upper_bounds - lower_bounds # for elevation > 0 ranges contains correct values
            e = (elevation - lower_bounds) / ranges # for elevation > 0 e contains values from range <0,1)
            e2 = (upper_bounds - elevation) / ranges
            e[diff < 0] = e2[diff < 0]
            ret = (ret + e * (ret_next - ret)) # linear interpolation

        if ret_bins:
            return ret, pred
        else:
            return ret

    @staticmethod
    def overlay(data: np.ndarray, x_assignment: np.ndarray, y_assignment: np.ndarray) -> np.ndarray:
        x_no = np.unique(x_assignment)
        y_no = np.unique(y_assignment)
        heatmap = np.empty((len(y_no), len(x_no)))
        counts = np.empty((len(y_no), len(x_no)))
        heatmap[:] = np.nan
        counts[:] = 0

        for i, _ in enumerate(data):
            x, y = x_assignment[i], y_assignment[i]
            if np.isnan(heatmap[y, x]):
                heatmap[y, x] = data[i]
            else:
                heatmap[y, x] += data[i]
            counts[y, x] += 1

        # mean aggregating
        for i, _ in enumerate(heatmap):
            for j, _ in enumerate(heatmap[i]):
                if not np.isnan(heatmap[i, j]):
                    heatmap[i, j] = heatmap[i, j] / counts[i, j]

        return heatmap

    @staticmethod
    def digitize(data: np.ndarray, bins_no: int) -> np.ndarray:
        bins = np.linspace(min(data), max(data), int(bins_no / 2) + 1)
        # default assigment starts from 1
        d = np.digitize(data, bins)
        diff = np.append(np.diff(data), [0])
        d[diff < 0] = bins_no - d[diff < 0]
        d[data == 0] = 0

        # print("bins", bins)
        # print("digitize", np.unique(d))
        # print("digitize", d.tolist())

        return d, bins

    @staticmethod
    def date_day_bins(timestamps: List[int]) -> List[int]:
        """Function assigns each day to other bin"""
        if len(timestamps) == 0:
            return []
        first = timestamps[0]
        a = []
        seconds_per_days = 60 * 60 * 24
        for i, ts in enumerate(timestamps):
            a.append(int((ts - first) / seconds_per_days))

        return a

    @staticmethod
    def from_timestamps(ts: List[float]) -> List[dt]:
        return [dt.fromtimestamp(int(t)) for t in ts]

    @staticmethod
    def window_moving_avg(data: np.ndarray, window_size: int, roll: bool = True) -> np.ndarray:
        """ Function compute moving average using selected window """
        return Optimized.apply_window_function(np.mean, data, window_size, roll)

    @staticmethod
    def window_max(data: np.ndarray, window_size: int, roll: bool = True) -> np.ndarray:
        """ Function compute max using selected window """
        return Optimized.apply_window_function(np.max, data, window_size, roll)

    @staticmethod
    def window_min(data: np.ndarray, window_size: int, roll: bool = True) -> np.ndarray:
        """ Function compute max using selected window """
        return Optimized.apply_window_function(np.min, data, window_size, roll)

    @staticmethod
    def window_subtraction(data1: np.ndarray,data2: np.ndarray, window_size: int, roll: bool = True) -> np.ndarray:
        """ Function compute max using selected window """

        d = np.zeros(data1.shape)
        d[window_size:] = np.array([np.sum(data1[i - window_size:i] - data2[i - window_size:i]) for i in range(window_size, len(data1))])

        if roll:
            d = np.roll(d, -int(window_size / 2))

        return d

    @staticmethod
    def apply_window_function(func: callable, data: np.ndarray, window_size: int, roll: bool = True) -> np.ndarray:
        """
        Function performs the given function on windows
        :param func: function to be used f(d:np.ndarray) -> [float | int]
        :param data: array to be calculated
        :param window_size: size of the window
        :param roll: if True, then data are shifted after operation
        :return: array that contains the results of func for each window
        """
        d = np.zeros(data.shape)
        d[int(window_size):] = np.array([func(data[i - +int(window_size):i]) for i in range(int(window_size), len(data))])

        if roll:
            d = np.roll(d, -int(window_size / 2))

        return d

    @staticmethod
    def overall_cloudiness_index(data: np.ndarray, expected_from_model: np.ndarray, window_size: int) -> np.ndarray:
        ex = Optimized.window_moving_avg(expected_from_model, window_size=window_size, roll=True)
        sub = Optimized.window_moving_avg(data, window_size=window_size, roll=True) - ex
        # prod = Optimized.window_moving_avg(data, window_size=window_size, roll=True)
        # ex = Optimized.window_moving_avg(expected_from_model, window_size=window_size, roll=True)
        # sub[sub > 0] = 0
        # factor = ex / prod
        sub = -sub
        return sub

    @staticmethod
    def variability_cloudiness_index(data: np.ndarray,
                                     expected_from_model: np.ndarray,
                                     window_size:int,
                                     ma_window_size:int = None,
                                     debug:bool = False):
        if ma_window_size is None:
            ma_window_size = window_size
        # prod = Optimized.window_moving_avg(data, window_size=ma_window_size, roll=True)
        # ex = Optimized.window_moving_avg(expected_from_model, window_size=ma_window_size, roll=True)

        # factor = ex / prod


        diff = np.array(np.diff(data).tolist() + [0])
        mx = Optimized.window_max(diff, window_size)
        mi = Optimized.window_min(diff, window_size)
        s = Optimized.window_subtraction(mx, mi, window_size) # / window_size
        # s = s * factor
        s[np.isnan(s)] = 0
        s[np.isinf(s)] = 0

        return s


    @staticmethod
    def centroids(x:np.ndarray,y:np.ndarray,classes:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        class_list = np.unique(classes).astype(int)
        centroids = np.array([[x[classes == c].mean(),
                               y[classes == c].mean()] for c in class_list])
        return class_list, centroids

    @staticmethod
    def select(*args, indices:np.ndarray):
        ret = []
        for a in args:
            ret.append(a[indices])
        return tuple(ret)
