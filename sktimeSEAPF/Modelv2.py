from __future__ import annotations  # type or "|" operator is available since python 3.10 for lower python used this line
# lib imports
import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sklearn.metrics import r2_score
from utils import Solar
from utils.Plotter import Plotter
from matplotlib import pyplot as plt
from datetime import datetime as dt
from typing import List
# package imports
from .Optimized import Optimized

from .Overlay import Overlay
from .Model import Model as basemodel

class Model(basemodel):
    """This class implements model with different prediction method. Not iterative. """
    def __init__(self,
                 latitude_degrees: float = 51,
                 longitude_degrees: float = 14,
                 x_bins: int = 10,
                 y_bins: int = 10,
                 bandwidth: float = 0.4,
                 window_size: int = None,
                 enable_debug_params: bool = True,
                 zeros_filter_modifier:float=0,
                 density_filter_modifier:float=0,
                 interpolation=False,
                 return_sequences = False,
                 ):

        super().__init__(
            latitude_degrees =latitude_degrees,
            longitude_degrees = longitude_degrees,
            x_bins = x_bins,
            y_bins = y_bins,
            bandwidth = bandwidth,
            window_size = window_size,
            enable_debug_params = enable_debug_params,
            zeros_filter_modifier = zeros_filter_modifier,
            density_filter_modifier = density_filter_modifier,
            interpolation = interpolation,
            return_sequences = return_sequences
        )

    def get_model_value(self, x, xfactor_a=1, xfactor_b=0, yfactor=1):
        """
        Function estimates actual model value for real-domain x argument based in the tabular (vector) formed model representation
        :param x: point for each model representation should be estimated
        :param xfactor_a: a in equation model value = f(a * x + b) * y, where f is estimation function
        :param xfactor_b: b in equation model value = f(a * x + b) * y, where f is estimation function
        :param yfactor: y model value = f(a * x + b) * y, where f is estimation function
        :return: actual model value for x
        """
        mx = len(self.model_representation_)
        return np.interp(xfactor_a * x * mx + xfactor_b, xp=list(range(mx)), fp=self.model_representation_) * yfactor

    def get_model_values(self,x, xfactor_a=None, xfactor_b=None, yfactor=None):
        """
        Function estimates actual model values for real-domain X arguments based in the tabular (vector) formed model representation
        :param x: 1-D array for which model values shoudl be estimated
        :param xfactor_a: 1-D list of a part factor model value = f(a * x + b) * y, where f is estimation function
        :param xfactor_b: 1-D list of b part factor model value = f(a * x + b) * y, where f is estimation function
        :param yfactor: 1-D list of y part factor model value = f(a * x + b) * y, where f is estimation function
        :return: list of actual model values for each x_i
        """
        if xfactor_a is None: xfactor_a = np.ones(len(x))
        if xfactor_b is None: xfactor_b = np.zeros(len(x))
        if yfactor is None: yfactor = np.ones(len(x))

        return np.array([self.get_model_value(x,xfa,xfb,yf) for x,xfa,xfb,yf in zip(x,xfactor_a, xfactor_b,yfactor)])

    def _predict(self, fh, X):
        ts = np.array([i*self.x_time_delta_ + self.cutoff for i in fh]).flatten()
        timestamps = (ts.astype(int) // 10**9).astype(int)
        elevation = Solar.elevation(Optimized.from_timestamps(timestamps), self.latitude_degrees,
                                    self.longitude_degrees) * 180 / np.pi

        day_progress = np.zeros(len(elevation))
        current_counter = 1
        for i,e in enumerate(elevation):
            if e > 0:
                day_progress[i] = current_counter
                current_counter += 1
            else:
                current_counter = 0

        #pred_bins = Optimized.model_assign_prediction_bins(self.model_representation_, self.elevation_bins_, elevation)

        global_max_zenith = Solar.sun_maximum_positive_elevation(self.latitude_degrees)
        zenith = Solar.zenith_elevation(timestamps, self.latitude_degrees)
        sunrise = Solar.sunrise_timestamp(timestamps, self.latitude_degrees)
        sunset = Solar.sunset_timestamp(timestamps, self.latitude_degrees)
        day_len = (sunset - sunrise) * 60 * 60 / self.x_time_delta_.total_seconds() # in number of samples
        day_progress_factor = day_progress / day_len

        mx_model_representation = max(self.model_representation_)
        y_scale_factor = (zenith / global_max_zenith) #* (self.max_seen / mx_model_representation)
        prediction = self.get_model_values(day_progress_factor, yfactor=y_scale_factor)
        self.debug_data_ = pd.DataFrame(data={
            "Elevation": elevation,
            #"Bins": pred_bins,
            "Zenith": zenith,
            "hra": Solar.hra_timestamp(timestamps, self.longitude_degrees),
            "sunset": sunset,
            "sunrise": sunrise,
            # "yFactor": 100 * y_scale_factor,
            # "xFactor" : 100 * day_len / Solar.longest_day(self.latitude_degrees), # time scale factor
            "day_progress": 100 * day_progress_factor,
            "prediction": prediction,
            "base_prediction": self.get_model_values(day_progress_factor)
        }, index=pd.DatetimeIndex(ts), dtype=float)


        pred = pd.Series(index=pd.DatetimeIndex(ts), data=prediction, dtype=float)
        pred.iloc[self.debug_data_["Elevation"] < 0] = 0
        pred.name = "Prediction"
        pred.index.name = "Datetime"

        #clear
        self.debug_data_["Elevation"].iloc[self.debug_data_["Elevation"] < 0] = 0


        return pred

        # data, bins = self._predict_step(elevation)
        #
        # debug_datas = pd.DataFrame(data={"Bins": bins, "Elevation": elevation}, index=pd.DatetimeIndex(ts), dtype=float)
        # debug_datas.name = "Debug Data"
        # debug_datas.index.name = "Debug"
        # self.debug_data_ = debug_datas
        #
        # pred = pd.Series(index=pd.DatetimeIndex(ts), data=data, dtype=float)
        #
        # pred.name = "Prediction"
        # pred.index.name = "Datetime"
        # return pred



    def __str__(self):
        # return "Model representation: " + str(self.model_representation_) + \
        #     " len(" + str(len(self.model_representation_)) + ")" + \
        #     "\nBins: " + str(self.elevation_bins_) + " len(" + str(len(self.elevation_bins_)) + ")"
        return "SEAPFv2 (" + str(self.get_params()) + ")"