# import os, sys
# if not os.getcwd() in sys.path:
#     sys.path.append(os.getcwd())

from sktimeSEAPF.Modelv2 import Model as SEAPF
from utils import Solar
import numpy as np

class Model(SEAPF):
    def __init__(self,
                 latitude_degrees: float = 51,
                 longitude_degrees: float = 14,
                 x_bins: int = 10,
                 y_bins: int = 10,
                 window_size: int = None,
                 enable_debug_params: bool = False,
                 interpolation=False,
                 return_sequences = False,
                 zeros_filter_modifier: float = 0,
                 density_filter_modifier: float = 0,
                 transformer = None):
        """
        transformer: class that implements fit and transform methods according to the sklearn or the
        name of available predefined transformers. Transformer change 2D observation into 2D array
        [[0,1,2],
         [3,4,5],  ->  [3, 3, 3]
         [6,7,8],
         [0,1,2]]
        once the observation array is transformed it is then proceeded as Overlay in the Base model.
        """
        super().__init__(
            latitude_degrees= latitude_degrees,
            longitude_degrees = longitude_degrees,
            x_bins = x_bins,
            y_bins = y_bins,
            window_size = window_size,
            enable_debug_params = enable_debug_params,
            interpolation = interpolation,
            return_sequences = return_sequences,
            zeros_filter_modifier = zeros_filter_modifier,
            density_filter_modifier = density_filter_modifier
        )

        self.transformer=transformer

    def _fit(self, y, X=None, fh=None):
        """
        Fit function that is similar to sklearn scheme X contains features while y contains corresponding correct values
        :param X: it should be 2D array [[ts1],[ts2],[ts3],[ts4],...] containing timestamps
        :param y: it should be 2D array [[y1],[y2],[y3],[y4],...] containing observations made at the corresponding timestamps
        :param zeros_filter_modifier:
        :param density_filter_modifier:
        :return: self
        """

        self._fit_compute_statistics(y, X, fh)
        self.overlay_ = self._fit_generate_overlay(y, X, fh)

        # self.overlay_ = self.overlay_.apply_zeros_filter(modifier=self.zeros_filter_modifier) \
        #     .apply_density_based_filter(modifier=self.density_filter_modifier)

        kde = self.overlay_.kde
        if self.transformer is not None:
            kde = self.transformer.fit(kde).transform(kde)
        else:
            print("SEAIPPF model: self.transformer should not be None. Pass one of available transformers to constructor")

        self.model_representation_ = np.apply_along_axis(lambda a: self.overlay_.bins[np.argmax(a)], 0, kde).flatten()
        # print(self.model_representation_)
        return self


    def get_params_info(self):
        params = {
            "x_bins": (10, 10, 80, True),
            "y_bins": (10, 10, 80, True),
        }

        if self.transformer is not None:
            transformer_params = self.transformer.get_params_info()
            for tp in transformer_params:
                params["transformer__" + tp] = transformer_params[tp]
        return params
