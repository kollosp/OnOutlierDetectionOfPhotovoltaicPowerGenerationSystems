if __name__ == "__main__": import __config__
import pandas as pd
import numpy as np
from math import ceil, sqrt
from datasets import utils
from SEAIPPF.Model import Model as SEAIPPFModel
from SEAIPPF.Transformers.TransformerTest import TransformerTest
from utils.Evaluate import Evaluate
from utils.Plotter import Plotter
from sklearn.metrics import r2_score

if __name__ == "__main__":
    data, ts = utils.load_pv(convert_index_to_time=True)


    result, evaluate = Evaluate.scipy_differential_evolution(
        data,
        ts,
        model=SEAIPPFModel(
            latitude_degrees=utils.LATITUDE_DEGREES,
            longitude_degrees=utils.LONGITUDE_DEGREES,
            x_bins=70,
            y_bins=70,
            transformer=TransformerTest()),
        include_parameters=["x_bins",
                            "y_bins",
                            "transformer__conv2Dx_shape_factor",
                            "transformer__conv2Dy_shape_factor",
                            "transformer__regressor_degrees"
    ])


