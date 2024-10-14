if __name__ == "__main__": import __config__
import pandas as pd
import numpy as np
from math import ceil, sqrt
from datasets import utils
from SEAIPPF.Model import Model as SEAIPPFModel
from SEAIPPF.Transformers.TransformerTest import TransformerTest
from SEAIPPF.Transformers.TransformerSimpleFiltering import TransformerSimpleFiltering
from matplotlib import pyplot as plt
from utils.Evaluate import Evaluate
from utils.Plotter import Plotter
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sktimeSEAPF.Optimized import Optimized

if __name__ == "__main__":
    data, ts = utils.load_pv(convert_index_to_time=True)
    start=288*150
    # data, ts = data[start:], ts[start:]
    data[data > 15] = 0
    model = SEAIPPFModel(
        latitude_degrees=utils.LATITUDE_DEGREES,
        longitude_degrees=utils.LONGITUDE_DEGREES,
        x_bins=180,
        y_bins=180,

        interpolation=True,
        enable_debug_params= True,
        transformer=TransformerSimpleFiltering(
            regressor_degrees=11,
            hit_points_max_iter=2,
            hit_points_neighbourhood=14,
            sklearn_regressor= LinearRegression())
            # sklearn_regressor = MLPRegressor(hidden_layer_sizes=(15,15), activation="relu",  max_iter=10000, tol=1e-5, verbose=True))
            # sklearn_regressor = MLPRegressor(hidden_layer_sizes=(20,20,15,15), activation="relu",  max_iter=4000, verbose=True))
            # sklearn_regressor = MLPRegressor(hidden_layer_sizes=(25,20,20), activation="logistic",  max_iter=10000, verbose=True))
            # sklearn_regressor = MLPRegressor(hidden_layer_sizes=(5,5,5), activation="logistic",  max_iter=10000, verbose=True)
    )

    shift = 0
    train_test_split = 288*365 + shift
    test_len = 288*400+shift

    y_train, y_test = data["Production"][shift:train_test_split], data["Production"][train_test_split:train_test_split+test_len]

    model.fit(y=y_train)
    fh = [i for i in range(0,test_len)]

    pred = model.predict(fh=fh)
    print(pred)
    model.plot()
    model.transformer.plot()

    y_test_avg = Optimized.window_moving_avg(y_test.values, window_size=7, roll=True)

    # print_data = pd.DataFrame({'s1': data["Production"], 's2': pred})
    # print_data.plot()
    print("Result R2:", r2_score(y_test, pred.values))
    print("Result MAE:", mean_absolute_error(y_test, pred.values))
    # plotter = Plotter(pred.index, [y_test, pred.values, model.debug_data["Elevation"].values], debug=True)
    plotter = Plotter(pred.index, [y_test_avg, y_test, pred.values], debug=False)
    plotter.show()
    plt.show()
