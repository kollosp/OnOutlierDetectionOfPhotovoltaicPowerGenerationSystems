if __name__ == "__main__": import __config__
import pandas as pd
import numpy as np
from math import ceil, sqrt
from datasets import utils
from SEAIPPF.Model import Model as SEAIPPFModel
from SEAIPPF.Transformers.TransformerTest import TransformerTest
from matplotlib import pyplot as plt
from utils.Evaluate import Evaluate
from utils.Plotter import Plotter
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

if __name__ == "__main__":
    data, ts = utils.load_pv(convert_index_to_time=True)
    start=288*150
    data, ts = data[start:], ts[start:]
    data[data > 15] = 0
    model = SEAIPPFModel(
        latitude_degrees=utils.LATITUDE_DEGREES,
        longitude_degrees=utils.LONGITUDE_DEGREES,
        x_bins=70,
        y_bins=70,
        interpolation=True,
        enable_debug_params= True,
        transformer=TransformerTest(
            regressor_degrees=1,
            # sklearn_regressor = MLPRegressor(hidden_layer_sizes=(15,15), activation="relu",  max_iter=10000, verbose=True))
            # sklearn_regressor = MLPRegressor(hidden_layer_sizes=(20,20,15,15), activation="relu",  max_iter=4000, verbose=True))
            sklearn_regressor = MLPRegressor(hidden_layer_sizes=(25,20,20), activation="logistic",  max_iter=10000, verbose=False))
            # sklearn_regressor = MLPRegressor(hidden_layer_sizes=(5,5,5), activation="logistic",  max_iter=10000, verbose=True))
    )

    shift = 140
    train_test_split = 288*80 + shift
    test_len = 288*30+shift

    y_train, y_test = data["Production"][shift:train_test_split], data["Production"][train_test_split:train_test_split+test_len]

    model.fit(y=y_train)
    fh = [i-train_test_split for i in range(train_test_split, train_test_split+test_len)]

    pred = model.predict(fh=fh)
    print(pred)
    model.plot()
    model.transformer.plot()

    # print_data = pd.DataFrame({'s1': data["Production"], 's2': pred})
    # print_data.plot()
    print("Result R2:", r2_score(y_test, pred.values))
    plotter = Plotter(pred.index, [y_test, pred.values, model.debug_data["Elevation"].values], debug=True)
    plotter.show()
    plt.show()
