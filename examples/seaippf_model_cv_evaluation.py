if __name__ == "__main__": import __config__
import pandas as pd
import numpy as np
from math import ceil, sqrt
from datasets import utils
from SEAIPPF.Model import Model as SEAIPPFModel
from SEAIPPF.Transformers.TransformerTest import TransformerTest
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sktime.forecasting.model_evaluation import evaluate
from utils.Plotter import Plotter
from sklearn.metrics import r2_score
from sktime.split import SlidingWindowSplitter
from sklearn.neural_network import MLPRegressor
from sktime.performance_metrics.forecasting import MeanAbsoluteError


def combine_evaluation_data(evaluation_return_data):
    data_columns = ["y_test","y_pred"]
    for dc in data_columns:
        if not dc in evaluation_return_data.columns:
            raise ValueError("Incorrect evaluation_return_data passed. Ensure return_data=True in sktime.forecasting.model_evaluation.evaluate function")

    #clear statistic table and data
    train_test_data = evaluation_return_data[data_columns]
    evaluation_return_data = evaluation_return_data.drop(columns=data_columns + ["y_train"])

    data_columns_series = []
    for dc in data_columns:
        y = pd.concat(train_test_data[dc].to_list(), axis=1)

        # y.fillna(0, inplace=True)
        y.columns = [f"{dc}{i}" for i, _ in enumerate(y.columns)]
        y = y.agg(np.nanmax, axis="columns")
        data_columns_series.append(y)
    df = pd.DataFrame({k:i.values for k,i in zip(data_columns, data_columns_series)})
    return df, evaluation_return_data

if __name__ == "__main__":
    data, ts = utils.load_pv(convert_index_to_time=True)
    start=0 #288*110
    end=288*400
    data, ts = data[start:end], ts[start:end]
    data[data > 15] = 0
    model = SEAIPPFModel(
        latitude_degrees=utils.LATITUDE_DEGREES,
        longitude_degrees=utils.LONGITUDE_DEGREES,
        x_bins=70,
        y_bins=70,
        interpolation=True,
        enable_debug_params= True,
        transformer=TransformerTest(
            regressor_degrees=11,
            sklearn_regressor = LinearRegression())

            #sklearn_regressor = MLPRegressor(hidden_layer_sizes=(15,15), activation="relu",  max_iter=10000, tol=1e-4, verbose=False))
            # sklearn_regressor = MLPRegressor(hidden_layer_sizes=(20,20,15,15), activation="relu",  max_iter=4000, verbose=True))
            # sklearn_regressor = MLPRegressor(hidden_layer_sizes=(25,20,20), activation="logistic",  max_iter=10000, verbose=True))
            # sklearn_regressor = MLPRegressor(hidden_layer_sizes=(5,5,5), activation="logistic",  max_iter=10000, verbose=True)

    )
    window_length = 288 * 365
    cv = SlidingWindowSplitter(window_length=window_length, step_length=288, fh=list(range(1,288+1)))

    results = evaluate(forecaster=model, y=data["Production"], cv=cv, return_data=True, scoring=MeanAbsoluteError())
    return_data,results = combine_evaluation_data(results)

    print(results.head())
    print(results.agg(np.mean, axis=0))
    # print(results)
    model.fit(data[0:window_length])
    print("R2: ", r2_score(return_data["y_pred"], return_data["y_test"]))
    model.plot()

    plotter = Plotter(return_data.index, [return_data[col].values for col in return_data.columns], debug=False)
    plotter.show()
    plt.show()
