import unittest
import os, sys

# setup relative import
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sktimeSEAPF.Modelv2 import Modelv2
from utils.Plotter import Plotter
unittest.TestLoader.sortTestMethodsUsing = None


class TestModel(unittest.TestCase):
    dataset, ts = None, None

    def setUp(self):
        path = os.path.join(os.getcwd(), "/".join(sys.argv[0].split("/")[:-1] + ["test_dataset.csv"]))

        df = pd.read_csv(path, header=None, sep=",", index_col=0)
        self.ts = df.index.to_numpy()
        df.index = pd.to_datetime(df.index, unit='s')
        df.index.names = ['Datetime']
        df.index.names = ['Datetime']
        self.dataset = df.rename(columns={1: "Production"})

        self.longitude_degrees = 14
        self.latitude_degrees = 51

        shift = 267
        train_days = 80 * 288
        test_days = 360 * 288
        self.elevation_bins = 90
        self.selected = self.dataset.iloc[shift:shift + train_days]
        self.test_selected = self.dataset.iloc[shift + train_days:shift + test_days+train_days]

        self.model = Modelv2(
            latitude_degrees=self.latitude_degrees,
            longitude_degrees=self.longitude_degrees,
            x_bins=90,
            y_bins=90,
            bandwidth=0.5,
            interpolation=True
        )
        self.model.fit(self.selected)

    def test_1_check_dataset(self):
        self.assertEqual(len(self.dataset), 364638)
        self.assertEqual(self.dataset.columns, ["Production"])

    def test_3_model_interference(self):
        test_len = len(self.test_selected)
        fh = list(range(test_len))
        prediction = self.model.predict(fh=fh)

        print(prediction)

        # self.test_selected.insert(1, "Prediction", prediction.values, True)
        #
        # # prediction.insert(0, self.test_selected["Production"], True)
        # plotter = Plotter(self.test_selected.index, [prediction[col] for col in prediction.columns], debug=True)

        plotter = Plotter(self.test_selected.index, [prediction["prediction"],prediction["base_prediction"], self.test_selected["Production"]], debug=True)
        # plotter = Plotter(self.test_selected.index, [prediction[col] for col in ["prediction", "day_progress",
        #                                                                          "Elevation", "hra", "sunrise", "sunset"]], debug=True)
        # self.model.plot()
        plotter.show()
        plt.show()

if __name__ == '__main__':
    unittest.main()