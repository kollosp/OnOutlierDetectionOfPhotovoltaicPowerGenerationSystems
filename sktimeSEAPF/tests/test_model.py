import unittest
import os, sys

# setup relative import
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sktimeSEAPF.Model import Model
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
        days = 80 * 288
        self.elevation_bins = 90
        self.selected = self.dataset.iloc[shift:shift + days]
        self.test_selected = self.dataset.iloc[shift + days:shift + days*2]

        self.model = Model(
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

    def test_2_fit_model(self):
        selected = self.selected.copy()
        # print(selected)

        self.assertEqual(np.all(np.isnan(self.model.overlay.overlay[:, 40:50])), True)

    def test_3_model_interference(self):
        test_len = len(self.test_selected)
        fh = list(range(test_len))
        prediction = self.model.predict(fh=fh)

        self.test_selected.insert(1, "Prediction", prediction.values, True)

        # prediction.insert(0, self.test_selected["Production"], True)
        plotter = Plotter(self.test_selected.index, [self.test_selected["Production"], self.test_selected["Prediction"]], debug=True)
        self.model.plot()
        plotter.show()
        plt.show()

if __name__ == '__main__':
    unittest.main()