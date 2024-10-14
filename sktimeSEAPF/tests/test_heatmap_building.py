import unittest
import os, sys

# setup relative import
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sktimeSEAPF.Optimized import Optimized
from sktimeSEAPF.Model import Model
from utils import Solar
unittest.TestLoader.sortTestMethodsUsing = None

class TestHeatmapBuilding(unittest.TestCase):
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
        days = 5 * 288
        self.elevation_bins = 90
        self.selected = self.dataset.iloc[shift:shift + days]

    def test_check_dataset(self):
        self.assertEqual(len(self.dataset), 364638)
        self.assertEqual(self.dataset.columns, ["Production"])

    def test_heatmap_build(self):
        selected = self.selected.copy()
        timestamps = selected.index.astype(int) / 10**9
        ts = Optimized.from_timestamps(timestamps)
        elevation = Solar.elevation(ts, self.latitude_degrees,
                                    self.longitude_degrees) * 180 / np.pi



        # elevation[elevation <= 0] = 0
        # create assignment series, which will be used in heatmap processing
        days_assignment = Optimized.date_day_bins(timestamps)
        elevation_assignment, self.elevation_bins_ = \
            Optimized.digitize(elevation, self.elevation_bins, mi=0, mx=Solar.sun_maximum_positive_elevation(self.latitude_degrees))
        # print("elevation_assignment", elevation_assignment)
        # plt.imshow(overlay)

        selected.insert(1, "Elevation", elevation, True)
        selected.insert(1, "Days", days_assignment, True)
        selected.insert(1, "Elevation Assig", elevation_assignment, True)

        selected = selected[selected["Elevation"] >= 0]

        overlay = Optimized.overlay(selected["Production"],
                                    selected["Elevation Assig"],
                                    selected["Days"],
                                    y_bins=len(np.unique(selected["Days"])),
                                    x_bins=self.elevation_bins)


        # selected.plot()
        # plt.imshow(overlay)
        # plt.show()

        # check for empty space in the middle
        self.assertEqual(np.all(np.isnan(overlay[:, 40:50])), True)


    def test_model_heatmap_generation(self):
        selected = self.selected.copy()
        # print(selected)
        model = Model(
            latitude_degrees=self.latitude_degrees,
            longitude_degrees=self.longitude_degrees,
            x_bins=90,
            y_bins=90,
            bandwidth=0.5
        )
        model.fit(selected)

        self.assertEqual(np.all(np.isnan(model.overlay.overlay[:, 40:50])), True)

        # plt.imshow(model.overlay.overlay)
        # plt.imshow(model.overlay.kde)
        #
        # model.plot()
        #
        #
        # plt.show()


if __name__ == '__main__':
    unittest.main()