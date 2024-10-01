import unittest
import numpy as np
import matplotlib.pyplot as plt

class TestOverlayMethods(unittest.TestCase):
    def test_overlay_zeros_filter(self):
        np.random.seed(0) # ensure same results for each repetition
        x = np.random.poisson(lam=1, size=30000)
        hist, bins = np.histogram(x, bins=30)
        print(hist)

        plt.bar(bins[0:-1], hist)
        plt.show()




if __name__ == '__main__':
    unittest.main()