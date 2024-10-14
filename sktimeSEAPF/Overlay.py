from sklearn.neighbors import KernelDensity
import numpy as np

from utils.Plotter import Plotter
from matplotlib import pyplot as plt

class ApplyKde:
    def __init__(self, kernel, bandwidth, bins):
        self._kernel = kernel
        self._bandwidth = bandwidth
        self._bins = bins

    def __call__(self, a):
        b = a[~np.isnan(a)].reshape(-1, 1)
        if len(b) == 0:
            return np.array([0] * len(self._bins))

        kde = KernelDensity(kernel=self._kernel, bandwidth=self._bandwidth)
        kde = kde.fit(b.reshape(-1, 1))
        log_dens = kde.score_samples(self._bins)

        ret = np.exp(log_dens)

        return ret

class Overlay:
    def __init__(self, overlay, _y_bins, _bandwidth):
        self._overlay = overlay
        self._y_bins = _y_bins
        self._bandwidth = _bandwidth

        if np.all(np.isnan(self._overlay)):
            raise ValueError("Overlay: Only nans")

        self._heatmap = np.zeros((self._y_bins, self._overlay.shape[1]))
        for j in range(self._heatmap.shape[1]):
            a = self._overlay[:, j]
            self._heatmap[:, j] = np.histogram(a[~np.isnan(a)], bins=int(self._y_bins))[0]
            su = np.nansum(self._heatmap[:, j])
            if su != 0:
                self._heatmap[:, j] = self._heatmap[:, j] / su


        # plt.imshow(self._heatmap)
        # plt.show()

        self._max_value_in_overlay = np.nanmax(self._overlay) #overlay[~np.isnan(overlay)].max()
        r = (0, self._max_value_in_overlay)
        bins_no = int(self._y_bins)
        self._bins = np.array([r[0] + (r[1] - r[0]) * i / (bins_no - 1) for i in range(bins_no)]).reshape(-1, 1)  # bins len



        apply_kde = ApplyKde(kernel="gaussian", bandwidth=self._bandwidth, bins=self._bins)
        self._kde = np.zeros(self._heatmap.shape)
        for j in range(self._kde.shape[1]):
            self._kde[:, j] = apply_kde(self._overlay[:, j])

    @property
    def bins(self):
        return self._bins

    @property
    def overlay(self):
        return self._overlay

    @property
    def kde(self):
        return self._kde

    @property
    def heatmap(self):
        return self._heatmap

    def plot(self, mod1 = 0, mod2= 0, title=""):
        fig, ax, cols, rows = Plotter.plot_2D_histograms(self.heatmap, self.kde)
        if title != "":
            fig.suptitle(title)
        ts = list(range(self.kde.shape[0]))
        for i in range(self.kde.shape[1]):
            y = i // cols
            x = i % rows
            axis = ax[y,x]
            # print(self.zeros_filter_threshold(i, modifier=mod1))
            threshold1 = int(self.kde.shape[0] * self.zeros_filter_threshold(i, modifier=mod1) / self._max_value_in_overlay)
            axis.axvline(x=threshold1, color="r")
            threshold2 = self.density_based_filter_threshold(i, modifier=mod2)
            axis.plot(ts, [threshold2] * len(ts), color="r")

    def zeros_filter_threshold(self,i, modifier:float, skip_if_threshold_lower_than:float = 0.1):
        d = self._kde[:, i]
        mx = self._max_value_in_overlay
        # print("zeros_filter_threshold", self._max_value_in_overlay, )

        su = d.sum()
        if su != 0:
            threshold = sum(d * [self._max_value_in_overlay * i / len(d) for i,_ in enumerate(d)]) / d.sum()
        else:
            return 0 # if no data are available in d -> return 0 immediately
        mi = d.min()

        # if default threshold is very low then skip this process
        if threshold < (self._max_value_in_overlay * skip_if_threshold_lower_than):
            return 0

        if modifier < 0:
            modifier = -modifier
            threshold = modifier * mi + (1 - modifier) * threshold
        elif modifier > 0:
            threshold = modifier * mx + (1 - modifier) * threshold

        if all(threshold > self._overlay[:, i]):
            nanmx =  np.nanmax(self._overlay[:, i])
            threshold = nanmx
            #print(f"zeros_filter_threshold: No value left in the array if threshold={mx}! Reverting max {nanmx} for distribution {i} ")


        return threshold

    def apply_zeros_filter(self, modifier:float = 0, skip_if_threshold_lower_than=0.2):
        """
        Function applies lowpass filtering It checks whether the values are below the threshold. Those values
        which are below are cleared
        :param: modifier - a value from <-1;1> which allow adjusting. 0 is neutral value
        :param: min_weight_left - allow highpass filter only if more than min_weight_left initial weight left in the
                                  distribution
        """
        #print("bApply zeros", self._overlay)
        for i in range(self._overlay.shape[1]):
            threshold = self.zeros_filter_threshold(i, modifier=modifier,
                                                    skip_if_threshold_lower_than=skip_if_threshold_lower_than)
            highpass = Overlay.highpass_filter(self._overlay[:, i], threshold)
            #print(i, "threshold", threshold, "highpass", highpass, all(np.isnan(highpass)))
            if not np.all(np.isnan(highpass)):
                #print("self._overlay[:, i] = highpass")
                self._overlay[:, i] = highpass

        #print("Apply zeros", self._overlay)
        return Overlay(self._overlay, self._y_bins, self._bandwidth)

    def density_based_filter_threshold(self, i, modifier:float):
        # mx = np.nanmax(self._kde[:, i])
        mi = np.nanmin(self._kde[:, i])
        threshold = self._kde[:, i].mean()

        if modifier < 0:
            modifier = -modifier
            threshold = modifier * mi + (1 - modifier) * threshold

        return threshold

    def apply_density_based_filter(self, modifier:float):
        """
        Function applies density based filtering on the overlay. It checks which values have higher density / probability
        than threshold. Those values are passing. Other are cleared
        :param: modifier - a value from <-1;0> which allow adjusting. 0 is neutral value
        """
        for i in range(self._overlay.shape[1]):
            d = self._kde[:, i]
            passing_bools = d >= self.density_based_filter_threshold(i, modifier)
            bins_boundaries = np.array([[self._max_value_in_overlay * i / len(d), self._max_value_in_overlay * (i+1) / len(d)]  for i,_ in enumerate(d)])
            bins_boundaries = bins_boundaries[passing_bools]
            r = np.zeros((self._overlay.shape[0]), dtype=self._overlay.dtype)
            for j in range(self._overlay.shape[0]):
                exists_in = False
                for _,boundaries in enumerate(bins_boundaries):
                    if boundaries[0] <= self._overlay[j, i] <= boundaries[1]:
                        exists_in = True

                r[j] = self._overlay[j, i] if exists_in else np.nan
            #revert if incorrect values
            if not all(np.isnan(r)):
                self._overlay[:, i] = r

        return Overlay(self._overlay, self._y_bins, self._bandwidth)


    @staticmethod
    def highpass_filter(data: np.ndarray, threshold: float):
        """
            The function apply high pass filter below the given threshold. It removes all values from overlay those not
            excesses the threshold
        """

        # data[data < threshold] = np.nan
        d = data.copy()
        d[d < threshold] = np.nan # remove from the overlay (observations) that are lower than threshold
        return d

    @staticmethod
    def lowpass_filter(data: np.ndarray, threshold: float):
        """
            The function apply low pass filter above the given threshold. It removes all values from overlay those
            excesses the threshold
        """
        data[data > threshold] = np.nan
        return data
