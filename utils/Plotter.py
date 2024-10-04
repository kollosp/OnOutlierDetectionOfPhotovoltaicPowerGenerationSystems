from __future__ import annotations  # type or "|" operator is available since python 3.10 for lower python used this line
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from typing import Protocol

from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates


class PlotterObject:
    """
    Class that stores information about plotted timeseries
    """
    def __init__(self, data:np.ndarray, unit:str = ""):
        self._data = data
        self._unit = unit

    @property
    def data(self):
        return self._data

    @property
    def unit(self):
        return self._unit

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

class Plotter:
    """
    Class for interactive plotting. It alows to move over the data
    """
    def __init__(self, x_axis:List | np.ndarray, list_of_data_or_plotter_object:List[np.ndarray | PlotterObject],
                 displayed_window_size:int = 1_000,
                 one_time_jump: int = 300,
                 debug:bool=False):
        self._list_of_data = list_of_data_or_plotter_object
        self._x_axis = x_axis
        self._current_position = 0
        self._one_time_jump = one_time_jump
        self._displayed_window_size = displayed_window_size
        self._fig, self._ax = None, None
        self._plots = []
        self._debug = debug
        self._data_updated = False

    @property
    def fig(self):
        return self._fig

    @property
    def x_axis(self):
        return self._x_axis

    @property
    def ax(self):
        return self._ax

    @property
    def current_position(self):
        return self._current_position

    @staticmethod
    def unit( obj):
        try:
            return obj.unit
        except:
            return ""

    def get_window(self, data:List | np.ndarray) -> Tuple[PlotterObject | List | np.ndarray]:
        start = self._current_position
        end = self._current_position + self._displayed_window_size

        if end > len(self._x_axis):
            end = len(self._x_axis) - 1

        return data[start:end]

    def repaint(self) -> None:
        x = self.get_window(self._x_axis)
        # repaint all plots (new windows)
        for pl, d in zip(self._plots, self._list_of_data):
            y = self.get_window(d)  # get window that should be displayed (a part of timeseries)
            pl.set_data(x, y)

            if self._debug:
                print(y)

        #update x-axis (without it the chart won't move)
        for ax in self._ax:
            ax.set_xlim(min(x), max(x))

        # update canvas
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def show(self) -> self:
        self._fig, self._ax = plt.subplots(1)

        myFmt = mdates.DateFormatter('%H:%M %d-%m-%y')
        self._ax.xaxis.set_major_formatter(myFmt)

        self._ax = [self._ax]
        self._plots = []
        x = self.get_window(self._x_axis)
        for d in self._list_of_data:  # Iterate over all timeseries that is displayed in this object
            y = self.get_window(d) #  get window that should be displayed (a part of timeseries)
            pl, = self._ax[0].plot(x, y)  # display the window
            self._plots.append(pl)  # save plot object. It is necessary to update it

        self._fig.canvas.mpl_connect('key_press_event', self.on_click) # update data on press
        self._fig.canvas.mpl_connect('key_release_event', self.on_release) #repaint on release
        return self

    def on_release(self, event):
        if self._data_updated and event.key in ["right", "left", "up", "down", "z", "x"]:
            # repaint if everything ok
            self.repaint()
            self._data_updated = False

    def on_click(self, event):
        #store current state
        previous_position = self._current_position
        displayed_window_size = self._displayed_window_size
        if event.key == "right":
            self._current_position = self._current_position + self._one_time_jump

        if event.key == "left":
            self._current_position = self._current_position - self._one_time_jump

        if event.key == "up":
            self._current_position = self._current_position + int(self._one_time_jump*0.2)

        if event.key == "down":
            self._current_position = self._current_position - int(self._one_time_jump*0.2)

        if event.key == "z":
            self._displayed_window_size += self._one_time_jump

        if event.key == "x":
            self._displayed_window_size -= self._one_time_jump

        #check border conditions
        if self._current_position > len(self._x_axis) - self._one_time_jump - 1 or self._current_position < 0:
            self._current_position = previous_position
            self._data_updated = False
            return


        self._data_updated = True

    @staticmethod
    def plot(ts: List | np.ndarray, data: List[List | np.array], slices: List[Tuple]):
        fig, axis = plt.subplots(len(slices))

        try:
            _ = iter(axis)
        except TypeError:
            # not iterable
            axis = [axis]
        else:
            # iterable
            pass

        for slc, ax in zip(slices, axis):
            for d in data:
                #print(ax)
                ax.plot(ts[slc[0]:slc[1]], d[slc[0]:slc[1]])

        return fig, axis

    @staticmethod
    def plot2(ts: np.ndarray, data: List[np.array], slices: List[Tuple], pred: List[np.array], unique_labels: np.array):
        fig, axis = Plotter.plot(ts, data, slices)

        for slc, ax in zip(slices, axis):
            color1 = 'tab:blue'
            ax.set_ylabel('Production', color=color1)
            ax.tick_params(axis='y', labelcolor=color1)

            ax2 = ax.twinx()
            color2 = 'tab:green'
            ax2.set_ylabel('Prediction', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_yticks(range(len(unique_labels)))
            ax2.set_yticklabels(unique_labels)

            for p in pred:
                ax2.plot(ts[slc[0]:slc[1]], p[slc[0]:slc[1]], '.', color=color2)

        return fig, axis

    @staticmethod
    def plot_scatter(x:np.ndarray, y:np.ndarray, classes:np.ndarray, centroids:List[List]=None, filter_class:List[int]=[],
                     fig=None, ax=None, x_label="X", y_label="Y", classes_txt=None, legend=True):
        if ax is None or fig is None:
            fig, ax = plt.subplots(1)
        class_list = np.unique(classes) # should be array of ints
        color_list = ["black", "r", "g", "b", "orange", "lime", "aqua", "fuchsia", "peru", "pink"]
        colors = ListedColormap(color_list)

        for i,c in enumerate(class_list):
            if len(filter_class)>0:
                if not c in filter_class:
                    continue
            idx = classes == c

            class_name = f"{c}"
            if classes_txt is not None:
                class_name = classes_txt[c]

            ax.scatter(x[idx], y[idx], c=color_list[c], cmap=colors,
                       marker=",", alpha=0.3, s=1, label=class_name)

        if centroids is not None:
            ax.scatter(centroids[:, 0], centroids[:, 1], c=[color_list[int(c)] for c in class_list], marker="o",
                       edgecolor='black', lw=1)
        if legend:
            lgn = ax.legend(markerscale=4.,)
            for lh in lgn.legendHandles:
                lh.set_alpha(1)

        # ax.set_xlabel("variability_cloudiness_index")
        # ax.set_ylabel("overall_cloudiness_index")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return fig, ax

    @staticmethod
    def plot_overlay(data: List[np.array], fig=None, ax=None):
        if ax is None or fig is None:
            fig, ax = plt.subplots(1)
        x = list(range(data.shape[1]))
        for y in range(data.shape[0]):
            ax.plot(x, data[y], color="b", alpha=0.1)

        return fig, ax

    @staticmethod
    def plot_2D_histograms(histogram: np.array,kde: np.array, fig=None, ax=None):
        count = kde.shape[1] # number of plots to paint
        rows = 6
        cols = count // rows + 1

        if ax is None or fig is None:
            fig, ax = plt.subplots(rows, cols)

        ts = list(range(kde.shape[0]))
        for i in range(kde.shape[1]):
            y = i // cols
            x = i % rows
            axis = ax[y,x]

            axis.fill_between(ts, kde[:,i], color="black", alpha=0.1)
            twin = axis.twinx()
            twin.plot(ts, histogram[:,i], color="b", alpha=1)

        return fig, ax, cols, rows
