from sklearn.base import BaseEstimator, TransformerMixin
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.utils.validation import check_is_fitted

class MyPipline(BaseEstimator,TransformerMixin):
    def __init__(self, methods):
        self.methods = methods

    def set_methods(self, methods):
        self.methods = methods

    @property
    def shape(self):
        return self.shape_

    def fit(self, X, y=None, params={}):
        self.shape_ = X.shape
        return self

    def transform(self, X, y=None, params={}):
        self.steps_ = [("input", X.copy())]

        for m in self.methods:
            # make provided numbers of steps

            iterations = 1
            if len(m) > 2 and m[2] and m[2] > 1:
                iterations = m[2]

            for r in range(0, iterations):
                m[1].fit(X)
                X = m[1].transform(X)
                steps = None
                if type(X) is tuple:
                    steps = X[1]
                    X = X[0]

                if steps is None:
                    self.steps_.append((m[0] + (f"({r})" if iterations > 1 else ""), X.copy()))
                else:
                    for i in range(1, len(steps)): # first step is input (copy of provided argument). avoid adding repeating heatmaps to steps
                        self.steps_.append((m[0] + (f"({r})" if iterations > 1 else "") +"."+ steps[i][0], steps[i][1].copy()))

        return X

    def statistics(self):
        return "Layers:" + "\n".join([step[0] for step in self.steps_])

    def get_step(self, title):
        for step in self.steps_:
            if step[0] == title:
                return step[1]
        raise ValueError("No layer named: ", title)

    def plot(self, rows = 3, ax=None, fig=None):
        l = len(self.steps_)
        count = sum(name[0] != "_" for name, _ in self.steps_)

        axis = ceil(count / rows ), rows

        if ax is None or fig is None:
            fig, ax = plt.subplots(axis[1], axis[0])

        _ax = [ax[i, j] for j in range(axis[0]) for i in range(axis[1])]

        i = 0
        for (name, result) in self.steps_:
            if name[0]!="_":
                _ax[i].set(title=f"{i+1}: {name}")
                sns.heatmap(result.reshape(self.shape_), cmap='viridis', ax=_ax[i])
                _ax[i].invert_yaxis()
                i += 1
        return fig, ax

    def get_params(self, deep=False):
        params = {}
        for step_name, step in self.methods:
            p = step.get_params(deep=deep)
            for key in p:
                params[f"{step_name}__{key}"] = p[key]

        return params