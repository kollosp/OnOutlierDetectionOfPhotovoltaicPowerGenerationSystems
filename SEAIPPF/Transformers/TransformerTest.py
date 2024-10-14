from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
import SEAIPPF.image as image
from SEAIPPF.MyPipeline import MyPipline
import numpy as np
from SEAIPPF.RegressorTransformer import RegressorTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MaxAbsScaler, PolynomialFeatures

class TransformerTest(BaseEstimator,TransformerMixin):
    def __init__(self,
                 regressor_degrees=11,
                 hit_points_neighbourhood=3,
                 conv2Dy_shape_factor = 0.1,
                 conv2Dx_shape_factor = 0.05,
                 hit_points_max_iter = 1,
                 rbf_epsilon=1/2,
                 sklearn_regressor = LinearRegression()):
        self.rbf_epsilon = rbf_epsilon
        self.sklearn_regressor = sklearn_regressor
        self.regressor_degrees = regressor_degrees
        self.hit_points_neighbourhood = hit_points_neighbourhood
        self.conv2Dx_shape_factor = conv2Dx_shape_factor
        self.conv2Dy_shape_factor = conv2Dy_shape_factor
        self.hit_points_max_iter = hit_points_max_iter

    def fit(self, X, y=None):
        return self

    def get_params_info(self):
        params = self.get_params()
        # Default value, min bound, max bound, integrity
        params["conv2Dx_shape_factor"] = (0.1, 0, 1, False)
        params["conv2Dy_shape_factor"] = (0.5, 0, 1, False)
        params["hit_points_neighbourhood"] = (3, 1, 12, True)
        params["regressor_degrees"] = (11, 4, 15, True)
        params["max_iter"] = (1, 1, 4, True)
        return params

    def transform(self, X, y=None):
        shape = X.shape[0]
        self.pipe_ = MyPipline([
                # ('conv2Dy', image.Convolution(kernel=np.ones((1, int(self.conv2Dy_shape_factor * shape) + 1)))),
            # ('Threshold', image.Threshold()),
            ('KernelProcess', image.KernelProcess(epsilon=self.rbf_epsilon, max_roots=100)),
            ('Threshold', image.Threshold()),
            ('Erosion', image.Erosion()),
            ('Max', image.HitPoints(max_iter=self.hit_points_max_iter,
                                           neighbourhood=self.hit_points_neighbourhood,
                                           preserve_original_values=True)),

            ('KernelProcess', image.KernelProcess(epsilon=self.rbf_epsilon, max_roots=100)),
            # ('KernelProcess', image.KernelProcess(epsilon=self.rbf_epsilon)),

            ('Max 2', image.HitPoints(max_iter=1, neighbourhood=self.hit_points_neighbourhood,
                                             preserve_original_values=False)),
            ('TakeLast', image.TakeLast()),
            ('Regressor', RegressorTransformer(regressor=make_pipeline(
                PolynomialFeatures(int(self.regressor_degrees)),
                self.sklearn_regressor
            )))])

        mask = self.pipe_.fit_transform(X, params={})

        return mask

    def plot(self, ax=None, fig=None):
        return self.pipe_.plot(ax=ax, fig=fig)


    def __str__(self):
        return f"TransformerTest({self.get_params()})"