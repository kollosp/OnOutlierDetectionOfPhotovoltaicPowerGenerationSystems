from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
import SEAIPPF.image as image
from SEAIPPF.MyPipeline import MyPipline
import numpy as np
from SEAIPPF.RegressorTransformer import RegressorTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MaxAbsScaler, PolynomialFeatures

class Transformer1(BaseEstimator,TransformerMixin):
    def __init__(self,
                 regressor_degrees=11,
                 hit_points_neighbourhood=3,
                 conv2Dy_shape_factor = 0.1,
                 conv2Dx_shape_factor = 0.5):
        self.regressor_degrees = regressor_degrees
        self.hit_points_neighbourhood = hit_points_neighbourhood
        self.conv2Dx_shape_factor = conv2Dx_shape_factor
        self.conv2Dy_shape_factor = conv2Dy_shape_factor

    def fit(self, X, y=None):
        return self

    def get_params_info(self):
        params = self.get_params()
        # Default value, min bound, max bound, integrity
        params["conv2Dx_shape_factor"] = (0.1, 0, 1, False)
        params["conv2Dy_shape_factor"] = (0.5, 0, 1, False)
        params["hit_points_neighbourhood"] = (3, 1, 12, True)
        params["regressor_degrees"] = (11, 4, 15, True)
        return params

    def transform(self, X, y=None):
        shape = X.shape[0]
        pipe = MyPipline([
                ('conv2Dy', image.Convolution(kernel=np.ones((1, int(self.conv2Dy_shape_factor * shape) + 1)))),
                ('conv2DX', image.Convolution(kernel=np.ones((int(self.conv2Dx_shape_factor * shape) + 1, 1)))),
                ('hit_points', image.HitPoints(max_iter=1, neighbourhood=self.hit_points_neighbourhood)),
                ('regressor', RegressorTransformer(regressor=make_pipeline(
                    PolynomialFeatures(int(self.regressor_degrees)),
                    LinearRegression()
                ))),
            ])

        mask = pipe.fit_transform(X, params={})

        return mask

    def __str__(self):
        return f"Transformer1({self.get_params()})"