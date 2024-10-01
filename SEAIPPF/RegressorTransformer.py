from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
class RegressorTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        shape = X.shape
        c = np.concatenate([[[j], [i], [w]]
                            for i, x in enumerate(X)
                            for j, w in enumerate(x)], axis=1)

        # c = np.concatenate((X,y,w), axis=1)
        # print({"X":X, "y":c[:,1], "sample_weight":w})

        # add more observations to replace sample_weight
        # f = X.flatten()
        # print(np.sum(f > 0))

        # print_arr = np.concatenate([
        #    np.array([c[0, i], c[1, i]]).reshape(-1,2) for i in range(c.shape[1]) if c[2, i] > 0
        # ])
        # print(print_arr)

        X = np.concatenate([
           # np.array([c[0, i], c[1, i]] * (c[2, i] * 10).astype(int)).reshape(-1,2) for i in range(c.shape[1]) if c[2, i] > 0
           np.array([c[0, i], c[1, i]] ).reshape(-1,2) for i in range(c.shape[1]) if c[2, i] > 0
        ])
        # print(X.shape)


        # cls = SVR(C=self.C, epsilon=self.epsilon, degree=self.degree)
        self.regressor.fit(X=X[:, 0:1], y=X[:, 1])

        pred = self.regressor.predict(X=np.array(list(range(0, shape[1]))).reshape(-1, 1))
        pred_arr = np.zeros(shape)
        pred_i = pred.astype(int)
        pred_i[pred_i >= pred_arr.shape[1]] = pred_arr.shape[1] -1
        # print(pred)
        for i in range(0, pred_arr.shape[1]):
            # print(pred_arr.shape[1], i, pred_i[i])
            if 0 <= pred_i[i] < pred_arr.shape[0] and 0 <= i < pred_arr.shape[1]:
                pred_arr[pred_i[i], i] = 1

        return pred_arr