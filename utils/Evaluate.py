
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.optimize import differential_evolution
from datetime import date

import logging

class Evaluate():
    def __init__(self, data, ts, model, file_log="file_log.txt"):
        self.data = data
        self.ts = ts
        self.model = model
        today = date.today()
        txt_datetime =today.strftime("%Y%m%d-%H%M%S")
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(f'{txt_datetime}_{file_log}'),
                logging.StreamHandler()
            ]
        )

        self.best_result = np.inf
        self.best_result_configuration = None
        self.best_result_iteration = 0
        self.iteration_counter = 0

        logging.info(f"======================= Start =======================")

    def params_info_to_scipy_genetic_config(self, params_info, include=[]):
        param_names = []
        bounds = []
        defaults = []
        integrality = []
        for key in params_info:
            if len(include) == 0 or key in include:
                b_default, b_min, b_max, p_integrality = params_info[key]
                defaults.append(b_default)
                param_names.append(key)
                bounds.append((b_min, b_max))
                integrality.append(p_integrality)

        return param_names, defaults, bounds, integrality

    def get_params(self):
        return  self.model.get_params()

    def generate_set_params(self, params=[], param_names=None):
        # set model parameters
        if param_names is not None and len(params) > 0:
            if len(param_names) != len(params):
                raise ValueError(f"If param_names provided then it must have same size as params. params={params},"
                                 f" param_names={param_names}")

            set_params = {name: value for name, value in zip(param_names, params)}
        else:
            set_params = {key: value for key, value in zip(self.model.get_params(), params)}
        return set_params


    def __call__(self, params=[], param_names=None):
        # set model parameters
        set_params = self.generate_set_params(params, param_names)
        self.model.set_params(**set_params)

        # evaluate on the dataset
        #print("Data len:", len(self.data) / 288)
        iterations = 5
        train_test_split = 288 * 100
        test_len = 288 * 40
        metrics = []
        for i in range(iterations):
            y_train, y_test = self.data["Production"][i * train_test_split:(i + 1) * train_test_split], \
                              self.data["Production"][(i + 1) * train_test_split:(i + 1) * train_test_split + test_len]

            models = [self.model]
            preds = []
            fh = [i - train_test_split for i in range(train_test_split, train_test_split + test_len)]
            for model in models:
                model.fit(y=y_train)
                # model.plot(plots=["model"])
                pred = model.predict(fh=fh)
                preds.append(pred)

            metrics.append([r2_score(y_test.values, pred.values) for pred in preds])

        metrics = np.array(metrics)

        # df = pd.DataFrame(data=metrics, columns=["reference R2", "R2"])
        summary = pd.DataFrame(data=[[m, s] for m, s in zip(np.mean(metrics, axis=0), np.std(metrics, axis=0))],
                               columns=["Mean", "Std"],
                               index=["Reference R2", "R2"])

        # print(df)
        # print(summary)
        ret = -1 * (summary["Mean"][-1] - 1)
        logging.info(f"It {self.iteration_counter}: Set params: {set_params}.\n Return {ret}, R2: {summary['Mean'][-1]}.\n"
                    f" Current best result in It {self.best_result_iteration}: {self.best_result}")

        if ret < self.best_result:
            self.best_result = ret
            self.best_result_iteration = self.iteration_counter
            self.best_result_configuration = set_params
            logging.info(f"It {self.iteration_counter}: New best found: {set_params}. Return {ret}, R2: {summary['Mean'][-1]}")

        self.iteration_counter += 1

        return ret

    @staticmethod
    def callback(intermediate_result):
        print("Differential evolution callback:",intermediate_result.x, intermediate_result.fun)

    @staticmethod
    def scipy_differential_evolution(data, ts, model, include_parameters):
        print("Available parameters", model.get_params_info())

        evaluate = Evaluate(data, ts, model)
        param_names, defaults, bounds, integrality = \
            evaluate.params_info_to_scipy_genetic_config(model.get_params_info(), include=include_parameters)

        result = differential_evolution(evaluate, bounds=bounds, integrality=integrality, args=[param_names])

        return result, evaluate