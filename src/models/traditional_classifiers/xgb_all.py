"""
Define Model Class for XGB all model. Array is flattened to a single dimension.
"""

import os, json

import numpy as np
import pandas as pd

from xgboost import XGBClassifier as XGBClassifier

XGBOOST_INPUT_PARAMS = ["n_estimators", "depth", "objective", "gamma", "learning_rate", "use_label_encoder"]


class XGBAll:
    """
    Model Class Wrapper for a XGBoost model.
    """

    def __init__(self, data_info: dict = {}, **kwargs):
        """
        Initialise object with model configuration.

        Params:
        - data_info: dict, contains information about data configuration, properties and contains data objects.
        - kwargs: model configuration parameters
        """

        # Get proper model_config
        model_config = {key: value for key, value in kwargs.items() if key in XGBOOST_INPUT_PARAMS}

        if "seed" in kwargs.keys():
            model_config["random_state"] = kwargs["seed"]

        # Initialise model
        self.model = XGBClassifier(verbosity=1, **model_config, use_label_encoder=False)

        # Initialise other useful information
        self.run_num = 1
        self.model_name = "XGBALL"

        # Useful for consistency
        self.model_config = model_config
        self.training_params = {}

    def train(self, data_info, **kwargs):
        """
        Wrapper method for fitting the model to input data.

        Params:
        - probability: bool value, indicating whether model should output hard outcome assignments, or probabilistic.
        - data_info: dictionary with data information, objects and parameters.
        """

        # Unpack relevant data information
        X_train, X_val, X_test = data_info["X"]
        y_train, y_val, y_test = data_info["y"]
        data_name = data_info["data_load_config"]["data_name"]

        # Update run_num to make space for new experiment
        run_num = self.run_num
        save_fd = f"experiments/{data_name}/{self.model_name}/"

        while os.path.exists(save_fd + f"run{run_num}/"):
            run_num += 1

        # make new folder and update run num
        os.makedirs(save_fd + f"run{run_num}/")
        self.run_num = run_num

        # Fit to concatenated X_train, X_val
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)

        # Flatten time and D_f dimensions
        X = X_train.reshape(X_train.shape[0], -1)
        y = np.argmax(y_train, axis=1)

        self.model.fit(X, y)

        return None

    def analyse(self, data_info):
        """
        Evaluation method to compute and save output results.

        Params:
        - data_info: dictionary with data information, objects and parameters.

        Returns:
            - y_pred: dataframe of shape (N, output_dim) with outcome probability prediction.
            - outc_pred: Series of shape (N, ) with predicted outcome based on most likely outcome prediction.
            - y_true: dataframe of shape (N, output_dim) ith one-hot encoded true outcome.

        Saves a variety of model information, as well.
        """

        # Unpack test data
        _, _, X_test = data_info["X"]
        _, _, y_test = data_info["y"]

        # Get basic data information
        data_properties = data_info["data_properties"]
        outc_dims = data_properties["outc_names"]
        data_load_config = data_info["data_load_config"]
        data_name = data_load_config["data_name"]

        # Obtain the ids for patients in test set
        id_info = data_info["ids"][-1]
        pat_ids = id_info[:, 0, 0]

        # Define save_fd, track_fd
        save_fd = f"results/{data_name}/{self.model_name}/run{self.run_num}/"

        if not os.path.exists(save_fd):
            os.makedirs(save_fd)

        # Flatten and make probability
        X_test = X_test.reshape(pat_ids.size, -1)
        output_test = self.model.predict_proba(X_test)

        # First, compute predicted y estimates
        y_pred = pd.DataFrame(output_test, index=pat_ids, columns=outc_dims)
        outc_pred = pd.Series(np.argmax(output_test, axis=-1), index=pat_ids)
        y_true = pd.DataFrame(y_test, index=pat_ids, columns=outc_dims)

        # Define clusters as outcome predicted groups
        pis_pred = y_pred
        clus_pred = outc_pred

        # ----------------------------- Save Output Data --------------------------------
        # Useful objects
        y_pred.to_csv(save_fd + "y_pred.csv", index=True, header=True)
        outc_pred.to_csv(save_fd + "outc_pred.csv", index=True, header=True)
        clus_pred.to_csv(save_fd + "clus_pred.csv", index=True, header=True)
        pis_pred.to_csv(save_fd + "pis_pred.csv", index=True, header=True)
        y_true.to_csv(save_fd + "y_true.csv", index=True, header=True)

        # save model parameters
        with open(save_fd + "data_config.json", "w+") as f:
            json.dump(data_info["data_load_config"], f, indent=4)

        with open(save_fd + "model_config.json", "w+") as f:
            json.dump(self.model_config, f, indent=4)

        # Return objects
        outputs_dic = {"save_fd": save_fd, "model_config": self.model_config,
                       "y_pred": y_pred, "class_pred": outc_pred, "clus_pred": clus_pred, "pis_pred": pis_pred,
                       "y_true": y_true
                       }

        # Print Data
        print(f"\n\n Results saved under {save_fd}")

        return outputs_dic
