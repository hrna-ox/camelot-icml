"""
Define Model Class for SVMAll model. SVM is fit to concatenated trajectories.
"""

import os, json

from sklearn.svm import SVC
import numpy as np
import pandas as pd


SVMALL_INPUT_PARAMS = ["C", "kernel", "degree", "gamma", "coef0", "shrinking", "probability", "tol", "class_weight",
                       "random_state", "verbose"]


class SVMAll(SVC):
    """
    Model Class Wrapper for an SVM model training on all (time, feature) pair values.
    """

    def __init__(self, data_info: dict = {}, probability: bool = True, **kwargs):
        """
        Initialise object with model configuration.

        Params:
        - data_info: dict, dictionary containing dataset information, including objects and properties.
        - kwargs: model configuration parameters
        """

        # Get proper model_config
        self.model_config = {key: value for key, value in kwargs.items() if key in SVMALL_INPUT_PARAMS}

        if "probability" not in self.model_config.keys():
            self.model_config["probability"] = probability

        # Initialise other useful information
        self.run_num = 1
        self.model_name = "SVMALL"

        # Useful for consistency
        self.training_params = {}

        # Initialise SVM object with this particular model config
        super().__init__(**self.model_config, verbose=True)

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

        # Get shape and flatten array
        N_test, T, D_f = X_train.shape
        X = X_train.reshape(N_test, -1)
        y = np.argmax(y_train, axis=1)

        # Fit model
        self.fit(X, y, sample_weight=None)

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
        track_fd = f"experiments/{data_name}/{self.model_name}/run{self.run_num}/"

        if not os.path.exists(save_fd):
            os.makedirs(save_fd)

        if not os.path.exists(track_fd):
            os.makedirs(track_fd)

        # Make prediction on test data
        if self.model_config["probability"] is True:
            X_test = X_test.reshape(X_test.shape[0], -1)
            output_test = self.predict_proba(X_test)

        else:
            # Predict gives categorical vector, and we one-hot encode output.
            output_test = np.eye(y_test.shape[-1])[self.predict(X_test)]

        # First, compute predicted y estimates
        y_pred = pd.DataFrame(output_test, index=pat_ids, columns=outc_dims)
        outc_pred = pd.Series(np.argmax(output_test, axis=-1), index=pat_ids)
        y_true = pd.DataFrame(y_test, index=pat_ids, columns=outc_dims)

        # ----------------------------- Save Output Data --------------------------------
        # Useful objects
        y_pred.to_csv(save_fd + "y_pred.csv", index=True, header=True)
        outc_pred.to_csv(save_fd + "outc_pred.csv", index=True, header=True)
        y_true.to_csv(save_fd + "y_true.csv", index=True, header=True)

        # save model parameters
        save_params = {**data_info["data_load_config"], **self.model_config, **self.training_params}
        with open(save_fd + "config.json", "w+") as f:
            json.dump(save_params, f, indent=4)

        with open(track_fd + "config.json", "w+") as f:
            json.dump(save_params, f, indent=4)

        with open(save_fd + "model_config.json", "w+") as f:
            json.dump(self.model_config, f, indent=4)

        with open(track_fd + "model_config.json", "w+") as f:
            json.dump(self.model_config, f, indent=4)

        # Return objects
        outputs_dic = {"save_fd": save_fd, "model_config": self.model_config,
                       "y_pred": y_pred, "class_pred": outc_pred, "y_true": y_true
                       }

        # Print Data
        print(f"\n\n Experiments saved under {track_fd} and {save_fd}")

        return outputs_dic
