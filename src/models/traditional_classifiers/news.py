import json
import os

import numpy as np
import pandas as pd

from src.data_processing.data_loading_utils import _is_id_feat


# ---------------------- AUXILIARY FUNCTIONS FOR NEWS II ---------------------
def score_HR(x):
    if x <= 40:
        return 3

    elif 40 < x <= 50:
        return 1

    elif 50 < x <= 90:
        return 0

    elif 90 < x <= 110:
        return 2

    elif 110 < x <= 130:
        return 2

    elif 130 < x:
        return 3


def score_RR(x):
    if x <= 8:
        return 3

    elif 8 < x <= 11:
        return 1

    elif 11 < x <= 20:
        return 0

    elif 20 < x <= 24:
        return 2

    elif 24 < x:
        return 3


def score_SBP(x):
    if x <= 90:
        return 3

    elif 90 < x <= 100:
        return 2

    elif 100 < x <= 110:
        return 1

    elif 110 < x <= 219:
        return 0

    elif x >= 219:
        return 3


def score_DBP(x):
    return 0


def score_FIO2(x):
    if x <= 22:
        return 0

    else:
        return 2


def score_AVPU(x):
    if x == 1:
        return 0

    else:
        return 3


def score_TEMP(x):
    if x <= 35.0:
        return 3

    elif 35.0 < x <= 36.0:
        return 1

    elif 36.0 < x <= 38.0:
        return 0

    elif 38.0 < x <= 39.0:
        return 1

    elif 39.0 < x:
        return 2


def score_SPO2_1(x):
    if x <= 91:
        return 3

    elif 91 < x <= 93:
        return 2

    elif 93 < x <= 95:
        return 1

    elif 95 < x:
        return 0


def score_SPO2_2(x, oxygen):
    if x <= 83:
        return 3

    elif 83 < x <= 85:
        return 2

    elif 85 < x <= 87:
        return 1

    elif 87 < x <= 92:
        return 0

    elif 92 < x and not oxygen:
        return 0

    elif 92 < x <= 94 and oxygen:
        return 1

    elif 94 < x <= 96 and oxygen:
        return 2

    elif 96 < x and oxygen:
        return 3


FEATURE_SCORING_DIC = {
    "HR": score_HR,
    "RR": score_RR,
    "SBP": score_SBP,
    "DBP": score_DBP,
    "FIO2": score_FIO2,
    "AVPU": score_AVPU,
    "TEMP": score_TEMP,
    "SPO2": score_SPO2_2
}


NEWS_INPUT_PARAMS = []


class NEWS:
    """
    Model Class Wrapper for a NEWS Classifier Model.
    """

    def __init__(self, data_info: dict, **kwargs):
        """
        Initialise object with model configuration.

        Params:
        - data_info: dict, contains information about data configuration, properties and contains data objects.
        - kwargs: model configuration parameters
        """
        # Get feature information
        feats = data_info["data_properties"]["feats"]

        # Identify those that are not id columns
        self.feats = [feat for feat in feats if not _is_id_feat(feat)]
        self.data_name = data_info["data_load_config"]["data_name"]

        # Get proper model_config
        self.model_config = {key: value for key, value in kwargs.items() if key in NEWS_INPUT_PARAMS}

        # Initialise other useful information
        self.run_num = 1
        self.model_name = "NEWS"

        # Useful for consistency
        self.training_params = {}

    def train(self, data_info, **kwargs):
        """
        Wrapper method for fitting the model to input data.

        Params:
        - probability: bool value, indicating whether model should output hard outcome assignments, or probabilistic.
        - data_info: dictionary with data information, objects and parameters.
        """

        # Get data_name
        data_name = data_info["data_load_config"]["data_name"]

        # Update run_num to make space for new experiment
        run_num = self.run_num
        save_fd = f"experiments/{data_name}/{self.model_name}/"

        while os.path.exists(save_fd + f"run{run_num}/"):
            run_num += 1

        # make new folder and update run num
        os.makedirs(save_fd + f"run{run_num}/")
        self.run_num = run_num

        return None

    def predict(self, X_test):
        """
        Make predictions on X_test array.

        Params:
        - X_test: numpy array of shape (N, D_f)

        Returns:
            - NEWS score for each patient, of shape (N, ).
        """

        # Initialize output array
        output_scores = np.zeros(X_test.shape[0])

        # Compute score per each feature
        num_feats_used = 0
        for feat_id, feat in enumerate(self.feats):

            # Which feature does it correspond to?
            if feat in FEATURE_SCORING_DIC.keys():

                # Load score for feature
                scoring_fn = FEATURE_SCORING_DIC[feat]

                if "mimic" in self.data_name.lower() and feat == "SPO2":
                    scoring_fn = score_SPO2_1  # MIMIC DOES NOT HAVE OXYGEN INFORMATION.

                # Save function to run on numpy array
                vec_scoring_fn = np.vectorize(scoring_fn)

                # Apply score to data
                output_scores += vec_scoring_fn(X_test[:, feat_id])

                # add 1 to the number of features
                num_feats_used += 1

        return output_scores

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
        data_load_config = data_info["data_load_config"]
        data_name = data_load_config["data_name"]

        # Obtain the ids for patients in test set
        id_info = data_info["ids"][-1]
        pat_ids = id_info[:, 0, 0]

        # Define save_fd, track_fd
        save_fd = f"results/{data_name}/{self.model_name}/run{self.run_num}/"

        if not os.path.exists(save_fd):
            os.makedirs(save_fd)

        # Make prediction on test data
        output_test = self.predict(X_test[:, -1, :])  # Apply NewsI on last observation sign.

        # Save model scores
        news_scores = pd.Series(output_test, index=pat_ids)
        news_scores.to_csv(save_fd + "news_scores.csv", index=True)

        # save model parameters
        with open(save_fd + "data_config.json", "w+") as f:
            json.dump(data_load_config, f, indent=4)

        # Return objects
        outputs_dic = {"save_fd": save_fd, "model_config": self.model_config,
                       "scores": news_scores, "y_true": y_test}

        # Print Data
        print(f"\n\n Results saved under {save_fd}")

        return outputs_dic
