#!/usr/bin/env python3
"""
Utility Functions for Loading Data and Saving/Evaluating Results.
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_processing.MIMIC.data_utils import convert_to_timedelta

# ---------------------------------------------------------------------------------------
"Global variables for specific dataset information loading."

HAVEN_PARSE_TIME_VARS = ['charttime', 'hadm_end_time', 'hadm_start_time', 'event_time']
HAVEN_VITALS = ['HR', 'RR', 'SBP', 'DBP', 'SPO2', 'FIO2', 'TEMP', 'AVPU']
HAVEN_SERUM = ['HGB', 'WBC', 'EOS', 'BAS', 'NEU', 'LYM']
HAVEN_BIOCHEM = ['ALB', 'CR', 'CRP', 'POT', 'SOD', 'UR']
HAVEN_STATIC = ['age', 'gender', 'is_elec', 'is_surg']
HAVEN_OUTCOME_NAMES = ['Healthy', 'Death', 'ICU', 'Card']

MIMIC_PARSE_TIME_VARS = ["intime", "outtime", "chartmax"]
MIMIC_PARSE_TD_VARS = ["sampled_time_to_end(1H)", "time_to_end", "time_to_end_min", "time_to_end_max"]
MIMIC_VITALS = ["TEMP", "HR", "RR", "SPO2", "SBP", "DBP"]
MIMIC_STATIC = ["age", "gender", "ESI"]
MIMIC_OUTCOME_NAMES = ["De" ,"I", "W", "Di"]

MAIN_ID_LIST = ["subject_id", "hadm_id", "stay_id", "patient_id", "pat_id"]  # Identifiers for main ids.

# ----------------------------------------------------------------------------------------
"Useful functions to define"


def _is_id_feat(feat):
    """
    Boolean indicator to identify if feature is a key identifier in data or not.

    Params:
    - feat: str, name of feature.

    Returns:
        - bool, indicates if feature is used as identifier in processing.
    """
    is_id_1 = feat in HAVEN_PARSE_TIME_VARS + MIMIC_PARSE_TIME_VARS + MIMIC_PARSE_TD_VARS + MAIN_ID_LIST
    is_id_2 = "time_to_end" in feat

    return is_id_1 or is_id_2


def _is_static_feat(feat):
    """
    Boolean indicator to identify if feature is a static variable in data or not.

    Params:
    - feat: str, name of feature.

    Returns:
        - bool, indicates if feature is a static biomedical variable.
    """
    is_static = feat in HAVEN_STATIC + MIMIC_STATIC

    return is_static


def _is_temporal_feat(feat):
    """
    Boolean indicator to identify if feature is temporal or not.

    Params:
    - feat: str, name of feature.

    Returns:
        - bool, indicates if feat is a biomedical variable which varies over time.
    """
    is_time_feat = feat in HAVEN_VITALS + HAVEN_SERUM + HAVEN_BIOCHEM + MIMIC_VITALS

    return is_time_feat


def convert_datetime_to_hour(series):
    """Convert pandas Series of datetime values to float Series with corresponding hour values"""
    seconds_per_hour = 3600

    return series.dt.total_seconds() / seconds_per_hour


def _get_features(key, data_name="HAVEN"):
    """
    Compute list of features to keep given key. Key can be one of:
    - str, where the corresponding features are selected according to the fn below.
    - list, where the corresponding features are the original list.
    """
    if isinstance(key, list):
        return key

    elif isinstance(key, str):
        if data_name == "HAVEN":
            vitals = HAVEN_VITALS
            vars1, vars2 = HAVEN_SERUM, HAVEN_BIOCHEM
            static = HAVEN_STATIC

        elif data_name == "MIMIC":
            vitals = MIMIC_VITALS
            static = MIMIC_STATIC
            vars1, vars2 = None, None

        elif data_name == "SAMPLE":
            vitals, vars1, vars2, static = None, None, None, None

        else:
            raise ValueError(f"Data Name does not match available datasets. Input provided {data_name}")

        # Add features given substrings of key. We initialise set in case of repetition (e.g. 'vars1-lab')
        features = set([])
        if "vit" in key.lower():
            features.update(vitals)

        if "vars1" in key.lower():
            features.update(vars1)

        if "vars2" in key.lower():
            features.update(vars2)

        if "lab" in key.lower():
            features.update(vars1)
            features.update(vars2)

        if "sta" in key.lower():
            features.update(static)

        if "all" in key.lower():
            features = _get_features("vit-lab-sta", data_name)

        sorted_features = sorted(features)  # sorted returns a list of features.
        print(f"\n{data_name} data has been subsettted to the following features: \n {sorted_features}.")

        return sorted_features

    else:
        raise TypeError(f"Argument key must be one of type str or list, type {type(key)} was given.")


def _numpy_forward_fill(array):
    """Forward Fill a numpy array. Time index is axis = 1."""
    array_mask = np.isnan(array)
    array_out = np.copy(array)

    # Add time indices where not masked, and propagate forward
    inter_array = np.where(~ array_mask, np.arange(array_mask.shape[1]).reshape(1, -1, 1), 0)
    np.maximum.accumulate(inter_array, axis=1,
                          out=inter_array)  # For each (n, t, d) missing value, get the previously accessible mask value

    # Index matching for output. For n, d sample as previously, use inter_array for previous time id
    array_out = array_out[np.arange(array_out.shape[0])[:, None, None],
                          inter_array,
                          np.arange(array_out.shape[-1])[None, None, :]]

    return array_out


def _numpy_backward_fill(array):
    """Backward Fill a numpy array. Time index is axis = 1"""
    array_mask = np.isnan(array)
    array_out = np.copy(array)

    # Add time indices where not masked, and propagate backward
    inter_array = np.where(~ array_mask, np.arange(array_mask.shape[1]).reshape(1, -1, 1), array_mask.shape[1] - 1)
    inter_array = np.minimum.accumulate(inter_array[:, ::-1], axis=1)[:, ::-1]
    array_out = array_out[np.arange(array_out.shape[0])[:, None, None],
                          inter_array,
                          np.arange(array_out.shape[-1])[None, None, :]]

    return array_out


def _median_fill(array):
    """Median fill a numpy array. Time index is axis = 1"""
    array_mask = np.isnan(array)
    array_out = np.copy(array)

    # Compute median and impute
    array_med = np.nanmedian(np.nanmedian(array, axis=0, keepdims=True), axis=1, keepdims=True)
    array_out = np.where(array_mask, array_med, array_out)

    return array_out


def _load(data_name, window=4):
    """Load Trajectory, Target data jointly given data folder."""

    # Make data folder
    data_fd = f"data/{data_name}/processed/"
    try:
        os.path.exists(data_fd)
    except AssertionError:
        print(data_fd)

    if "HAVEN" in data_name:

        # Load Data
        X = pd.read_csv(data_fd + "COPD_VLS_process.csv", parse_dates=HAVEN_PARSE_TIME_VARS, header=0)
        y = pd.read_csv(data_fd + "copd_outcomes.csv", index_col=0)

    elif "MIMIC" in data_name:

        # Load Data
        X = pd.read_csv(data_fd + "vitals_process.csv", parse_dates=MIMIC_PARSE_TIME_VARS, header=0, index_col=0)
        y = pd.read_csv(data_fd + f"outcomes_{window}h_process.csv", index_col=0)

        # Convert columns to timedelta
        X = convert_to_timedelta(X, *MIMIC_PARSE_TD_VARS)

    elif "SAMPLE" in data_name:

        # Load data
        X = None
        y = None

    else:
        raise ValueError(f"Data Name does not match available datasets. Input Folder provided {data_fd}")

    return X, y


def impute(X):
    """
    Imputation of 3D array accordingly with time as dimension 1:
     1st - forward value propagation,
     2nd - backwards value propagation,
     3rd - median value imputation.

    Mask returned at the end, corresponding to original missing values.
    """
    impute_step1 = _numpy_forward_fill(X)
    impute_step2 = _numpy_backward_fill(impute_step1)
    impute_step3 = _median_fill(impute_step2)

    # Compute mask
    mask = np.isnan(X)

    return impute_step3, mask


def get_ids(data_name):
    """
    Get input id information.

    Params:
    - data_folder: str, folder of dataset, or name of dataset.

    Returns:
        - Tuple of id col, time col and whether time to end needs computation.
    """

    if "HAVEN" in data_name:
        id_col, time_col, needs_time_to_end = "subject_id", "charttime", True

    elif "MIMIC" in data_name:
        id_col, time_col, needs_time_to_end = "stay_id", "sampled_time_to_end(1H)", False

    elif "SAMPLE" in data_name:
        id_col, time_col, needs_time_to_end = None, None, None

    else:
        raise ValueError(f"Data Name does not match available datasets. Input Folder provided {data_name}")

    return id_col, time_col, needs_time_to_end


def _get_outcome_names(data_name):
    """Return the corresponding outcome columns given dataset name."""
    if data_name == "HAVEN":
        return HAVEN_OUTCOME_NAMES

    elif data_name == "MIMIC":
        return MIMIC_OUTCOME_NAMES

    elif data_name == "SAMPLE":
        return None


def _check_input_format(X, y):
    """Check conditions to confirm model input."""

    try:
        # Length and shape conditions
        cond1 = X.shape[0] == y.shape[0]
        cond2 = len(X.shape) == 3
        cond3 = len(y.shape) == 2

        # Check non-missing values
        cond4 = np.sum(np.isnan(X)) + np.sum(np.isnan(y)) == 0

        # Check y output is one hot encoded
        cond5 = np.all(np.sum(y, axis=1) == 1)

        assert cond1
        assert cond2
        assert cond3
        assert cond4
        assert cond5

    except Exception as e:
        print(e)
        raise AssertionError("One of the check conditions has failed.")


# ---------------------------------------------------------------------------------------
"Main Class for Data Processing."


def _subset_to_balanced(X, y, mask, ids):
    """Subset samples so dataset is more well sampled."""
    class_numbers = np.sum(y, axis=0)
    largest_class, target_num_samples = np.argmax(class_numbers), np.sort(class_numbers)[-2]
    print("\nSubsetting class {} from {} to {} samples.".format(largest_class, class_numbers[largest_class],
                                                              target_num_samples))

    # Select random
    largest_class_ids = np.arange(y.shape[0])[y[:, largest_class] == 1]
    class_ids_samples = np.random.choice(largest_class_ids, size=target_num_samples, replace=False)
    ids_to_remove_ = np.setdiff1d(largest_class_ids, class_ids_samples)

    # Remove relevant ids
    X_out = np.delete(X, ids_to_remove_, axis=0)
    y_out = np.delete(y, ids_to_remove_, axis=0)
    mask_out = np.delete(mask, ids_to_remove_, axis=0)
    ids_out = np.delete(ids, ids_to_remove_, axis=0)

    return X_out, y_out, mask_out, ids_out


class DataProcessor:
    """
    Data Processor Class for Data Loading and conversion to input. By default, time columns should be in dt.datetime
    format and are converted to hours.

    Params:
        - data_name: str, which dataset to load
        - outcome_window: str, how many hours to outcome from outtime admission.
        - feat_set: str or list. If list, list of features to consider for input. If str, convert to list according
        to corresponding name.
        - time_range: tuple of floats, for each admission, consider only admissions between these endpoints.
        - include_time: bool, indicates whether to add time difference between consecutive observations as a variable.
    """

    def __init__(self, data_name="HAVEN", target_window=4, feat_set='vitals',
                 time_range=(24, 72)):

        # Dataset selection and feature subset
        self.dataset_name = data_name
        self.target_window = target_window
        self.feat_set = feat_set

        # Subset observations between time range
        self.time_range = time_range

        # Ids depending on data
        self.id_col = None
        self.time_col = None
        self.needs_time_to_end_computation = False

        # Mean, Divisor for normalisation. Initialise to None.
        self.min = None
        self.max = None

    def load_transform(self):
        """Load Dataset and Transform to input format."""
        data = self.load()
        output = self.transform(data)

        return output

    def load(self):
        """Load Dataset according to given parameters."""

        # Load data
        data = _load(self.dataset_name, window=self.target_window)

        # Get data info
        self.id_col, self.time_col, self.needs_time_to_end_computation = get_ids(self.dataset_name)

        return data

    def transform(self, data):
        """Transform data according to instance attributes."""

        # Expand data
        x, y = data

        # Add Time to End and Truncate if needed
        x_inter = self._add_time_to_end(x)
        x_inter = self._truncate(x_inter)
        self._check_correct_time_conversion(x_inter)

        # subset to relevant features (keeps self.id_col and self.time_col still)
        x_subset, features = self.subset_to_features(x_inter)

        # Convert to 3D array
        x_inter, pat_time_ids = self.convert_to_3darray(x_subset)

        # Normalise array
        x_inter = self.normalise(x_inter)

        # Impute missing values
        x_out, mask = impute(x_inter)

        # Do things to y
        outcomes = _get_outcome_names(self.dataset_name)
        y_data = y[outcomes]
        y_out = y_data.to_numpy().astype("float32")

        # Check data loaded correctly
        _check_input_format(x_out, y_out)

        return x_out, y_out, mask, pat_time_ids, features, outcomes, x_subset, y_data

    def _add_time_to_end(self, X):
        """Add new column to dataframe - this computes time to end of grouped observations, if needed."""
        x_inter = X.copy(deep=True)

        # if time to end has not been computed
        if self.needs_time_to_end_computation is True:

            # Compute datetime values for time until end of group of observations
            times = X.groupby(self.id_col).apply(lambda x: x.loc[:, self.time_col].max() - x.loc[:, self.time_col])

            # add column to dataframe after converting to hourly times.
            x_inter["time_to_end"] = convert_datetime_to_hour(times).values

        else:
            x_inter["time_to_end"] = x_inter[self.time_col].values
            x_inter["time_to_end"] = convert_datetime_to_hour(x_inter.loc[:, "time_to_end"])

        # Sort data
        self.time_col = "time_to_end"
        x_out = x_inter.sort_values(by=[self.id_col, "time_to_end"], ascending=[True, False])

        return x_out

    def _truncate(self, X):
        """Truncate dataset on time to end column according to self.time_range."""
        try:
            min_time, max_time = self.time_range
            return X[X['time_to_end'].between(min_time, max_time, inclusive="left")]

        except Exception:
            raise ValueError(f"Could not truncate to {self.time_range} time range successfully")

    def _check_correct_time_conversion(self, X):
        """Check addition and truncation of time index worked accordingly."""

        cond1 = X[self.id_col].is_monotonic
        cond2 = X.groupby(self.id_col).apply(lambda x: x["time_to_end"].is_monotonic_decreasing).all()

        min_time, max_time = self.time_range
        cond3 = X["time_to_end"].between(min_time, max_time, inclusive='left').all()

        assert cond1 is True
        assert cond2 == True
        assert cond3 == True

    def subset_to_features(self, X):
        """Subset only to variables which were selected"""
        features = [self.id_col, "time_to_end"] + _get_features(self.feat_set, self.dataset_name)

        return X[features], features

    def convert_to_3darray(self, X):
        """Convert a pandas dataframe to 3D numpy array of shape (num_samples, num_timestamps, num_variables)."""

        # Obtain relevant shape sizes
        max_time_length = X.groupby(self.id_col).count()["time_to_end"].max()
        num_ids = X[self.id_col].nunique()

        # Other basic definitions
        feats = [col for col in X.columns if col not in [self.id_col, "time_to_end"]]
        list_ids = X[self.id_col].unique()

        # Initialise output array and id-time array
        out_array = np.empty(shape=(num_ids, max_time_length, len(feats)))
        out_array[:] = np.nan

        # Make a parallel array indicating id and corresponding time
        id_times_array = np.empty(shape=(num_ids, max_time_length, 2))

        # Set ids in this newly generated array
        id_times_array[:, :, 0] = np.repeat(np.expand_dims(list_ids, axis=-1), repeats=max_time_length, axis=-1)

        # Iterate through ids
        for id_ in tqdm(list_ids):
            # Subset data to where matches respective id
            index_ = np.where(list_ids == id_)[0]
            x_id = X[X[self.id_col] == id_]

            # Compute negative differences instead of keeping the original times.
            x_id_copy = x_id.copy()
            x_id_copy["time_to_end"] = - x_id["time_to_end"].diff().values

            # Update target output array and time information array
            out_array[index_, :x_id_copy.shape[0], :] = x_id_copy[feats].values
            id_times_array[index_, :x_id_copy.shape[0], 1] = x_id["time_to_end"].values

        return out_array.astype("float32"), id_times_array.astype("float32")

    def normalise(self, X):
        """Given 3D array, normalise according to min-max method."""
        self.min = np.nanmin(X, axis=0, keepdims=True)
        self.max = np.nanmax(X, axis=0, keepdims=True)

        return np.divide(X - self.min, self.max - self.min)

    def apply_normalisation(self, X):
        """Apply normalisation with current parameters to another dataset."""
        if self.min is None or self.max is None:
            raise ValueError(f"Attributes min and/or max are not yet computed. Run 'normalise' method instead.")

        else:
            return np.divide(X - self.min, self.max - self.min)
