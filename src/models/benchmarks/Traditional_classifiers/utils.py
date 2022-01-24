#!/usr/bin/env python3
"""
Util functions to run Model. Includes Data loading, etc...
"""
from logging import lastResort
import os, sys
import datetime as dt

import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.stats import mode
from tslearn.utils import to_time_series_dataset
from tqdm import tqdm

time_columns = ['charttime', 'hadm_end_time', 'hadm_start_time', 'event_time']
vitals       = ['HR', 'RR', 'SBP', 'DBP', 'SPO2', 'FIO2', 'TEMP', 'AVPU']
serum        = ['HGB', 'WBC', 'EOS', 'BAS', 'EBR', 'NEU', 'LYM', 'NLR']
biochem      = ['ALB', 'CR', 'CRP', 'POT', 'SOD', 'UR']
static       = ['age', 'gender', 'cci', 'is_elec', 'is_surg']
target_vars  = ['Healthy', 'Death', 'ICU', 'Card']

def _check_is_folder_(folder_path):
    "Check folder_path is a valid folder_path"

    try:
        assert os.path.exists(folder_path)
    except Exception as e:
        raise ValueError("Folder path does not exist - Input {}".format(folder_path))

def _convert_datetime_to_hour(Series):
    "Convert pandas Series of datetime values to float Series with corresponding hour values"
    seconds_per_hour = 3600

    return Series.dt.total_seconds()/seconds_per_hour

def _get_features_from_name(name = "vitals"):
    """
    Given name, obtain relevant set of features
    """
    global vitals, serum, biochem, static
    
    # Start with vitals and add
    out_feats_ = set([])

    if name is None:
        return vitals
    
    if 'vit' in name.lower():
        out_feats_.update(vitals)
        
    
    if 'ser' in name.lower():
        # Add serum variables
        out_feats_.update(serum)
    
    
    if 'bio' in name.lower():
        # Add biochem variables
        out_feats_.update(biochem)
        
    
    if 'sta' in name.lower():
        # Add static variables
        out_feats_.update(static)
        
    
    if 'lab' in name.lower():
        # Add serum and biochem variables
        out_feats_.update(biochem)
        out_feats_.update(serum)
        
        
    if 'all' in name.lower():
        # Select all variables
        out_feats_ = str(_get_features_from_name('bio-ser-sta'))
        
    print("Subsetting to {} features: {}".format(name, out_feats_))
    
    return list(out_feats_)
    
def _numpy_forward_fill(array):
    "Forward Fill a numpy array. Time index is axis = 1"
    array_mask = np.isnan(array)
    array_out  = np.copy(array)

    # Add time indices where not masked, and propagate forward
    inter_array = np.where(~ array_mask, np.arange(array_mask.shape[1]).reshape(1, -1, 1), 0)
    np.maximum.accumulate(inter_array, axis = 1, out = inter_array)   # For each (n, t, d) missing value, get the previously accessible mask value
    
    # Index matching for output. For n, d sample as previously, use inter_array for previous time id
    array_out = array_out[np.arange(array_out.shape[0])[:, None, None],
                        inter_array,
                        np.arange(array_out.shape[-1])[None, None, :]]

    return array_out

def _numpy_backward_fill(array):
    "Backward Fill a numpy array. Time index is axis = 1"
    array_mask = np.isnan(array)
    array_out  = np.copy(array)

    # Add time indices where not masked, and propagate backward
    inter_array = np.where(~ array_mask, np.arange(array_mask.shape[1]).reshape(1, -1, 1), array_mask.shape[1] - 1)
    inter_array = np.minimum.accumulate(inter_array[:, ::-1], axis = 1)[:, ::-1]   # For each (n, t, d) missing value, get the previously accessible mask value
    array_out = array_out[np.arange(array_out.shape[0])[:, None, None],
                        inter_array,
                        np.arange(array_out.shape[-1])[None, None, :]]

    return array_out
    
def _median_fill(array):
    "Median fill a numpy array. Time index is axis = 1"
    array_mask = np.isnan(array)
    array_out  = np.copy(array)

    # Compute median and impute
    array_med  = np.nanmedian(array, axis = 0, keepdims = True)
    array_out  = np.where(array_mask, array_med, array_out)

    return array_out

class data_processor():
    def __init__(self, id_column = 'subject_id', time_column = 'charttime', feature_set_names = 'vitals', fill_limit = None, mask_value = 0.0,
                norm_method = "min-max", roughly_balanced = None):
        self.id_col            = id_column
        self.time_col          = time_column
        self.feature_set_names = feature_set_names
        self.fill_limit        = fill_limit
        self.mask_value        = mask_value
        self.norm_method       = norm_method
        self.roughly_balanced  = roughly_balanced

        # Initialise to None
        self.time_range  = None
        self.feature_set = None
        self.include_time= None
        
    def _check_correct_time_conversion(self, X):
        "Check addition and truncation of time index worked accordingly."
        try:
            assert X[self.id_col].is_monotonic
            assert X.groupby(self.id_col).filter(lambda x: not x["time_to_end"].is_monotonic_decreasing).empty
            
            min_time, max_time = self.time_range
            assert X["time_to_end"].between(min_time, max_time, inclusive = 'left').all()

        except Exception:
            raise ValueError("Time to End column has not been correctly added")

    def _add_time_to_end(self, X):
        "Add new column to dataframe - this computes time to end of grouped observations"
        x_inter = X.copy(deep = True)

        # Compute datetime values for time until end of group of observations
        times = X.groupby(self.id_col).apply(lambda x:
                    x.loc[:, self.time_col].max() - x.loc[:, self.time_col])

        # add column to dataframe after converting to hourly times.
        x_inter["time_to_end"] = _convert_datetime_to_hour(times).values

        return x_inter

    def _truncate(self, X):
        "Truncate dataset on time to end column according to self.time_range."
        try:
            min_time, max_time = self.time_range
            return X[X['time_to_end'].between(min_time,  max_time, inclusive= "left")]

        except Exception:
            raise ValueError("Could not truncate to {} time range successfully".format(self.time_range))

    def _add_time_to_end_and_truncate(self, X):
        "Add time to end and truncate according to time_range"
        return self._truncate(self._add_time_to_end(X))

    def _subset_to_features(self, X):
        "Subset only to variables which were selected"
        features = [self.id_col, "time_to_end", "charttime"] + list(_get_features_from_name(self.feature_set))

        return X[features], features

    def convert_to_3Darray(self, X):
        "Convert a pandas dataframe to 3D numpy array of shape (num_samples, num_timesteps, num_variables)"

        # Obtain relevant shape sizes
        max_time_length, num_ids = X["time_to_end"].nunique(), X[self.id_col].nunique()
        feats                    = [col for col in X.columns if col not in [self.id_col, self.time_col, "time_to_end"]]
        ids                      = X[self.id_col].unique()

        if self.include_time:
            feats = feats + ["time_to_end"]
        
        # Initialise output array and id-time array
        out_array      = np.empty(shape = (num_ids, max_time_length, len(feats)))
        out_array[:]   = np.nan

        id_times_array = np.empty(shape = (num_ids, max_time_length, 2))
        id_times_array[:, :, 0] = np.repeat(np.expand_dims(ids, axis = -1), repeats = max_time_length, axis = -1)

        # Iterate through ids
        for id_ in tqdm(ids):
            index_ = np.where(ids == id_)[0]
            x_id = X[X[self.id_col] == id_]

            # Update target output array
            out_array[index_, :x_id.shape[0], :] = x_id[feats].values
            id_times_array[index_, :x_id.shape[0], 1] = x_id["time_to_end"].diff().values

        return out_array.astype("float32"), id_times_array.astype("float32")

    def normalise(self, X):
        "Given 3D array, normalise according to normalisation method. Available choices include None, min-max and norm."
        
        print("Normalised!")
        if self.norm_method is None:
            # Not normalising
            print("Inputs have not been normalised!")
            self.mu, self.sigma = None, None

            return X
        
        elif self.norm_method.lower() == 'min-max':
            self.mu = np.nanmin(X, axis = 0)
            self.sigma = np.nanmax(X, axis = 0) - np.nanmin(X, axis = 0)

            return np.divide(X - self.mu, self.sigma)

        elif self.norm_method.lower() == 'norm':
            self.mu = np.nanmean(X, axis = 0)
            self.sigma = np.nanstd(X, axis = 0)

            return np.divide(X - self.mu, self.sigma)

        else:
            raise ValueError("Norm Method attribute not valid: Got {}".format(self.norm_method))
 

    def apply_normalise(self, X):
        "Normalise X according to existing mean and sigma"
        return np.divide(X - self.mu, self.sigma)

    def impute(self, X):
        "Imputation of 3D array accordingly: 1st - forward value propagation, 2nd - backwards value propagation, 3rd - median value imputation. Mask kept from the first iteration."
        impute_step1 = _numpy_forward_fill(X)
        impute_step2 = _numpy_backward_fill(impute_step1)
        impute_step3 = _median_fill(impute_step2)

        # Compute mask
        mask = np.isnan(X)

        return impute_step3, mask

    def _load(self, folder_dir, X_y_name, time_vars):
        """
        Given a folder path and name for X, y datasets to load, load data according to time-vars in X data.
        """
        X_path, y_path = folder_dir + X_y_name[0] + ".csv", folder_dir + X_y_name[1] + ".csv"

        try:
            X = pd.read_csv(X_path, parse_dates = time_vars)
            y = pd.read_csv(y_path, index_col = 0)
            print("Data {} - {} loaded successfully.".format(X_y_name[0], X_y_name[1]))

            return X, y

        except Exception:
            raise ValueError("""Data incorrectly loaded with inputs:\n
                                Folder: {} \n 
                                X-y name: {} - {} \n
                                time_vars: {}""".format(folder_dir, X_y_name[0], X_y_name[1], time_vars))

    def load_transform(self, folder_dir, X_y_name, time_range = (24, 72), feature_set = "vitals", include_time = None):
        """
        Load and transform the dataset for inputting to model. 
        This function updates transformation parameters: time_range, feature_set and include_time.

        returns: transformed X, y arrays ready for model imput, as given by transform method.
        """
        self.include_time = include_time
        self.feature_set  = feature_set
        self.time_range   = time_range

        # Check folder exists
        _check_is_folder_(folder_dir)

        # Load data
        global time_columns
        X, y = self._load(folder_dir, X_y_name, time_columns)
        
        return self.transform(X, y)
        
    def transform(self, X, y):

        # make copies
        x_inter = self._add_time_to_end_and_truncate(X)
        self._check_correct_time_conversion(x_inter)

        # subset to relevant features
        x_inter, feats = self._subset_to_features(x_inter)

        # Convert to 3D array
        x_inter, ids = self.convert_to_3Darray(x_inter)

        # Normalise array
        x_inter = self.normalise(x_inter)

        # Impute missing values
        x_out, mask = self.impute(x_inter)

        # Do things to y
        y_data           = y[target_vars]
        y_out            = y_data.to_numpy()  

        if self.roughly_balanced:
            x_out, y_out, mask, ids = self._subset_to_balanced(x_out, y_out, mask, ids)

        # Check data loaded correctly
        self._check_input_format(x_out, y_out)

        return x_out.astype("float32"), y_out.astype("float32"), mask, ids, feats

    def _check_input_format(self, X, y):
        "Check conditions to confirm model input."

        try:
            # Length and shape conditions
            assert X.shape[0] == y.shape[0]
            assert len(X.shape) ==3
            assert len(y.shape) == 2

            # Check non-missing values
            assert np.sum(np.isnan(X)) + np.sum(np.isnan(y)) == 0

            # Check normalisation
            if self.norm_method is not None:
                assert np.all(np.abs(np.amin(X, axis = 0)) < 1e-8)
                assert np.all(np.abs(np.nanmax(X, axis = 0) - np.nanmin(X, axis = 0)) - 1 < 1e-8)

            # Check y output is one hot encoded
            assert np.all(np.sum(y, axis = 1) == 1)

        except Exception as e:
            print(e)
            raise AssertionError("One of the check conditions has failed.")

    def _subset_to_balanced(self, X, y, mask, ids):
        "Subset samples so dataset is more well sampled."
        class_numbers     = np.sum(y, axis = 0)
        largest_class, target_num_samples = np.argmax(class_numbers), np.sort(class_numbers)[-2]
        print("Subsetting class {} from {} to {} samples.".format(largest_class, class_numbers[largest_class], target_num_samples))

        # Select random
        largest_class_ids = np.arange(y.shape[0])[y[:, largest_class] == 1]
        class_ids_samples = np.random.choice(largest_class_ids, size = target_num_samples, replace = False)
        ids_to_remove_    = np.setdiff1d(largest_class_ids, class_ids_samples)

        # Remove relevant ids
        X_out    = np.delete(X, ids_to_remove_, axis = 0)
        y_out    = np.delete(y, ids_to_remove_, axis = 0)
        mask_out = np.delete(mask, ids_to_remove_, axis = 0)
        ids_out  = np.delete(ids, ids_to_remove_, axis = 0)

        return X_out, y_out, mask_out, ids_out


def merge_dictionaries(dict1, dict2, alpha, beta):
    "Merge configurations for initialisation models with new alpha, beta"
    output_dic = dict1.copy()

    for key, value in dict2.items():
        output_dic[key] = value

    # Update alpha and beta
    output_dic["alpha"] = alpha
    output_dic["beta"]  = beta

    return output_dic

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




















