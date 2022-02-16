"""
Utility functions for the SOM-VAE model
Copyright (c) 2018
Author: Vincent Fortuin
Institution: Biomedical Informatics group, ETH Zurich
License: MIT License
"""

import numpy as np
import os
import pandas as pd
from tslearn.utils import to_time_series_dataset


def name_vit_keys(name):
    """
    Given name, obtain relevant set of features
    """
    vitals = ['HR', 'RR', 'SBP', 'DBP', 'SPO2', 'FIO2', 'TEMP', 'AVPU']
    serum  = ['HGB', 'WGC', 'EOS', 'BAS', 'EBR', 'NEU', 'LYM', 'NLR']
    biochem= ['ALB', 'CR', 'CRP', 'POT', 'SOD', 'UR']
    static = ['age', 'gender', 'cci', 'elective_admission', 'surgical_admission']
    
    # Vitals are always considered.
    out_feats_ = set(vitals)
    
    if 'vit' in name.lower():
        # Add vitals
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
        
        
    if 'all' in name.lower():
        # Select all variables
        out_feats_ = str(name_vit_keys('bio-ser-sta'))
    
    return list(out_feats_)
    


def load_from_csv(folder_path, data_name, time_range = (0, 3), features = 'vitals'):
    
    """
    Import Data given csv file. We check for feature selection, time-period selection and normalisation.
    
    Input:
        - folder_path: a path corresponding to the overall folder path
        - data_name: name of vital data or tuple of vital name and outcome name
        - time_range: Number of maximum days to outcome to consider. 
                Alternatively, a string with lower and upper bounds to consider (default: (0, 3))
        - features: Set of features to consider. Vitals always considered (default: 'vitals')
        
    
    - Data is loaded according to feature set and dtime_range.
    - Data is normalised
    
    
    - Return triple of arrays corresponding to vital signs and target outcomes/phenotypes and ids (may be disregarded if extension npy or npz)
    """
    if not os.path.exists(folder_path):
        print('Wrong folder_path specified!')
        return None
    
    # Analyse data_name
    if type(data_name) == tuple:
        # X, y names are given
        X_name, y_name = data_name
        
    else:
        # assume X name is given only
        X_name         = data_name
        assert type(X_name) == str
        
        y_name         = 'y'
        
    
    if 'csv' not in X_name[-3:]:
        X_name = X_name + '.' + 'csv'
        
    if 'csv' not in y_name[-3:]:
        y_name = y_name + '.' + 'csv'
        
    
    # Specify data path
    data_path  = folder_path + X_name
    assert os.path.exists(data_path) or os.path.exists(data_path + '.csv') or os.path.exists(data_path + '.npy')

    
    try:
        X = pd.read_csv(data_path, parse_dates = ['charttime', 'hadm_end_time', 'hadm_start_time', 'event_time'])
        y = pd.read_csv(folder_path + y_name, index_col = 0)
        
        print('Data {} loaded successfully.'.format(data_path))
            
    except:
        print('Wrong data name specified')
        raise ValueError
    
    
    # Convert with timedeltas
    if 'time_to_outcome' in X.columns:
        X['time_to_outcome'] = pd.to_timedelta(X.loc[:, 'time_to_outcome'])
        X.sort_values(by = ['subject_id', 'time_to_outcome'], ascending = [True, False], inplace = True)
        
    elif 'time_to_outcome' not in X.columns:
        print('Computing time to outcome')
        
        # Compute time to outcome
        X['time_to_outcome'] = X.groupby('subject_id').apply(lambda x: x['charttime'].norm_max() - x['charttime']).values
        X.sort_values(by = ['subject_id', 'time_to_outcome'], ascending = [True, False], inplace = True)
        
    assert X.subject_id.is_monotonic
    print('Data sorted and timedeltas, datetimes converted.')
        
        
    # Select time_range
    if type(time_range) == tuple:
        start_, stop_ = time_range
        
    else:
        # Time Range only the upper bound. Lower bound assumed to be 0
        stop_  = time_range
        start_ = 0    
    
    
    # Select admission between start_ and stop_
    data_time_reg_  = X[X['time_to_outcome'].dt.total_seconds().between(start_ * 24 * 3600, stop_ * 24 * 3600, inclusive=  True)]
    print('Data subsetted to time window of interest - ({} - {}) days before an outcome'.format(start_, stop_))
    
    
    # Subset to features
    X_features      = data_time_reg_[name_vit_keys(features) + ['subject_id']]
    ids             = data_time_reg_[['subject_id', 'hadm_id', 'time_to_outcome']]
    print('Data subsetted to specific features.')
    
    
    # Normalise Data
    print('\n --------------------- \n')
    print('Normalising Data as min-max scaler')
    x_npy_          = X_features.groupby('subject_id').apply(lambda x: x[name_vit_keys(features)].to_numpy()).values
    x_npy_          = to_time_series_dataset(x_npy_)
    assert len(x_npy_.shape) == 3
    
    # Average across batch for each time and input dimension
    batch_min       = np.nanmin(x_npy_, axis = 0).reshape(1, x_npy_.shape[1], x_npy_.shape[2])
    batch_range     = np.nanmax(x_npy_, axis = 0) - batch_min
    X   = np.divide(x_npy_ - batch_min, batch_range)

    # Check normalisation
    if np.isnan(X).sum() > 0:              # Some nan values not filled exactly
        print('Not all values filled! ')
        X   = np.nan_to_num(X, copy = False, nan = 0.0)
        assert np.any(np.isnan(X)) == False
        
        
    # Check normalisation was correct
    assert np.all(np.abs(np.amin(X, axis = 0)) < 1e-8)
    assert np.all(np.abs(np.nanmax(X, axis = 0) - np.nanmin(X, axis = 0) - 1) < 1e-8)

    print('Normalisation and NAn division computed.\n')
    
    
    # Obtain phenotypes
    y_data          = y[['Healthy', 'Death', 'ICU', 'Card']]
    y_npy_          = y_data.to_numpy()
    
    print('Shapes :', y_npy_.shape, x_npy_.shape)
    
    # Check things make sense
    assert np.sum(y_data.index != ids.subject_id.unique()) == 0
    assert y_npy_.shape[0] == x_npy_.shape[0]
    assert len(X.shape) == 3
    assert X.shape[0] == ids.subject_id.nunique()
    
    return X.astype('float32'), y_npy_.astype('float32'), ids
    
    
    
    
    
    
    
    
    
    

def interpolate_arrays(arr1, arr2, num_steps=100, interpolation_length=0.3):
    """Interpolates linearly between two arrays over a given number of steps.
    The actual interpolation happens only across a fraction of those steps.

    Args:
        arr1 (np.array): The starting array for the interpolation.
        arr2 (np.array): The end array for the interpolation.
        num_steps (int): The length of the interpolation array along the newly created axis (default: 100).
        interpolation_length (float): The fraction of the steps across which the actual interpolation happens (default: 0.3).

    Returns:
        np.array: The final interpolated array of shape (num_steps,arr1.shape).
    """
    assert arr1.shape == arr2.shape, "The two arrays have to be of the same shape"
    
    # Define start, intermediate and end steps
    start_steps = int(num_steps*interpolation_length)
    inter_steps = int(num_steps*((1-interpolation_length)/2))
    end_steps = num_steps - start_steps - inter_steps
    
    # Initialize interpolation # Shape (num inter steps, arr1.shape)
    interpolation = np.zeros([inter_steps]+list(arr1.shape))
    
    # Compute difference
    arr_diff = arr2 - arr1
    
    # Iterate over number of inter_steps
    for i in range(inter_steps):
        interpolation[i] = arr1 + (i/(inter_steps-1))*arr_diff
        
    # Expand start and end arrays
    start_arrays = np.concatenate([np.expand_dims(arr1, 0)] * start_steps)
    end_arrays = np.concatenate([np.expand_dims(arr2, 0)] * end_steps)
    
    # Adjoin all arrays.
    final_array = np.concatenate((start_arrays, interpolation, end_arrays))
    
    return final_array


def compute_NMI(cluster_assignments, class_assignments):
    """Computes the Normalized Mutual Information between cluster and class assignments.
    Compare to https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    
    Args:
        cluster_assignments (list): List of cluster assignments for every point.
        class_assignments (list): List of class assignments for every point.

    Returns:
        float: The NMI value.
    """
    assert len(cluster_assignments) == len(class_assignments), "The inputs have to be of the same length."
    
    # clusters = prediction, classes = true values
    clusters = np.unique(cluster_assignments)
    classes = np.unique(class_assignments)
    
    # Number of clusters identified, samples and clusters.
    num_samples = len(cluster_assignments)
    num_clusters = len(clusters)
    num_classes = len(classes)
    
    assert num_classes > 1, "There should be more than one class."
    
    # Dictionary of dictionaries 
    cluster_class_counts = {cluster_: {class_: 0 for class_ in classes} for cluster_ in clusters}
    
    # Iterate through zip of both 
    for cluster_, class_ in zip(cluster_assignments, class_assignments):
        cluster_class_counts[cluster_][class_] += 1
    
    cluster_sizes = {cluster_: sum(list(class_dict.values())) for cluster_, class_dict in cluster_class_counts.items()}
    class_sizes = {class_: sum([cluster_class_counts[clus][class_] for clus in clusters]) for class_ in classes}
    
    I_cluster_class = H_cluster = H_class = 0
    
    for cluster_ in clusters:
        for class_ in classes:
            if cluster_class_counts[cluster_][class_] == 0:
                pass
            else:
                I_cluster_class += (cluster_class_counts[cluster_][class_]/num_samples) * \
                (np.log((cluster_class_counts[cluster_][class_]*num_samples)/ \
                        (cluster_sizes[cluster_]*class_sizes[class_])))
                        
    for cluster_ in clusters:
        H_cluster -= (cluster_sizes[cluster_]/num_samples) * np.log(cluster_sizes[cluster_]/num_samples)
                
    for class_ in classes:
        H_class -= (class_sizes[class_]/num_samples) * np.log(class_sizes[class_]/num_samples)
        
    NMI = (2*I_cluster_class)/(H_cluster+H_class)
    
    return NMI


def compute_purity(cluster_assignments, class_assignments):
    """Computes the purity between cluster and class assignments.
    Compare to https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    
    Args:
        cluster_assignments (list): List of cluster assignments for every point.
        class_assignments (list): List of class assignments for every point.

    Returns:
        float: The purity value.
    """
    assert len(cluster_assignments) == len(class_assignments)
    
    num_samples = len(cluster_assignments)
    num_clusters = len(np.unique(cluster_assignments))
    num_classes = len(np.unique(class_assignments))
    
    cluster_class_counts = {cluster_: {class_: 0 for class_ in np.unique(class_assignments)}
                            for cluster_ in np.unique(cluster_assignments)}
    
    for cluster_, class_ in zip(cluster_assignments, class_assignments):
        cluster_class_counts[cluster_][class_] += 1
        
    total_intersection = sum([max(list(class_dict.values())) for cluster_, class_dict in cluster_class_counts.items()])
    
    purity = total_intersection/num_samples
    
    return purity
