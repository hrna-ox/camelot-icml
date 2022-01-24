from tqdm import tqdm
import itertools
import os
import utils

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.metrics.cluster import contingency_matrix

def purity_score(y_true, y_pred):
    contingency_matrix_ = contingency_matrix(y_true, y_pred)
    
    return np.sum(np.amax(contingency_matrix_, axis = 0)) / np.sum(contingency_matrix_)


np.set_printoptions(precision=3, suppress = True)

# generated_seeds = [1001, 1012,1134, 2475, 6138, 7415, 1663, 7205, 9253, 1782]
generated_seeds_5 = [1001, 1012, 1134, 2475, 6138]
parameter_set   = [0.001, 0.1]
K_set           = [4, 8, 12, 20, 50]

cluster_name = "ACTPC-labels.csv"
y_name      = "ACTPC-y.csv"

for alpha, beta, K, seed in itertools.product(parameter_set, parameter_set, 
                                              K_set,  generated_seeds_5):
    print(alpha, beta, K, seed)
    os.system("python run-actpc.py --alpha {} --beta {} --K {} --seed {}".format(
        alpha, beta, K, seed))
        


# Load configuration parameters
load_init_config = {"id_column": 'subject_id', "time_column": 'charttime', "feature_set_names": 'vitals', 
                    "fill_limit": None, "norm_method": None,
                    "roughly_balanced": None}
load_config = {"folder_dir": '/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/data/processed/',
    "X_y_name": ('COPD_VLS_process', 'copd_outcomes'), "time_range": (24, 72), "feature_set": 'vit-lab-sta', "include_time": None}



if __name__ == "__main__":
    data_processor = utils.data_processor(**load_init_config)
    X, y, mask, ids, feats = data_processor.load_transform(**load_config)
    

    # Separate into train test and normalise
    X_train, X_test, y_train, y_test, id_train, id_test, mask_train, mask_test = train_test_split(
        X, y, ids, mask, train_size=0.4, random_state=2323,
        shuffle=True, stratify=np.argmax(y, axis=-1))

    data_processor.norm_method = 'min-max'
    X_train = data_processor.normalise(X_train)
    X_test = data_processor.apply_normalise(X_test)

    # Separate into train and validation test
    X_train, X_val, y_train, y_val, id_train, id_val, mask_train, mask_val = train_test_split(
        X_train, y_train, id_train, mask_train, train_size=0.6, random_state=2323,
        shuffle=True, stratify=np.argmax(y_train, axis=-1))
    

    # Load labels
    cluster_predictions = pd.read_csv(cluster_name, index_col = 0, header = 0)
    assert cluster_predictions.shape[0] == X_test.shape[0]

    scores_df = pd.DataFrame(data = np.nan, index = ["SIL_eucl","SIL_cos", "DBI", "VRI", "purity", "alpha", "beta", "seed", "K"], columns = cluster_predictions.columns)
    X = X_test.reshape(X_test.shape[0], -1)
    
    # Iterate through the columns
    for col in tqdm(cluster_predictions.columns):
        labels = cluster_predictions[col]
        alpha, beta, seed, K = [float(string[1:]) for string in col.split("_")]
        
        if labels.nunique() == 1:
            sil_euc, sil_cos, dbi, vri, purity = np.nan, np.nan, np.nan, np.nan
        else:
            sil_euc = silhouette_score(X, labels, metric = "euclidean")
            sil_cos = silhouette_score(X, labels, metric = "cosine")
            dbi = davies_bouldin_score(X, labels)
            vri = calinski_harabasz_score(X, labels)
            purity = purity_score(np.argmax(y_test, axis = 1), labels)
        
        scores_df[col] = [sil_euc, sil_cos, dbi, vri, purity, alpha, beta, seed, K]
        
    scores_df = scores_df.T.astype(np.float32).reset_index(drop = True)
    
    # Visualise results
    mean_results = scores_df.groupby(["alpha", "beta", "K"], as_index = True).mean()
    std_results  = scores_df.groupby(["alpha", "beta", "K"], as_index = True).std()
    
    
    # COMPUTE Y SCORES
    y_predictions_df = pd.read_csv(y_name, index_col = 0, header = 0)
    labels_true      = np.argmax(y_test, axis = 1)
    y_scores_df = pd.DataFrame(data = np.nan, index = ["auc_unweig", "auc_weig", "f1", "bac", "rec", "nmi",
                                                       "alpha", "beta", "seed", "K"],
                               columns = y_predictions_df.columns)
    
    for col in tqdm(y_predictions_df.columns):
        y_pred = y_predictions_df[col].values.reshape(X_test.shape[0], -1)
        labels_pred = np.argmax(y_pred, axis = 1)
        
        try:
            alpha, beta, seed, K = [float(string[1:]) for string in col.split("_")]
        except Exception:
            pass
        
        # Compute scores
        auc_unweig = roc_auc_score(labels_true, y_pred, average = "macro", multi_class = "ovr")
        auc_weig   = roc_auc_score(labels_true, y_pred, average = "weighted",  multi_class = "ovr")
        f1  = f1_score(labels_true, labels_pred, average = "macro")
        bac = balanced_accuracy_score(labels_true, labels_pred)
        rec = recall_score(labels_true, labels_pred, average = "macro")
        nmi = normalized_mutual_info_score(labels_true, labels_pred)
        
        y_scores_df[col] = [auc_unweig, auc_weig, f1, bac, rec, nmi, alpha, beta, seed, K]
        
        
    # Average results
    y_scores_df = y_scores_df.T
    y_mean_results = y_scores_df.groupby(["alpha", "beta", "K"], as_index = True).mean()
    y_std_results = y_scores_df.groupby(["alpha", "beta", "K"], as_index = True).std()
        
        