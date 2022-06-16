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
latent_set      = [16, 64]

for alpha, beta, gamma, tau, latent_dim, seed in itertools.product(parameter_set, parameter_set, parameter_set, parameter_set,
                                                           latent_set, generated_seeds_5):
    
    os.system("python somvae_train.py with alpha={} beta={} gamma={} tau={} latent_dim={} seed={}".format(
        alpha, beta, gamma, tau, latent_dim, seed))
        


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
    predictions = pd.read_csv("SOM-VAE-labels.csv", index_col = 0, header = 0)
    assert predictions.shape[0] == X_test.shape[0]

    scores_df = pd.DataFrame(data = np.nan, index = ["SIL_eucl","SIL_cos", "DBI", "VRI", "alpha", "beta", "gamma", "tau", "latent_dim", "seed"], columns = predictions.columns)
    X = X_test.reshape(X_test.shape[0], -1)
    labels_true      = np.argmax(y_test, axis = 1)
    y_scores_df = pd.DataFrame(data = np.nan, index = ["auc_unweig", "auc_weig", "f1", "bac", "rec", "nmi",
                                                       "alpha", "beta", "gamma", "tau", "latent_dim", "seed"],
                               columns = predictions.columns)
    
    # Iterate through the columns
    for col in tqdm.tqdm(predictions.columns):
        labels = predictions[col]
        alpha, beta, gamma, tau, latent_dim, som_dim, seed = [string[1:] for string in col.split("_")]
        
        if labels.nunique() == 1:
            sil_euc, sil_cos, dbi, vri = np.nan, np.nan, np.nan, np.nan
        else:
            sil_euc = silhouette_score(X, labels, metric = "euclidean")
            sil_cos = silhouette_score(X, labels, metric = "cosine")
            dbi = davies_bouldin_score(X, labels)
            vri = calinski_harabasz_score(X, labels)
        
        scores_df[col] = [sil_euc, sil_cos, dbi, vri, alpha, beta, gamma, tau, latent_dim, seed]
        
        y_scores = np.zeros(shape = (labels_true.shape[0], 4))
        array = np.zeros(shape = (np.unique(labels).size, 4))
        i = 0
        for label in np.unique(labels):
           array[i, :] = list(np.sum(y_test[labels == label, :].reshape(-1, 4),
                                             axis = 0))
           i = i + 1
           
        labelling_pred = pd.DataFrame(array, index = np.unique(labels))
        labelling_pred = labelling_pred.divide(np.sum(labelling_pred, axis = 1),
                                   axis = 0)
        
        for pat_id in range(y_scores.shape[0]):
            y_scores[pat_id, :] = labelling_pred.loc[labels[pat_id], :].values
        
        labels_pred = np.argmax(y_scores, axis = 1)
        
        # Compute scores
        auc_unweig = roc_auc_score(labels_true, y_scores, average = "macro", multi_class = "ovr")
        auc_weig   = roc_auc_score(labels_true, y_scores, average = "weighted",  multi_class = "ovr")
        f1  = f1_score(labels_true, labels_pred, average = "macro")
        bac = balanced_accuracy_score(labels_true, labels_pred)
        rec = recall_score(labels_true, labels_pred, average = "macro")
        nmi = normalized_mutual_info_score(labels_true, labels_pred)
        
        y_scores_df[col] = [auc_unweig, auc_weig, f1, bac, rec, nmi, alpha, beta,gamma, tau, latent_dim, seed]
    
    
    
    
    scores_df = scores_df.T.astype(np.float32).reset_index(drop = True)
    
    # Visualise results
    mean_results = scores_df.groupby(["alpha", "beta", "gamma", "tau", "latent_dim"], as_index = False).mean()
    std_results  = scores_df.groupby(["alpha", "beta", "gamma", "tau", "latent_dim"], as_index = False).std()
    
    # Average results
    y_scores_df = y_scores_df.T.astype(float)
    y_mean_results = y_scores_df.groupby(["alpha", "beta", "gamma", "tau", "latent_dim"], as_index = True).mean()
    y_std_results = y_scores_df.groupby(["alpha", "beta", "gamma", "tau", "latent_dim"], as_index = True).std()
    
    
        
        
   
        
    