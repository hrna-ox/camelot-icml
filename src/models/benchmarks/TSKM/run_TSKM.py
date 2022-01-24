import os
try:
    os.chdir("Desktop/COPD/adding-attention/scripts/models/actpc-patient/")
except:
    print(os.getcwd())
    
import utils
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

# from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.metrics.cluster import contingency_matrix

def purity_score(y_true, y_pred):
    contingency_matrix_ = contingency_matrix(y_true, y_pred)
    
    return np.sum(np.amax(contingency_matrix_, axis = 0)) / np.sum(contingency_matrix_)


np.set_printoptions(precision=3, suppress = True)

from tqdm import tqdm
import pandas as pd
import itertools

import os
import sys

possible_clusters = range(2, 50)

np.set_printoptions(precision=3, suppress = True)
sys.path.append("scripts/models/actpc-patient")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load configuration parameters
load_init_config = {"id_column": 'subject_id', "time_column": 'charttime', "feature_set_names": 'vitals', 
                    "fill_limit": None, "norm_method": None,
                    "roughly_balanced": None}
load_config = {"folder_dir": '/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/data/processed/',
    "X_y_name": ('COPD_VLS_process', 'copd_outcomes'), "time_range": (24, 72), "feature_set": 'vit-lab-sta', "include_time": None}
\


def main():
    return None


for i in range(1):
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
    
    
    # Make predictions
    predictions_df = pd.DataFrame(data = np.nan, index = range(X_test.shape[0]), 
                          columns = ["k{}_s{}".format(k, s) for k,s in  itertools.product(possible_clusters, generated_seeds)])
    
    for seed in tqdm(generated_seeds):
        for k in possible_clusters:
            km = KMeans(n_clusters = k, init = 'k-means++', n_init = 20, max_iter = 500, tol = 1e-6,
                verbose = 0, random_state = seed)
            km.fit(X_train.reshape(X_train.shape[0], -1))
            predictions_df["k{}_s{}".format(k, seed)] = km.predict(X_test.reshape(X_test.shape[0], -1))
            predictions_df["k{}_s{}".format(k, seed)]
            
    # save
    predictions_df.to_csv("tksm_predictions")
    
    # compute scores
    predictions_df = pd.read_csv("../TSKM/tksm_predictions", index_col = 0, header = 0)
    scores_df = pd.DataFrame(data = np.nan, index = ["SIL_eucl","SIL_cos", "DBI", "VRI", "k", "seed"], columns = [])    
    labels_true      = np.argmax(y_test, axis = 1)
    y_scores_df = pd.DataFrame(data = np.nan, index = ["auc_unweig", "auc_weig", "f1", "bac", "rec", "nmi", "k", "seed"],
                               columns = predictions_df.columns)
    
    
    for seed in tqdm(generated_seeds):
        for k in possible_clusters:
            col = "k{}_s{}".format(k, seed)
            X, labels = X_test.reshape(X_test.shape[0], -1), predictions_df["k{}_s{}".format(k, seed)]
            
            sil_euc = silhouette_score(X, labels, metric = "euclidean")
            sil_cos = silhouette_score(X, labels, metric = "cosine")
            dbi = davies_bouldin_score(X, labels)
            vri = calinski_harabasz_score(X, labels)
            
            scores_df[col] = [sil_euc, sil_cos, dbi, vri, k, seed]
        
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
            
            y_scores_df[col] = [auc_unweig, auc_weig, f1, bac, rec, nmi, k, seed]
                
            
            
    scores_df = scores_df.T
    scores_df["seed"] = [int(info[1]) for info in scores_df.index.values()]
    scores_df["k"]    = [int(info[0]) for info in scores_df.index.values()]
    
    # Visualise results
    mean_results = scores_df.groupby("k", as_index = True).mean().sort_index()
    std_results  = scores_df.groupby("k", as_index = True).std().sort_index()

    y_scores_df = y_scores_df.T.astype(float)
    y_mean_results = y_scores_df.groupby(["k"], as_index = True).mean()
    y_std_results = y_scores_df.groupby(["k"], as_index = True).std()
    








    
    best_k = pd.Series(data = np.nan, index = ["SIL_eucl", "SIL_cos", "DBI", "VRI"])
    best_k.loc["SIL_eucl", "SIL_cos", "VRI"] = mean_results.idxmax(axis = 0)[["SIL_eucl", "SIL_cos", "VRI"]]
    best_k.loc["DBI"] = mean_results.idxmin(axis = 0)["DBI"]
    
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    colors = get_cmap("tab10").colors
    
    fig, ax = plt.subplots(nrows = 2, ncols = 2, sharex = True)
    ax = ax.reshape(-1)
    
    for id_ in range(4):
        col = mean_results.columns[id_]
        color = colors[id_]
        
        ax[id_].plot(mean_results.index, mean_results[col], color = color)
        ax[id_].plot(mean_results.index, mean_results[col] + std_results[col], linestyle = "--", alpha = 0.6, color = color)
        ax[id_].plot(mean_results.index, mean_results[col] - std_results[col], linestyle = "--", alpha = 0.6, color = color)
        
        ax[id_].axvline(best_k.loc[col], linestyle = "--", color = "black", label = "best = %d" % best_k.loc[col])
        
        ax[id_].set_title("% results" % col)
        ax[id_].legend()
        
    # Best result is 8 - display metrics
    for col in best_k.index:
        print("TSKM result {}: {:.2f} ({:.2f})".format(col, mean_results.loc[8, col], std_results.loc[8, col]))
    
    
if __name__ == "__main__":
    main()