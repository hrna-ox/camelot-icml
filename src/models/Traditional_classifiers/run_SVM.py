import src.models.benchmarks.Traditional_classifiers.utils as utils
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, normalized_mutual_info_score

from tqdm import tqdm
import itertools

from sklearn.svm import SVC
np.set_printoptions(precision=3, suppress = True)



# ------------------------------ Auxiliary functions --------------------
def fit_trajectories(X: np.ndarray, y: np.ndarray):
    """
    

    Parameters
    ----------
    X : np.ndarray
        DESCRIPTION.
    y : np.ndarray
        DESCRIPTION.

    Returns
    -------
    None.

    """
def convert_to_news_format(X: np.ndarray, y: np.ndarray):
    """
    Conversion of 3D array numpy array to time-wise classification 
    for comparison with NEWS II algorithm

    Parameters
    ----------
    X : np.ndarray of shape N x T x D, where N is the number of samples,
    T is the maximum length of observations and D is the number of features.
    
    y : np.ndarray of target labels of shape N x O, where N is the number 
    of samples, and O is the number of outcome classes. y in one-hot format.

    Returns
    -------
    data_feats :  2D numpy array of shape (N x D) x T
    labels : 2D numpy array of shape (NxD) x T.
    """
    assert X.shape[0] == y.shape[0]
    
    # Obtain dimension information
    N, T, D = X.shape
    N, O = y.shape
    
    
    # Re-order axis and convert
    # X_new = np.transpose(X, axes = [0, 2, 1])       # shape N x D x T
    X_new = np.transpose(X, axes = [0, 1, 2])
    data_feats = np.reshape(X_new, newshape = (np.multiply(N, T), D))
    
    # Convert labels
    # labels = np.repeat(np.expand_dims(y, axis = 1), repeats = D, axis = 1).reshape((np.multiply(N, D), O))
    labels = np.repeat(np.expand_dims(y, axis = 1), repeats = T, axis = 1).reshape((np.multiply(N, T), O))
    
    return data_feats, labels



def compute_mode(array: np.ndarray) -> np.ndarray:
    """
    Compute most common value in array the last axis
    
    Parameters
    ------------
    array: np.ndarray of shape  A x D, where D is the number of features
    
    Returns
    ------------
    mode: np.ndarray of shape (A, ) with the mode taken across D dimensions.
    """
    
    # Compute useful info
    A, D = array.shape
    array_1d = array.reshape(-1)
    num_classes = np.unique(array_1d).size
    
    
    # Convert to one hot encoding
    target_one_hot = np.eye(num_classes)[array_1d]
    one_hot_resh  = target_one_hot.reshape(A, D, num_classes)        
    
    
    # Compute mode as class column with highest indicators
    sum_feats = np.sum(one_hot_resh, axis = 1, keepdims = False)  # summing across features
    mode = np.argmax(sum_feats, axis = -1)                      # compute mode
    
    return mode
    
    

# ---------------------- Load configuration ----------------------------
load_init_config = {"id_column": 'subject_id', "time_column": 'charttime', 
                    "feature_set_names": 'vitals',  "fill_limit": None, 
                    "norm_method": None, "roughly_balanced": None}

load_config = {"folder_dir": '/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/data/processed/',
    "X_y_name": ('COPD_VLS_process', 'copd_outcomes'), "time_range": (24, 72),
    "feature_set": 'vit-lab-sta', "include_time": None}

# Varying params
generated_seeds = [1001, 1012,1134, 2475, 6138]
possible_clusters = range(2, 50)


# --------------------------- Main function ------------------------------
if __name__ == "__main__":
    
    # Load data
    data_processor = utils.DataProcessor(**load_init_config)
    X, y, mask, ids, feats = data_processor.load_transform(**load_config)
    

    # Train Test Split and Normalise
    X_train_all, X_test, y_train_all, y_test, id_train, id_test, mask_train, mask_test = train_test_split(
        X, y, ids, mask, train_size=0.4, random_state=2323,
        shuffle=True, stratify=np.argmax(y, axis=-1))

    data_processor.norm_method = 'min-max'
    X_train_all = data_processor.normalise(X_train_all)
    X_test = data_processor.apply_normalise(X_test)
    
    
    
    # Conversion to useful format
    X_feat_train, y_feat_train = convert_to_news_format(X_train_all, y_train_all)
    X_feat_test, _  = convert_to_news_format(X_test, y_test)
    labels_train, labels_true = np.argmax(y_feat_train,axis = 1), np.argmax(y_test, axis = 1)
    
    
    
    # ----------------- Run SVM ------------------
    N_test, T_test, D_test = X_test.shape
    O_test = y_test.shape[-1]
    
    # Parameter selection
    C_values = [0.001, 0.1, 10]
    # kernel_values   = ["rbf", "linear", "poly"]
    kernel_values = ["linear", "rbf"]
    seeds    = generated_seeds
    
    
    # ------------------ Scores ----------------------
    scores_df = pd.DataFrame(data = np.nan, index = [], columns = ["C", "kernel", "seed",
                                                "auc_unweig", "ac_weig", "f1", "rec", "nmi"])
    i = 0    
    for C, kernel, seed in tqdm(itertools.product(C_values, kernel_values, seeds)):
        print("Current params: \nC: {} \n kernel: {} \n seed {} \n".format(C, kernel, seed))
        
        # SVM fit
        SVM = SVC(C = C, kernel = "linear", random_state = seed, probability=True)
        SVM.fit(X_feat_train, labels_train)
        
        # Make predictions and compute ode
        SVM_pred = SVM.predict_proba(X_feat_test).reshape(N_test, T_test, O_test)
        y_pred = np.mean(SVM_pred, axis = 1, keepdims = False) 
        labels_pred = compute_mode(np.argmax(SVM_pred, axis = -1))     
        
        print("Number of predicted class counts: \n", np.unique(labels_pred, return_counts = True))
        
        
        # Compute scores
        auc_unweig = roc_auc_score(labels_true, y_pred, average = "macro", multi_class = "ovr")
        auc_weig   = roc_auc_score(labels_true, y_pred, average = "weighted",  multi_class = "ovr")
        f1  = f1_score(labels_true, labels_pred, average = "macro")
        rec = recall_score(labels_true, labels_pred, average = "macro")
        nmi = normalized_mutual_info_score(labels_true, labels_pred)
        
        scores_df.loc[i, :] = [C, kernel, seed, auc_unweig, auc_weig, f1, rec, nmi]
        
        print("NMI:", nmi)
        print("auc: ", auc_unweig, auc_weig)
        i += 1
    
    
    # -------------------------- Aggreggate scores over seed ---------------
    scores_df.to_csv("SVM-scores-method2.csv", index = True, header = True)
    
    scores_df = pd.read_csv("SVM-scores-method2.csv", index_col = 0, header = 0)
    avg_results = scores_df.groupby(["C", "kernel"], as_index = True).mean()
    std_results = scores_df.groupby(["C", "kernel"], as_index = True).std()
    
    # Compute best scores
    track_metric = "f1"
    print("\n\nBest performance according to {}: \n".format(track_metric))
    print("seed-averaged mean results: \n ", avg_results.loc[avg_results[track_metric].idxmax(), :])
    print("seed-averaged std results: \n ", std_results.loc[avg_results[track_metric].idxmax(), :])
        
        
 