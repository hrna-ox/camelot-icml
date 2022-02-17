import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, normalized_mutual_info_score
from xgboost.sklearn import XGBClassifier

from tqdm import tqdm
import itertools
np.set_printoptions(precision=3, suppress = True)

import utils
from svm_all import convert_to_news_format, compute_mode
from svm_all import generated_seeds, possible_clusters



# ---------------------- Load configuration ----------------------------
load_init_config = {"id_column": 'subject_id', "time_column": 'charttime', 
                    "feature_set_names": 'vitals',  "fill_limit": None, 
                    "norm_method": None, "roughly_balanced": None}

load_config = {"folder_dir": '/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/data/processed/',
    "X_y_name": ('COPD_VLS_process', 'copd_outcomes'), "time_range": (24, 72),
    "feature_set": 'vit-lab-sta', "include_time": None}



# --------------------------- Main function ------------------------------
if __name__ == "__main__":
    
    # Load data
    data_processor = utils.data_processor(**load_init_config)
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
    
    
    
    # ----------------- Run XGBOOST ------------------
    N_test, T_test, D_test = X_test.shape
    O_test = y_test.shape[-1]
    
    
    estimator_range = [50, 100, 200, 500]
    depth_range     = [1, 3, 5, 10]
    gamma_range = [0.0, 0.1, 0.2, 0.6]
    lr_range = [0.001, 0.01, 0.1, 0.5]
    seeds    = generated_seeds
    
    
    
    # --------------- Scores ---------------
    xgb_scores_df = pd.DataFrame(data = np.nan, index = [], 
                 columns = ["n_estimator", "depth", "learning_rate", "gamma",
                        "seed", "auc_unweig", "ac_weig", "f1", "rec", "nmi"])
    i = 0

    for n_est_, depth_, gamma_, lr_, seed in tqdm(
            itertools.product(estimator_range, depth_range, 
                              gamma_range, lr_range, seeds)):
        print("""Current params: \nn_est: {} \ndepth: {} \ngamma {} \n
              learning_Rate: {} \nseed: {} \n""".format(n_est_, depth_,
              gamma_, lr_, seed))
               
    
        # Fit data
        XGB = XGBClassifier(n_estimator=n_est_, depth = depth_, random_state = seed,
                            objective="softmax", gamma = gamma_,
                            scale_pos_weight = 1, learning_rate = lr_,
                            use_label_encoder=False, verbosity = 0)
        XGB.fit(X_feat_train, labels_train)
        
        
        # Make probability predictions
        XGB_pred = XGB.predict_proba(X_feat_test).reshape(N_test, D_test, O_test)
        y_pred = np.mean(XGB_pred, axis = 1, keepdims = False)
        labels_pred = compute_mode(np.argmax(XGB_pred, axis = -1))
        
        print("Number of predicted class counts: \n", np.unique(labels_pred, return_counts = True))
        
        
        # Compute scores
        auc_unweig = roc_auc_score(labels_true, y_pred, average = "macro", multi_class = "ovr")
        auc_weig   = roc_auc_score(labels_true, y_pred, average = "weighted",  multi_class = "ovr")
        f1  = f1_score(labels_true, labels_pred, average = "macro")
        rec = recall_score(labels_true, labels_pred, average = "macro")
        nmi = normalized_mutual_info_score(labels_true, labels_pred)
        
        # Update scores table
        xgb_scores_df.loc[i, :] = [n_est_, depth_, lr_, gamma_, seed, auc_unweig, auc_weig, f1, rec, nmi]
        i += 1
        
        print("NMI:", nmi)
    
    
    # ------------------ Aggreggate across seeds ----------------------------
    xgb_scores_df.to_csv("XGB-scores.csv", index = True, header = True)
    xgb_avg_results = xgb_scores_df.groupby(["n_estimator", "depth","learning_rate", "gamma"], as_index = False).mean()
    xgb_std_results = xgb_scores_df.groupby(["n_estimator", "depth","learning_rate", "gamma"], as_index = False).std()
    
    
    track_metric = "f1"
    print("\n\nBest performance according to {}: \n".format(track_metric))
    print("seed-averaged mean results: \n ", xgb_avg_results.loc[xgb_avg_results[track_metric].idxmax(), :])
    print("seed-averaged std results: \n ", xgb_std_results.loc[xgb_avg_results[track_metric].idxmax(), :])
        
    
        
        
    
    