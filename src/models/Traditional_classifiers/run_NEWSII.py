from typing import Union, List
import utils
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, normalized_mutual_info_score
np.set_printoptions(precision=3, suppress = True)



# ---------------------- AUXILIARY FUNCTIONS FOR NEWS II ---------------------
def score_HR(x):
    if x <= 40:
        return 3
    
    elif 40 < x and x <= 50:
        return 1
    
    elif 50 < x and x <= 90:
        return 0

    elif 90 < x and x <= 110:
        return 2

    elif 110 < x and x <= 130:
        return 2
    
    elif 130 < x:
        return 3 

def score_RR(x):
    if x <= 8:
        return 3
    
    elif 8 < x and x <= 11:
        return 1
    
    elif 11 < x and x <= 20:
        return 0
    

    elif 20 < x and x <= 24:
        return 2
    
    elif 24 < x:
        return 3
    
def score_SBP(x):
    if x <= 90:
        return 3
    
    elif 90 < x and x <= 100:
        return 2
    
    elif 100 < x and x <= 110:
        return 1
    
    elif 110 < x and x <= 219:
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
    
    elif 35.0 < x and x <= 36.0:
        return 1
    
    elif 36.0 < x and x <= 38.0:
        return 0
    
    elif 38.0 < x and x <= 39.0:
        return 1
    
    elif 39.0 < x:
        return 2
    
def score_SPO2_1(x):
    if x <= 91:
        return 3
    
    elif 91 < x and x <= 93:
        return 2
    
    elif 93 < x and x <= 95:
        return 1
    
    elif 95 < x:
        return 0
    
def score_SPO2_2(x, oxygen):
    if x <= 83:
        return 3
    
    elif 83 < x and x <= 85:
        return 2
    
    elif 85 < x and x <= 87:
        return 1
    
    elif 87 < x and x <= 92:
        return 0
    
    elif 92 < x and not oxygen:
        return 0
    
    elif 92 < x and x <= 94 and oxygen:
        return 1
    
    elif 94 < x and x <= 96 and oxygen:
        return 2
    
    elif 96 < x and oxygen:
        return 3
    
feature_scoring_dic = {
    "HR": score_HR,
    "RR": score_RR,
    "SBP": score_SBP,
    "DBP": score_DBP,
    "FIO2": score_FIO2,
    "AVPU": score_AVPU,
    "TEMP": score_TEMP,
    "SPO2": score_SPO2_2
}

class NewsII:
    def __init__(self, feature_scoring_dic):
        self.feature_score = feature_scoring_dic
        
        
    def _check_validity(self, columns):
        "Check all feature score keys are also present in columns"
        try:
            assert set(self.feature_score.keys()).issubset(set(columns))
        
        except Exception:
            raise ValueError("Columns do not match scoring keys!")
        
        
    def evaluate(self, X: np.ndarray, X_columns: Union[List, pd.Index]):
        """
        Evaluation function for data X and columns as given by X_columns.

        Parameters
        ----------
        X : data 3D array of shape N x T x D', where N is the number of 
        samples, T the maximum number of time-steps and D' the number of feats.
        D' can include more features than the ones specified in the scoring 
        function. These will be ignored in calculation
        
        X_columns : List or pandas Index of corresponding column names 
        for data X. Has size (D, ).

        Returns
        -------
        score : Corresponding NewsII score.
        """
        
        # Quick check conditions
        self._check_validity(X_columns)
        assert X.shape[-1] == len(list(X_columns))
        
        # Subset only to last time-step
        data = X[:, -1, :]                  

        
        # Initialise score array
        score = np.zeros(shape = X.shape[0])
        
        
        # Compute oxygen presence
        fio2_id = X_columns.index("FIO2")
        oxygen = np.vectorize(score_FIO2)(data[:, fio2_id]) > 0
        
        
        # Iterate through each vital sign and compute score accordingly
        for vital in self.feature_score.keys():
            
            # Obtain id and access data
            idx = X_columns.index(vital)
            vital_info = data[:, idx]
            
            # Compute score by iterating
            if vital != "SPO2":
                score_feat = np.vectorize(self.feature_score[vital])(vital_info)
                
            else:
                score_feat = np.array([score_SPO2_2(x, has_oxygen) for
                               x, has_oxygen in zip(vital_info, oxygen)])
                
            score += score_feat
        
        return score



def compute_binary(one_hot_labels: np.ndarray, target_label: int):
    """
    Compute a vector of binary labels indiciating the presence OR NOT of 
    a SPECIFIC target class within the set of labels one_hot_labels.

    Parameters
    ----------
    one_hot_labels : np.ndarray of shape N x O, where N is the number of 
    samples and O is the number of outcome classes. 
    
    target_label : int. Must be within the range [0, ..., O - 1]

    Returns
    -------
    binarised:: np.ndarray of shape (N, ) with binary entry indicating 
    one_hot_labels matches target_label.
    """
    assert target_label in range(one_hot_labels.shape[-1])
    
    binarised = (np.argmax(one_hot_labels, axis = -1) == target_label)
    
    return binarised.astype(int)


# ------------------------- Configuration loading ---------------------
# Load configuration parameters
load_init_config = {"id_column": 'subject_id', "time_column": 'charttime', 
                    "feature_set_names": 'vitals', "fill_limit": None, 
                    "norm_method": None, "roughly_balanced": None}
load_config = {"folder_dir": '/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/data/processed/',
    "X_y_name": ('COPD_VLS_process', 'copd_outcomes'), "time_range": (24, 72),
    "feature_set": 'vit-lab-sta', "include_time": None}




# ------------------------ Main function -------------------------
if __name__ == "__main__":
    
    # Load Data
    data_processor = utils.data_processor(**load_init_config)
    X, y, mask, ids, cols = data_processor.load_transform(**load_config)


    # Split data and Normalised- Data is not normalised with NEWS II
    X_train_all, X_test, y_train_all, y_test, id_train, id_test, mask_train, mask_test = train_test_split(
        X, y, ids, mask, train_size=0.4, random_state=2323,
        shuffle=True, stratify=np.argmax(y, axis=-1))

    labels_train, labels_true = np.argmax(y_train_all, axis = 1), np.argmax(y_test, axis = 1)
    feats = [col for col in cols if col not in ["subject_id", "charttime", "time_to_end"]]

    
    # ------------------- Evaluate NEWS II ---------------------
    # Run News II
    news = NewsII(feature_scoring_dic = feature_scoring_dic)
    news_score = news.evaluate(X_test, feats)
    news_score = news_score / np.max(news_score)
    reverse_news_score = 1 - news_score
    
    # Predicted labels
    # list_of_thresholds = np.arange(start = 0.0, stop = 1.0, step = 0.1)
    list_of_thresholds = [0.5]
    
    # Create targets for each class
    targets    = ["Healthy", "Death", "ICU", "Cardiac"]
    class_scores = pd.DataFrame(data = np.nan, index = ["auc", "f1", "rec"],
                                columns = targets)
    reverse_class_scores = pd.DataFrame(data = np.nan, index = ["auc", "f1", "rec"],
                                columns = targets)
    
    weights = np.sum(y_test, axis = 0) / np.sum(y_test)
    
    # Compute class AUROC and average
    for target_label in range(y_test.shape[-1]):
        
        # Binary targets for the labels
        labels_true = compute_binary(y_test, target_label)
        
        # scores per class
        auc = roc_auc_score(labels_true, news_score)
        nmi = normalized_mutual_info_score(labels_true, news_score)
        
        f1_list = [f1_score(labels_true, (news_score >= threshold)) for
               threshold in list_of_thresholds]
        rec_list = [recall_score(labels_true, (news_score >= threshold)) for
               threshold in list_of_thresholds]
        
        # Pick best threshold
        f1 = np.max(f1_list)
        rec = rec_list[f1_list.index(f1)]
        print("Regular threshold: ", list_of_thresholds[f1_list.index(f1)])
        
        # reverse score per class
        reverse_auc = roc_auc_score(labels_true, reverse_news_score)
        
        reverse_f1_list = [f1_score(labels_true, (news_score < threshold)) for
               threshold in list_of_thresholds]
        reverse_rec_list = [recall_score(labels_true, (news_score < threshold)) for
               threshold in list_of_thresholds]
        
        # Pick best threshold
        reverse_f1 = np.max(reverse_f1_list)
        reverse_rec = reverse_rec_list[reverse_f1_list.index(reverse_f1)]
        print("Regular threshold: ", 
              list_of_thresholds[reverse_f1_list.index(reverse_f1)])
        
        # Update class scores
        class_scores.iloc[:, target_label] = [auc, f1, rec]
        reverse_class_scores.iloc[:, target_label] = [reverse_auc, 
                                                      reverse_f1, reverse_rec]
    

    # Compute Unweighted Average and weighted averages
    auc_avg = class_scores.loc["auc", :].mean()
    auc_weig = np.sum(np.multiply(class_scores.loc["auc", :], weights))
    f1_avg = class_scores.loc["f1", :].mean()
    rec_avg = class_scores.loc["rec", :].mean()
    
    # Compute Unweighted Average and weighted averages for reverse scores
    rev_auc_avg = reverse_class_scores.loc["auc", :].mean()
    rev_auc_weig = np.sum(np.multiply(reverse_class_scores.loc["auc", :], weights))
    rev_f1_avg = reverse_class_scores.loc["f1", :].mean()
    rev_rec_avg = reverse_class_scores.loc["rec", :].mean()
    
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    
    # Print results
    print("Results for News II: \n")
    print("AUC average: ", auc_avg)
    print("AUC weighted: ", auc_weig)
    print("F1 avg: ", f1_avg)
    print("Recall avg: ", rec_avg)     
    
    
    # Print results
    print("Results for News II: \n")
    print("AUC average: ", rev_auc_avg)
    print("AUC weighted: ", rev_auc_weig)
    print("F1 avg: ", rev_f1_avg)
    print("Recall avg: ", rev_rec_avg)     
    
    
    
    
    
    
    
    
    
    
    