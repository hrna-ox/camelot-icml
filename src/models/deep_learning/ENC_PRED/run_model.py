"""
Main model training.
"""
import os, argparse, sys
from pathlib import Path
import json, csv, time

import numpy as np
import pandas as pd
from tensorflow.keras import optimizers

from sklearn.model_selection import train_test_split

# Add Directory path if does not exist
sys.path.append(str(Path(os.path.abspath(__file__)).parents[4]))
from src.models.benchmarks.ENC_PRED.model import LSTMEP
from src.data_processing import data_loading_utils as data_utils
from src.models import model_utils as utils

# ----------------------------------------------------------------------------------------
"Environment setting"
np.set_printoptions(precision=3, suppress=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

RESULTS_FD = "results/evaluate.py/"
TRACK_FD = "experiments/evaluate.py/"
    
# -----------------------------------------------------------------------------------------
"PARSING ARGUMENTS"

parser = argparse.ArgumentParser()

# General Params
parser.add_argument('--num_clusters', default=6, help='number of maximum clusters', type=int)
parser.add_argument('--latent_dim', default=128, help="dimensionality of latent space", type=int)
parser.add_argument('--seed', default=2323, type=int)


# Other Network Params
parser.add_argument('--regulariser_params', default=(0.01, 0.01), type=float,
                    help="tuple of l1 and l2 regularisation.")
parser.add_argument('--dropout', default=0.6, type=float, help="default dropout value")

# Other Encoder Parameters
parser.add_argument('--enc_hidden_layers', default=4, type=int, help="num encoder hidden layers.")
parser.add_argument('--enc_hidden_nodes', default=32, type=int, help="num encoder hidden nodes.")
parser.add_argument('--enc_state_fn', default="tanh", type=str, help="LSTM state update activation function.")
parser.add_argument('--enc_recurr_fn', default="sigmoid", type=str, help="LSTM gate activation function")
parser.add_argument('--enc_recurr_dropout', default=0.0, type=float, help="dropout for recurrent step.")


# Other Predictor Parameters
parser.add_argument('--pred_hidden_layers', default=4, type=int, help="num predictor hidden layers.")
parser.add_argument('--pred_hidden_nodes', default=32, type=int, help="num predictor hidden nodes.")
parser.add_argument('--pred_act_fn', default="sigmoid", type=str, help="predictor activation function.")

# Compilation Parameters
parser.add_argument('--lr', default=0.001, type=float, help="learning rate for evaluate.py training phase.")
parser.add_argument('--optimiser', default='adam', type=str, help="optimisation algorithm for evaluate.py training phase.")

# Training Parameters
parser.add_argument('--tts_split_1', default=0.4, type=float, help="ratio between train+val sets and all data.")
parser.add_argument('--tts_split_2', default=0.6, type=float, help="ratio between train set and train + val.")
parser.add_argument('--epochs', default=100, type=int, help="epochs for evaluate.py training phase.")
parser.add_argument('--batch_size', default=64, type=int, help="batch size for evaluate.py training phase.")


# Data Loading Parameters
parser.add_argument('--data_name', default="HAVEN", type=str, help="which dataset to load for.")
parser.add_argument('--id_col', default="subject_id", type=str, help="identifier column for patient information.")
parser.add_argument('--time_col', default="charttime", type=str, help="identifier for temporal column.")
parser.add_argument('--feat_set', default="all", type=str, help="feature set to consider for analysis.")
parser.add_argument('--time_range', default=(24, 72) , type=tuple, help="Min-Max values to subset for input.")
parser.add_argument('--include_time', default=False, type=bool, help="Whether to include time difference between"
                                                                     "observations as another feature.")



# -----------------------------------------------------------------------------------------
"CONFIGURATION PARAMETERS LOADING"

# Load Params
params = parser.parse_args()
latent_dim, seed = params.latent_dim, params.seed

# Encoder
enc_hidden_layers, enc_hidden_nodes = params.enc_hidden_layers, params.enc_hidden_nodes
enc_state_fn, enc_fn, enc_rec_dropout = params.enc_state_fn, params.enc_recurr_fn, params.enc_recurr_dropout

# Identifier and Predictor
pred_hidden_layers, pred_hidden_nodes, pred_fn = params.pred_hidden_layers, params.pred_hidden_nodes, params.pred_act_fn

# Other Network Params
regulariser_params = params.regulariser_params
dropout = params.dropout

# Compilation params
lr = params.lr
optimiser = params.optimiser

# Training Params
tts_split_1, tts_split_2 = params.tts_split_1, params.tts_split_2
epochs = params.epochs
bs = params.batch_size


# Data Loading Params
data_name = params.data_name
id_col, time_col, feat_set = params.id_col, params.time_col, params.feat_set
time_range, include_time = params.time_range, params.include_time

# --------------------------------------------------------------------------------
"Define input parameters from configuration."

# Network Parameters
predictor_params = {"hidden_layers": pred_hidden_layers,
                    "hidden_nodes": pred_hidden_nodes,
                    "activation_fn": pred_fn}


encoder_params = {"hidden_layers": enc_hidden_layers,
                  "hidden_nodes": enc_hidden_nodes,
                  "state_fn": enc_state_fn,
                  "recurrent_fn": enc_fn,
                  "recurrent_dropout": enc_rec_dropout}

# Data Loading Parameters
data_load_config = {"data_name": data_name, "id_column": id_col, "time_column": time_col, "feat_set": feat_set,
                    "time_range": time_range, "include_time": include_time}


# ----------------------- Main Processing --------------------------------------
if __name__ == "__main__":

    # ----------------------------------------------------------------------------
    "Data LOADING and PROCESSING"
    data_processor = data_utils.DataProcessor(**data_load_config)
    x, y, mask, ids, feats, outcomes, X_og, y_og = data_processor.load_transform()
    print(f"{data_name} data successfully loaded.")

    # Separate into train, val and test data
    X_train, X_test, y_train, y_test, id_train, id_test, mask_train, mask_test = train_test_split(
        x, y, ids, mask, train_size=tts_split_1, random_state=seed,
        shuffle=True, stratify=np.argmax(y, axis=-1))
    
    X_train, X_val, y_train, y_val, id_train, id_val, mask_train, mask_val = train_test_split(
        X_train, y_train, id_train, mask_train, train_size=tts_split_2, random_state=seed,
        shuffle=True, stratify=np.argmax(y_train, axis=-1))

    
    # Normalise and do the same for Validation and Test sets.
    X_train = data_processor.normalise(X_train)
    X_val = data_processor.apply_normalisation(X_val)
    X_test = data_processor.apply_normalisation(X_test)

    # Separate into train and validation test
    output_dim = y_train.shape[-1]

    # ----------------------------------------------------------------------------
    "MODEL Training."
    
    # ADD GPU CONFIGURATION HERE
    model = LSTMEP(output_dim=output_dim, latent_dim=latent_dim, seed=seed,
                   regulariser_params=regulariser_params, dropout=dropout,
                   encoder_params=encoder_params, predictor_params=predictor_params)
    model.build(X_train.shape)

    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimiser, loss="categorical_crossentropy", run_eagerly=True)

    "Main Training Phase"
    print("-" * 20, "/n", "STARTING MAIN TRAINING PHASE")

    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=bs, epochs=epochs, verbose=1)

    # ------------------------------------------------------------------------------
    "Saving Model, Results and Configuration"

    # Prepare Save Environment
    if not os.path.exists(RESULTS_FD):
        os.makedirs(RESULTS_FD)
    
    run_num = 1
    while os.path.exists(TRACK_FD + f"run{run_num}/"):
        run_num += 1
        
    save_fd = RESULTS_FD + f"run{run_num}/"
    track_fd = TRACK_FD + f"run{run_num}/"
    os.makedirs(save_fd)
    os.makedirs(track_fd)
    
    # Obtain indices to convert results to pd DataFrame to save to csv
    outc_dims = outcomes
    pat_ids = id_test[:, 0, 0]    
   
    # Compute Patient related information (predictions, and cluster related assignments)
    y_pred = pd.DataFrame(model.predict(X_test), index=pat_ids, columns=outc_dims)
    y_true = pd.DataFrame(y_test, index=pat_ids, columns=outc_dims)
    
    
    # Save Results
    y_pred.to_csv(save_fd + "y_pred.csv", index=True, header=True)
    y_true.to_csv(save_fd + "y_true.csv", index=True, header=True)
    
    # Save Model Configuration on both results and experiments
    with open(save_fd + "config", "w+") as f:
        json.dump(vars(params), f)
        f.close()
        
    with open(track_fd + "config", "w+") as f:
        json.dump(vars(params), f)
        f.close()
        
    

    
    # ------------------------------------------------------------------------------
    """Finally, print some basic statistics for analysing this run"""
    y_true = y_test
    y_pred = y_pred.values
    auc, f1, rec, pur = utils.super_scores(y_true, y_pred)
    
    print("Supervised Performance:", f"AUC: {auc:.2f}", f"f1: {f1:.2f}", f"rec: {rec:.2f}", f"pur: {pur:.2f}", sep = "\n")
    
    
    
    # ------------------------------------------------------------------------------
    """Add configuration information and scores to a master excel file."""
    run_ids = [time.time(), run_num]
    results = [auc, f1, rec, pur]
    param_names, param_values = vars(params).keys(), vars(params).values()
    
    if not os.path.exists(TRACK_FD + "summary.csv"):
        with open(TRACK_FD + "summary.csv", "w+", newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(["time (unix)", "run"] + list(param_names) + ["auc", "f1", "recall", "purity"])
            
        f.close()
        
    with open(TRACK_FD + "summary.csv", "a", newline = "") as f:
        writer = csv.writer(f)
        
        new_row = [time.time(), run_num] + list(param_values) + [auc, f1, rec, pur]
        writer.writerow(new_row)
        
        f.close()
   
    np.savez("data/ICLR-submitted/test_data.npz", 
             X_test=X_test, y_test=y_test, id_test=id_test, mask_test=mask_test,
             X_og=X_og, y_og=y_og, feats=feats)
