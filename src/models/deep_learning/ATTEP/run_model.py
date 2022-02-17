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
from src.models.traditional_clustering.ATTEP.model import ATTEP
from src.models.traditional_clustering.ATTEP.model_utils import get_callbacks
from src.data_processing import data_loading_utils as data_utils
from src.models import model_utils as utils

# ----------------------------------------------------------------------------------------
"Environment setting"
np.set_printoptions(precision=3, suppress=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

RESULTS_FD = "results/ATTEP/"
TRACK_FD = "experiments/ATTEP/"
    
# -----------------------------------------------------------------------------------------
"PARSING ARGUMENTS"

parser = argparse.ArgumentParser()

# General Params
parser.add_argument('--num_clusters', default=6, help='number of maximum clusters', type=int)
parser.add_argument('--latent_dim', default=128, help="dimensionality of latent space", type=int)
parser.add_argument('--seed', default=2323, type=int)

# Loss function Params
parser.add_argument('--alpha', default=0.01, type=float)
parser.add_argument('--beta', default=0.001, type=float)

# Other Network Params
parser.add_argument('--regulariser_params', default=(0.01, 0.01), type=float,
                    help="tuple of l1 and l2 regularisation.")
parser.add_argument('--dropout', default=0.6, type=float, help="default dropout value")

# Other Encoder Parameters
parser.add_argument('--custom_layer_activation', default="linear", type=str,
                    help="activation function for custom feature projection layer.")
parser.add_argument('--enc_hidden_layers', default=2, type=int, help="num encoder hidden layers.")
parser.add_argument('--enc_hidden_nodes', default=32, type=int, help="num encoder hidden nodes.")
parser.add_argument('--enc_state_fn', default="tanh", type=str, help="LSTM state update activation function.")
parser.add_argument('--enc_recurr_fn', default="sigmoid", type=str, help="LSTM gate activation function")
parser.add_argument('--enc_recurr_dropout', default=0.0, type=float, help="dropout for recurrent step.")

# Other Identifier Parameters
parser.add_argument('--iden_hidden_layers', default=2, type=int, help="num identifier hidden layers.")
parser.add_argument('--iden_hidden_nodes', default=32, type=int, help="num identifier hidden nodes.")
parser.add_argument('--iden_act_fn', default="sigmoid", type=str, help="identifier activation function.")

# Other Predictor Parameters
parser.add_argument('--pred_hidden_layers', default=2, type=int, help="num predictor hidden layers.")
parser.add_argument('--pred_hidden_nodes', default=32, type=int, help="num predictor hidden nodes.")
parser.add_argument('--pred_act_fn', default="sigmoid", type=str, help="predictor activation function.")

# Compilation Parameters
parser.add_argument('--lr', default=0.001, type=float, help="learning rate for main.py training phase.")
parser.add_argument('--clus_lr', default=0.001, type=float, help="cluster representation update rate.")
parser.add_argument('--lr_init', default=0.001, type=float, help="learning rate for initialisation.")
parser.add_argument('--optimiser_init', default='adam', type=str, help="optimisation algorithm for initialisation.")
parser.add_argument('--optimiser', default='adam', type=str, help="optimisation algorithm for main.py training phase.")

# Training Parameters
parser.add_argument('--tts_split_1', default=0.4, type=float, help="ratio between train+val sets and all data.")
parser.add_argument('--tts_split_2', default=0.6, type=float, help="ratio between train set and train + val.")
parser.add_argument('--epochs_init', default=50, type=int, help="epochs for initialisation steps.")
parser.add_argument('--epochs', default=50, type=int, help="epochs for main.py training phase.")
parser.add_argument('--batch_size_init', default=64, type=int, help="batch size for initialisation.")
parser.add_argument('--batch_size', default=64, type=int, help="batch size for main.py training phase.")

# Callback Parameters
parser.add_argument('--track_loss', default='L1', type=str, help="main.py loss to track during training.")
parser.add_argument('--early_stop', default=True, type=bool, help="whether to stop early if no loss improvement.")
parser.add_argument('--lr_scheduler', default=True, type=bool, help="whether to dynamically update learning rate.")
parser.add_argument('--tensorboard', default=True, type=bool,
                    help="whether to save model training to tensorboard logs.")
parser.add_argument('--min_delta', default=0.0001, type=float, help="delta required to observe an improvement.")
parser.add_argument('--patience', default=50, type=float, help="how many epochs to wait for improvement.")

# Data Loading Parameters
parser.add_argument('--data_name', default="HAVEN", type=str, help="which dataset to load for.")
parser.add_argument('--id_col', default="subject_id", type=str, help="identifier column for patient information.")
parser.add_argument('--time_col', default="charttime", type=str, help="identifier for temporal column.")
parser.add_argument('--feat_set', default="vitals", type=str, help="feature set to consider for analysis.")
parser.add_argument('--time_range', default=(24, 72) , type=tuple, help="Min-Max values to subset for input.")
parser.add_argument('--include_time', default=False, type=bool, help="Whether to include time difference between"
                                                                     "observations as another feature.")



# -----------------------------------------------------------------------------------------
"CONFIGURATION PARAMETERS LOADING"

# Load Params
params = parser.parse_args()
K, latent_dim, seed = params.num_clusters, params.latent_dim, params.seed
alpha, beta = params.alpha, params.beta

# Encoder
att_layer_fn = params.custom_layer_activation
enc_hidden_layers, enc_hidden_nodes = params.enc_hidden_layers, params.enc_hidden_nodes
enc_state_fn, enc_fn, enc_rec_dropout = params.enc_state_fn, params.enc_recurr_fn, params.enc_recurr_dropout

# Identifier and Predictor
iden_hidden_layers, iden_hidden_nodes, iden_fn = params.iden_hidden_layers, params.iden_hidden_nodes, params.iden_act_fn
pred_hidden_layers, pred_hidden_nodes, pred_fn = params.pred_hidden_layers, params.pred_hidden_nodes, params.pred_act_fn

# Other Network Params
regulariser_params = params.regulariser_params
dropout = params.dropout

# Compilation params
lr, clus_lr, lr_init = params.lr, params.clus_lr, params.lr_init
optimiser_init, optimiser = params.optimiser_init, params.optimiser

# Training Params
tts_split_1, tts_split_2 = params.tts_split_1, params.tts_split_2
epochs_init, epochs = params.epochs_init, params.epochs
bs_init, bs = params.batch_size_init, params.batch_size

# Callback params
track_loss = params.track_loss
early_stop, lr_scheduler = params.early_stop, params.lr_scheduler
tensorboard, min_delta, patience = params.tensorboard, params.min_delta, params.patience

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

identifier_params = {"hidden_layers": iden_hidden_layers,
                     "hidden_nodes": iden_hidden_nodes,
                     "activation_fn": iden_fn}

encoder_params = {"activation": att_layer_fn,
                  "hidden_layers": enc_hidden_layers,
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
    X_test = data_processor.apply_normalisation(X_test)

    # Separate into train and validation test
    output_dim = y_train.shape[-1]

    # ----------------------------------------------------------------------------
    "MODEL Initialisation."
    
    # ADD GPU CONFIGURATION HERE
    model = ATTEP(num_clusters=K, output_dim=output_dim, latent_dim=latent_dim, seed=seed,
           alpha=alpha, beta=beta, regulariser_params=regulariser_params, dropout=dropout,
           encoder_params=encoder_params, identifier_params=identifier_params,
           predictor_params=predictor_params, cluster_rep_lr=clus_lr, optimizer_init=optimiser_init)
    model.build(X_train.shape)

    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimiser, run_eagerly=True)

    # Initialise model
    print("-" * 20, "\n", "Initialising Model", sep = "\n")
    model.initialise_model(data=(X_train, y_train), val_data=(X_val, y_val), epochs=epochs_init,
                           learning_rate=lr_init, batch_size=bs_init)

    # ------------------------------------------------------------------------------
    "Main Training Phase"

    print("-" * 20, "/n", "STARTING MAIN TRAINING PHASE")
    
    callbacks, run_num = get_callbacks(track_loss=track_loss, early_stop=early_stop, lr_scheduler=lr_scheduler,
                      tensorboard=tensorboard, min_delta=min_delta, patience=patience)

    
        
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=bs, epochs=epochs, verbose=1,
              callbacks=callbacks)

    # ------------------------------------------------------------------------------
    "Saving Model, Results and Configuration"

    # Prepare Save Environment
    if not os.path.exists(RESULTS_FD):
        os.makedirs(RESULTS_FD)
    
    save_fd = RESULTS_FD + f"run{run_num}/"
    track_fd = TRACK_FD + f"run{run_num}/"
    assert not os.path.exists(save_fd) and os.path.exists(track_fd)
    os.makedirs(save_fd)
    
    # Obtain indices to convert results to pd DataFrame to save to csv
    cluster_names = list(range(K))
    outc_dims = outcomes
    pat_ids = id_test[:, 0, 0]    
   
    # Compute cluster related information
    cluster_phens = pd.DataFrame(model.compute_clus_phens(), index=cluster_names, columns=outc_dims)
    cluster_reps = model.get_cluster_reps()
    
    # Compute Patient related information (predictions, and cluster related assignments)
    y_pred = pd.DataFrame(model.predict(X_test), index=pat_ids, columns=outc_dims)
    pis_pred = pd.DataFrame(model.compute_pis(X_test), index=pat_ids, columns=cluster_names)
    clusters_pred = pd.Series(model.predict_clus(X_test), index=pat_ids)
    y_true = pd.DataFrame(y_test, index=pat_ids, columns=outc_dims)
    
    
    # Save Results
    cluster_phens.to_csv(save_fd + "cluster_phenotypes.csv", index=True, header=True)
    np.save(save_fd + "cluster_representations.npy", cluster_reps, allow_pickle=True)
    
    y_pred.to_csv(save_fd + "y_pred.csv", index=True, header=True)
    pis_pred.to_csv(save_fd + "pis_pred.csv", index=True, header=True)
    clusters_pred.to_csv(save_fd + "clusters_pred.csv", index=True, header=True)
    
    y_true.to_csv(save_fd + "y_true.csv", index=True, header=True)
    
    
    # Save Init Losses
    enc_pred_tracker = model._enc_pred_loss_tracker
    iden_tracker = model._iden_loss_tracker
    
    enc_pred_tracker.index.name, iden_tracker.index.name = "epoch", "epoch"
    
    enc_pred_tracker.to_csv(save_fd + "enc_pred_init_loss.csv", index=True, header=True)
    iden_tracker.to_csv(save_fd + "iden_init_loss.csv", index=True, header=True)
    
    
    
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
    # sil, sil_avg, dbi, dbi_avg, vri, vri_avg = utils.unsuper_scores(X_test, clusters_pred)
    
    print("Supervised Performance:", f"AUC: {auc:.2f}", f"f1: {f1:.2f}", f"rec: {rec:.2f}", f"pur: {pur:.2f}", sep = "\n")
    # print("Unsupervised Scores:", f"SIL: {sil:.2f}, {sil_avg:.2f}", f"DBI: {dbi:.2f}, {dbi_avg:.2f}", 
    #       f"vri: {vri:.2f} {vri_avg:.2f}", sep = "\n")
    
    cluster_dist = pd.Series(data=0, index=cluster_names)
    for clus in cluster_names:
        cluster_dist.loc[clus] = np.sum(clusters_pred==clus)
    
    print("Cluster Assignment distribution: ", cluster_dist, sep = "\n")
    print("Num Clusters with patients: {}".format(np.sum(cluster_dist != 0)))
    
    
    
    
    # ------------------------------------------------------------------------------
    """Add configuration information and scores to a master excel file."""
    run_ids = [time.time(), run_num]
    results = [auc, f1, rec, pur, sil, sil_avg, dbi, dbi_avg, vri, vri_avg]
    param_names, param_values = vars(params).keys(), vars(params).values()
    
    if not os.path.exists(TRACK_FD + "summary.csv"):
        with open(TRACK_FD + "summary.csv", "w+", newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(["time (unix)", "run"] + list(param_names) + ["auc", "f1", "recall", "purity",
                             "sil", "sil_avg", "dbi", "dbi_avg", "vri", "vri_avg"])
            
        f.close()
        
    with open(TRACK_FD + "summary.csv", "a", newline = "") as f:
        writer = csv.writer(f)
        
        new_row = [time.time(), run_num] + list(param_values) + [auc, f1, rec, pur, sil, sil_avg, dbi, dbi_avg, vri, vri_avg]
        writer.writerow(new_row)
        
        f.close()
   
    np.savez("data/ICLR-submitted/test_data.npz", 
             X_test=X_test, y_test=y_test, id_test=id_test, mask_test=mask_test,
             X_og=X_og, y_og=y_og, feats=feats)
