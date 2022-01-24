"""
Main model training.
"""
import os, argparse, sys
from pathlib import Path
import json, csv, time

import numpy as np
import pandas as pd
from tensorflow.keras import optimizers
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Add Directory path if does not exist
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
from src.models.main.model import CAMELOT
from src.models.main.model_utils import get_callbacks
from src.training import data_loader as data_utils
from src.training import utils as utils

# ----------------------------------------------------------------------------------------
"Environment setting"
np.set_printoptions(precision=3, suppress=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

RESULTS_FD = "results/main/"
TRACK_FD = "experiments/main/"
OUTCOMES = ["Healthy", "Death", "ICU", "Cardiac"]
    
# %% Configuration parameters

parser = argparse.ArgumentParser()

# General Params
parser.add_argument('--data_info', default=data_utils.data_loader(), type=dict, help="collection of input data info")
parser.add_argument('--num_clusters', default=6, help='number of maximum clusters', type=int)
parser.add_argument('--latent_dim', default=128, help="dimensionality of latent space", type=int)
parser.add_argument('--seed', default=2323, type=int)

# Loss function Params
parser.add_argument('--alpha', default=0.01, type=float)
parser.add_argument('--beta', default=0.1, type=float)

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
parser.add_argument('--lr', default=0.001, type=float, help="learning rate for main training phase.")
parser.add_argument('--clus_lr', default=0.001, type=float, help="cluster representation update rate.")
parser.add_argument('--lr_init', default=0.001, type=float, help="learning rate for initialisation.")
parser.add_argument('--optimiser_init', default='adam', type=str, help="optimisation algorithm for initialisation.")
parser.add_argument('--optimiser', default='adam', type=str, help="optimisation algorithm for main training phase.")

# Training Parameters
parser.add_argument('--epochs_init', default=100, type=int, help="epochs for initialisation steps.")
parser.add_argument('--epochs', default=300, type=int, help="epochs for main training phase.")
parser.add_argument('--batch_size_init', default=64, type=int, help="batch size for initialisation.")
parser.add_argument('--batch_size', default=64, type=int, help="batch size for main training phase.")

# Callback Parameters
parser.add_argument('--track_loss', default='L1', type=str, help="main loss to track during training.")
parser.add_argument('--early_stop', default=True, type=bool, help="whether to stop early if no loss improvement.")
parser.add_argument('--lr_scheduler', default=True, type=bool, help="whether to dynamically update learning rate.")
parser.add_argument('--tensorboard', default=True, type=bool,
                    help="whether to save model training to tensorboard logs.")
parser.add_argument('--min_delta', default=0.0001, type=float, help="delta required to observe an improvement.")
parser.add_argument('--patience', default=500, type=float, help="how many epochs to wait for improvement.")



# -----------------------------------------------------------------------------------------
"CONFIGURATION PARAMETERS LOADING"

# Load Params
params = parser.parse_args()
data_info = params.data_info
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
epochs_init, epochs = params.epochs_init, params.epochs
bs_init, bs = params.batch_size_init, params.batch_size

# Callback params
track_loss = params.track_loss
early_stop, lr_scheduler = params.early_stop, params.lr_scheduler
tensorboard, min_delta, patience = params.tensorboard, params.min_delta, params.patience




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


# ----------------------- Main Processing --------------------------------------
if __name__ == "__main__":

    # ----------------------------------------------------------------------------

    X_train, X_val, X_test = data_info["X"]
    y_train, y_val, y_test = data_info["y"]
    id_train, id_val, id_test = data_info["id"]
    mask_train, mask_val, mask_test = data_info["mask"]
    output_dim = data_info["output_dim"]
    norm_min, norm_max = data_info["norm_min"], data_info["norm_max"]
    data_loading_config = data_info["data_load_config"]

    # ----------------------------------------------------------------------------
    "MODEL Initialisation."
    
    # ADD GPU CONFIGURATION HERE
    model = CAMELOT(num_clusters=K, output_dim=output_dim, latent_dim=latent_dim, seed=seed,
           alpha=alpha, beta=beta, regulariser_params=regulariser_params, dropout=dropout,
           encoder_params=encoder_params, identifier_params=identifier_params,
           predictor_params=predictor_params, cluster_rep_lr=clus_lr, optimizer_init=optimiser_init)
    model.build(X_train.shape)

    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimiser, run_eagerly=True)

    # Initialise model
    epochs_init = 100
    print("-" * 20, "\n", "Initialising Model", sep = "\n")
    model.initialise_model(data=(X_train, y_train), val_data=(X_val, y_val), epochs=epochs_init,
                           learning_rate=lr_init, batch_size=bs_init)

    # ------------------------------------------------------------------------------
    "Main Training Phase"

    print("-" * 20, "\n", "STARTING MAIN TRAINING PHASE")
    
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
    
    print(f"Saving Under Folder 'main' with Run number {run_num}")
    
    
    # Compute test results and save 
    # Obtain indices to convert results to pd DataFrame to save to csv
    cluster_names = list(range(K))
    outc_dims = OUTCOMES
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
    
    enc_pred_tracker.to_csv(track_fd + "enc_pred_init_loss.csv", index=True, header=True)
    iden_tracker.to_csv(track_fd + "iden_init_loss.csv", index=True, header=True)
    
    
    # save parasm
    # Save Model Configuration on both results and experiments
    # with open(save_fd + "config", "w+") as f:
    #     json.dump(vars(params), f)
    #     f.close()
        
    # with open(track_fd + "config", "w+") as f:
    #     json.dump(vars(params), f)
    #     f.close()
        
    


    # Evaluate results
    # auc, f1, rec = os.system("""python run_model.py --K {} --latent_dim {} --seed {}""".format(
    #     K, latent_dim, seed))
    
    
    # ------------------------------------------------------------------------------
    """Finally, print some basic statistics for analysing this run"""
    y_true = y_test
    y_pred = y_pred.values
    # auc, f1, rec, pur = utils.super_scores(y_true, y_pred)
    # sil, sil_avg, dbi, dbi_avg, vri, vri_avg = utils.unsuper_scores(X_test, clusters_pred)
    
    # print("Supervised Performance:", f"AUC: {auc:.2f}", f"f1: {f1:.2f}", f"rec: {rec:.2f}", f"pur: {pur:.2f}", sep = "\n")
    # print("Unsupervised Scores:", f"SIL: {sil:.2f}, {sil_avg:.2f}", f"DBI: {dbi:.2f}, {dbi_avg:.2f}", 
    #       f"vri: {vri:.2f} {vri_avg:.2f}", sep = "\n")
    
    cluster_dist = pd.Series(data=0, index=cluster_names)
    for clus in cluster_names:
        cluster_dist.loc[clus] = np.sum(clusters_pred==clus)
    
    print("Cluster Assignment distribution: ", cluster_dist, sep = "\n")
    print("Num Clusters with patients: {}".format(np.sum(cluster_dist != 0)))
    
    
    outcome_df = pd.read_csv("data/HAVEN/processed/copd_outcomes.csv", index_col = 0)
    outcomes = outcome_df.loc[clusters_pred.index, :]
    
    for clus in cluster_names:
        print("Outcome distribution in Cluster {}".format(clus))

        outcome_distribution = outcomes[clusters_pred==clus].sum(axis = 0)
        
        print(outcome_distribution)
    
    alpha_sc, beta_sc, gamma_sc = model.compute_attention_rnn_encoder_scores(X_test)
    np.savez(save_fd + "attention-all.npz", alpha=alpha_sc, beta=beta_sc, gamma=gamma_sc)
    
    
    # ------------------------------------------------------------------------------
    # """Add configuration information and scores to a master excel file."""
    # run_ids = [time.time(), run_num]
    # results = [auc, f1, rec, pur, sil, sil_avg, dbi, dbi_avg, vri, vri_avg]
    # param_names, param_values = vars(params).keys(), vars(params).values()
    
    # if not os.path.exists(TRACK_FD + "summary.csv"):
    #     with open(TRACK_FD + "summary.csv", "w+", newline = "") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["time (unix)", "run"] + list(param_names) + ["auc", "f1", "recall", "purity",
    #                          "sil", "sil_avg", "dbi", "dbi_avg", "vri", "vri_avg"])
            
    #     f.close()
        
    # with open(TRACK_FD + "summary.csv", "a", newline = "") as f:
    #     writer = csv.writer(f)
        
    #     new_row = [time.time(), run_num] + list(param_values) + [auc, f1, rec, pur, sil, sil_avg, dbi, dbi_avg, vri, vri_avg]
    #     writer.writerow(new_row)
        
    #     f.close()
       
    # np.savez("data/ICLR-submitted/test_data.npz", 
    #          X_test=X_test, y_test=y_test, id_test=id_test, 
    #          mask_test=mask_test, feats=data_info["feats"])


# %% Compute attention as required
scores = np.load("data/ICLR-submitted/main/Inside-modelpatient/a0.01_b0.001_g0.01_K6_l128_lr0.001_ep50_s1717/attention-all.npz")["arr_0"]
clusters_pred = pd.read_csv("data/ICLR-submitted/main/Inside-modelpatient/a0.01_b0.001_g0.01_K6_l128_lr0.001_ep50_s1717/clusters_df.csv", index_col=0)["a0.01_b0.001_g0.01_K6_l128_lr0.001_ep50_s1717"]


# scores_1 = model.compute_attention_rnn_encoder_scores(X_test)
# scores_2 = model.compute_attention_rnn_encoder_scores(X_test)

# alpha_1, beta_1, gamma_1 = scores_1
# alpha_2, beta_2, gamma_2 = scores_2

outcome_df = pd.read_csv("data/HAVEN/processed/copd_outcomes.csv", index_col = 0)
outc_test = outcome_df.loc[pat_ids, :]
num_outcs = outc_test.shape[-1]

# Compute attention for each cluster and outcome group in each cluster
new_shape = (num_outcs, K) + (12, 20)
global_attention_1_1 = np.zeros(shape=new_shape)
num_pats = np.zeros(shape = (num_outcs, K, 1, 1))

for i_clus in range(K):
    for i_outcs in range(num_outcs):
        
        # Compute patients within cluster and corresponding outcome
        has_outc_ = outc_test.iloc[:, i_outcs]
        is_in_clus_ = (clusters_pred == i_clus)
        
        target_samples = np.logical_and(has_outc_, is_in_clus_)
        
        if np.sum(target_samples) > 0:
            num_pats[i_outcs, i_clus, 0, 0] = np.sum(target_samples)
            
            # Compute averages over target samples
            avg_1_1 = np.mean(scores[target_samples, :, :, i_clus], axis = 0)
            
            
            # Update
            global_attention_1_1[i_outcs, i_clus, :, :] = avg_1_1

# Plot
import matplotlib.pyplot as plt
import seaborn as sns

def make_attention_plots(attention_array):
    
    fig, ax = plt.subplots(nrows=num_outcs, ncols=K, sharex=True, sharey=True)
    
    for i_clus in range(K):
        for i_outcs in range(num_outcs):
            sns.heatmap(attention_array[i_outcs, i_clus, :, :],
                        ax = ax[i_outcs, i_clus], annot=False, cmap="OrRd",
                        vmax=0.03, vmin=0.0)
    plt.suptitle("Per cluster per outcome attention map")
    plt.show()
    
    
    fig_2, ax_2 = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
    ax_2 = ax_2.reshape(-1)
    norm_num_pats = num_pats / np.sum(num_pats, axis = 0, keepdims=True)
    
    for i_clus in range(K):
        sns.heatmap(
            np.sum(
                np.multiply(
                    attention_array[:, i_clus, :, :], 
                    norm_num_pats[:, i_clus, :, :]
                ), 
            axis = 0, keepdims = False
            ), 
        ax = ax_2[i_clus], annot=False, cmap="OrRd",
                        vmax=0.03, vmin=0.0
        )
    plt.suptitle("Cluster weighted sum (outcome-normalised)")
    plt.show()
    
    
    
    fig_3, ax_3 = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
    ax_3 = ax_3.reshape(-1)
    for i_clus in range(K):
        sns.heatmap(np.mean(attention_array[:, i_clus, :, :], axis =0),
                    ax = ax_3[i_clus], annot=False, cmap="OrRd",
                        vmax=np.max(attention_array), vmin=0.0)
    plt.suptitle("Cluster averaging")
    plt.show()
    
    
make_attention_plots(global_attention_1_1)

    
    
    
    








