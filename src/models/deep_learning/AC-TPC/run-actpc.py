import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import random
import os
import argparse

from sklearn.model_selection import train_test_split

#performance metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

#user defined
import utils
from utils_log import save_logging, load_logging
from class_AC_TPC import AC_TPC, initialize_embedding


load_init_config = {"id_column": 'subject_id', "time_column": 'charttime', "feature_set_names": 'vitals', 
                    "fill_limit": None, "norm_method": None,
                    "roughly_balanced": None}
load_config = {"folder_dir": '/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/data/processed/',
    "X_y_name": ('COPD_VLS_process', 'copd_outcomes'), "time_range": (24, 72), "feature_set": 'vit-lab-sta', "include_time": None}


def f_get_minibatch(mb_size, x, y):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb   = x[idx].astype(float)    
    y_mb   = y[idx].astype(float)    

    return x_mb, y_mb

### PERFORMANCE METRICS:
def f_get_prediction_scores(y_true_, y_pred_):
    if np.sum(y_true_) == 0: #no label for running roc_auc_curves
        auroc_ = -1.
        auprc_ = -1.
    else:
        auroc_ = roc_auc_score(y_true_, y_pred_)
        auprc_ = average_precision_score(y_true_, y_pred_)
    return (auroc_, auprc_)

def f_get_weighted_prediction_scores(y_true_, y_pred_):
    if np.sum(y_true_) == 0: #no label for running roc_auc_curves
        auroc_ = -1.
        auprc_ = -1.
    else:
        auroc_weig_ = roc_auc_score(y_true_, y_pred_, average = 'weighted')
        auprc_weig_ = average_precision_score(y_true_, y_pred_, average = 'weighted')
    return (auroc_weig_, auprc_weig_)
    

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    c_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)


def init_arg():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', help = 'name to save experiment on', type = str)
    parser.add_argument('--data_name', help = 'name of dataset', type = str)
    parser.add_argument('--y_type', default = 'categorical', help = 'name to save experiment on', type = str)
    parser.add_argument('--K', default=6, help='number of maximum clusters', type=int)

    parser.add_argument('--h_dim_FC', default=30, help='number of hidden nodes in FC-layers', type=int)
    parser.add_argument('--h_dim_RNN', default=30, help='number of hidden nodes in RNN', type=int)

    parser.add_argument('--n_layer_enc', default=4, help='number of layers -- encoder', type=int)
    parser.add_argument('--n_layer_sel', default=4, help='number of layers -- selector', type=int)
    parser.add_argument('--n_layer_pre', default=4, help='number of layers -- predictor', type=int)

    parser.add_argument("--rnn_type", choices=['LSTM','GRU'], default='LSTM', type=str)
    
    parser.add_argument("--lr_rate_init", default=0.001, type=float)
    parser.add_argument("--lr_rate_clu_1", default=1e-3, type=float)
    parser.add_argument("--lr_rate_clu_2", default=1e-3, type=float)

    parser.add_argument("--itrs_init1", help='initialization for encoder-predictor', default=4000, type=int)
    parser.add_argument("--itrs_init2", help='initialization for selector and embedding', default=4000, type=int)
    parser.add_argument("--itrs_clu", default=150, type=int)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--keep_prob", help='keep probability for dropout', default=0.8, type=float)
    
    parser.add_argument('--alpha', default=0.01, type=float)
    parser.add_argument('--beta', default=0.01, type=float)
    parser.add_argument('--seed', default = 1717, type = int)

    return parser.parse_args()


if __name__ == '__main__':

    args                 = init_arg()
    
    data_name            = args.data_name
    name                 = args.name
    y_type               = args.y_type
    
    # IMPORT DATASET
    data_processor = utils.data_processor(**load_init_config)
    X, y, mask, ids, feats = data_processor.load_transform(**load_config)
    
    # Prepare folder to be saved
    model_folder = './models/ac-tpc-og/{}'.format(name)
    results_folder = './results/ac-tpc-og/{}'.format(name)
    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)    
    K                    = args.K
    h_dim_FC             = args.h_dim_FC
    h_dim_RNN            = args.h_dim_RNN
    num_layer_encoder    = args.n_layer_enc
    num_layer_selector   = args.n_layer_sel
    num_layer_predictor  = args.n_layer_pre
    rnn_type             = args.rnn_type
    x_dim = np.shape(X)[2]
    y_dim = np.shape(y)[-1]
    z_dim = h_dim_RNN * num_layer_encoder
    max_length = np.shape(X)[1]
    seed = args.seed

    y = np.repeat(np.expand_dims(y, axis = 1), repeats = X.shape[1], axis = 1)
    tr_data_x, te_data_x, tr_data_y, te_data_y = train_test_split(
        X, y, train_size=0.4, random_state=2323,
        shuffle=True, stratify=np.argmax(y, axis=-1))

    data_processor.norm_method = 'min-max'
    tr_data_x = data_processor.normalise(tr_data_x)
    te_data_x = data_processor.apply_normalise(te_data_x)

    # Separate into train and validation test
    tr_data_x, val_data_x, tr_data_y, val_data_y = train_test_split(
        tr_data_x, tr_data_y, train_size=0.6, random_state=2323,
        shuffle=True, stratify=np.argmax(tr_data_y[:, 0, :], axis=-1))
    

    alpha      = args.alpha
    beta       = args.beta

    mb_size    = args.batch_size
    
    input_dims ={
        'x_dim': x_dim,
        'y_dim': y_dim,
        'y_type': y_type,
        'max_cluster': K,
        'max_length': max_length    
    }
    
    keep_prob  = args.keep_prob
    
    network_settings ={
        'h_dim_encoder': h_dim_RNN,
        'num_layers_encoder': num_layer_encoder,
        'rnn_type': rnn_type,
        'rnn_activate_fn': tf.nn.tanh,
        'h_dim_selector': h_dim_FC,
        'num_layers_selector': num_layer_selector,
        'h_dim_predictor': h_dim_FC,
        'num_layers_predictor': num_layer_predictor,
        'keep probability': keep_prob,
        'fc_activate_fn': tf.nn.relu,
        'alpha': alpha,
        'beta': beta,
        'K': K,
        'name': name,
        'data name': data_name,
        'y_type': y_type,
        'batch size': mb_size,
        'seed': seed,
        'output': "categorical"
    }


    lr_rate    = args.lr_rate_init
    
    mb_size    = args.batch_size
    ITERATION  = args.itrs_init1
    check_step = 1000

    save_path = model_folder + 'init/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    print('Initialize Network...')

    tf.reset_default_graph()

    # Turn on xla optimization
    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto(log_device_placement = True)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement     = True
        sess = tf.Session(config=config)


        model = AC_TPC(sess, "AC_TPC", input_dims, network_settings)


        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer(), feed_dict={model.E:np.zeros([K, z_dim]).astype(float)})

        avg_loss  = 0
        for itr in range(ITERATION):
            x_mb, y_mb  = f_get_minibatch(mb_size, tr_data_x, tr_data_y)

            _, tmp_loss = model.train_mle(x_mb, y_mb, lr_rate, keep_prob)
            avg_loss   += tmp_loss/check_step

            if (itr+1)%check_step == 0:                
                tmp_y, tmp_m = model.predict_y_hats(val_data_x)

                y_pred = tmp_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
                y_true = val_data_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]

                AUROC = np.zeros([y_dim])
                AUPRC = np.zeros([y_dim])
                for y_idx in range(y_dim):
                    auroc, auprc = f_get_prediction_scores(y_true[:, y_idx], y_pred[:, y_idx])
                    AUROC[y_idx] = auroc
                    AUPRC[y_idx] = auprc

                print ("ITR {:05d}: loss_2={:.3f} | va_auroc:{:.3f}, va_auprc:{:.3f}".format(
                        itr+1, avg_loss, np.mean(AUROC), np.mean(AUPRC))
                      )        
                avg_loss = 0

        saver.save(sess, save_path + 'model')
        save_logging(network_settings, save_path + 'network_settings.txt')



        M          = int(tr_data_x.shape[0]/mb_size) #for main algorithm
        keep_prob  = args.keep_prob
        lr_rate1   = args.lr_rate_clu_1
        lr_rate2   = args.lr_rate_clu_2

        save_path = model_folder + 'train/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)



        ### LOAD INITIALIZED NETWORK
        load_path = model_folder + 'init/'

        tf.reset_default_graph()

        # Turn on xla optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        network_settings = load_logging(load_path + 'network_settings.txt')
        z_dim = network_settings['num_layers_encoder'] * network_settings['h_dim_encoder']

        model = AC_TPC(sess, "AC_TPC", input_dims, network_settings)

        saver = tf.train.Saver()
        saver.restore(sess, load_path + 'model')


        print('=============================================')
        print('===== INITIALIZING EMBEDDING & SELECTOR =====')
        # K-means over the latent encodings
        e, s_init, tmp_z = initialize_embedding(model, tr_data_x, K)
        e = np.arctanh(e)
        sess.run(model.EE.initializer, feed_dict={model.E:e}) #model.EE = tf.nn.tanh(model.E)

        # update selector wrt initial classes
        ITERATION  = args.itrs_init2
        check_step = 1000

        avg_loss_s = 0
        for itr in range(ITERATION):
            z_mb, s_mb = f_get_minibatch(mb_size, tmp_z, s_init)
            _, tmp_loss_s = model.train_selector(z_mb, s_mb, lr_rate1, k_prob=keep_prob)

            avg_loss_s += tmp_loss_s/check_step
            if (itr+1)%check_step == 0:
                print("ITR:{:04d} | Loss_s:{:.4f}".format(itr+1, avg_loss_s) )
                avg_loss_s = 0

        tmp_ybars = model.predict_yy(np.tanh(e))
        new_e     = np.copy(e)
        print('=============================================')


        print('=============================================')
        print('========== TRAINING MAIN ALGORITHM ==========')
        '''
            L1: predictive clustering loss
            L2: sample-wise entropy loss
            L3: embedding separation loss
        '''

        ITERATION     = args.itrs_clu
        check_step    = 10

        avg_loss_c_L1 = 0
        avg_loss_a_L1 = 0
        avg_loss_a_L2 = 0
        avg_loss_e_L1 = 0 
        avg_loss_e_L3 = 0

        va_avg_loss_L1 = 0
        va_avg_loss_L2 = 0
        va_avg_loss_L3 = 0
        
        main_training_loss = pd.DataFrame(data = None, columns = ['iter', 'L1c', 'L1a', 'L2', 'L1e', 'L3', 'VL1', 'VL2', 'VL3'])
        row_ = 0
        
        for itr in range(ITERATION):        
            e = np.copy(new_e)

            for _ in range(M):
                x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)

                _, tmp_loss_c_L1  = model.train_critic(x_mb, y_mb, lr_rate1, keep_prob)
                avg_loss_c_L1    += tmp_loss_c_L1/(M*check_step)

                x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)

                _, tmp_loss_a_L1, tmp_loss_a_L2 = model.train_actor(x_mb, y_mb, alpha, lr_rate2, keep_prob)
                avg_loss_a_L1 += tmp_loss_a_L1/(M*check_step)
                avg_loss_a_L2 += tmp_loss_a_L2/(M*check_step)

            for _ in range(M):
                x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)

                _, tmp_loss_e_L1, tmp_loss_e_L3 = model.train_embedding(x_mb, y_mb, beta, lr_rate1, keep_prob)
                avg_loss_e_L1  += tmp_loss_e_L1/(M*check_step)
                avg_loss_e_L3  += tmp_loss_e_L3/(M*check_step)


            x_mb, y_mb = f_get_minibatch(mb_size, val_data_x, val_data_y)
            tmp_loss_L1, tmp_loss_L2, tmp_loss_L3 = model.get_losses(x_mb, y_mb)

            va_avg_loss_L1  += tmp_loss_L1/check_step
            va_avg_loss_L2  += tmp_loss_L2/check_step
            va_avg_loss_L3  += tmp_loss_L3/check_step

            new_e = sess.run(model.embeddings)

            if (itr+1)%check_step == 0:
                tmp_ybars = model.predict_yy(new_e)
                print ("ITR {:04d}: L1_c={:.3f}  L1_a={:.3f}  L1_e={:.3f}  L2={:.3f}  L3={:.3f} || va_L1={:.3f}  va_L2={:.3f}  va_L3={:.3f}".format(
                    itr+1, avg_loss_c_L1, avg_loss_a_L1, avg_loss_e_L1, avg_loss_a_L2, avg_loss_e_L3,
                    va_avg_loss_L1, va_avg_loss_L2, va_avg_loss_L3
                ))
                
                main_training_loss[row_]  = [itr+1, avg_loss_c_L1, avg_loss_a_L1, avg_loss_e_L1, avg_loss_a_L2, avg_loss_e_L3,
                    va_avg_loss_L1, va_avg_loss_L2, va_avg_loss_L3]
                
                row_ += 1
                
                
                avg_loss_c_L1 = 0
                avg_loss_a_L1 = 0
                avg_loss_a_L2 = 0
                avg_loss_e_L1 = 0
                avg_loss_e_L3 = 0
                va_avg_loss_L1 = 0
                va_avg_loss_L2 = 0
                va_avg_loss_L3 = 0
        print('=============================================')
        main_training_loss.to_csv(results_folder + 'main_loss.csv')

        saver.save(sess, save_path + 'model')

        save_logging(network_settings, save_path + 'network_settings.txt')
        np.savez(save_path + 'embeddings.npz', e=e)


        _, tmp_pi, tmp_m = model.predict_zbars_and_pis_m2(te_data_x)

        tmp_pi = tmp_pi.reshape([-1, K])[tmp_m.reshape([-1]) == 1]

        ncol = nrow = int(np.ceil(np.sqrt(K)))
        plt.figure(figsize=[4*ncol, 2*nrow])
        for k in range(K):
            plt.subplot(ncol, nrow, k+1)
            plt.hist(tmp_pi[:, k])
        plt.suptitle("Clustering assignment probabilities")
        # plt.show()
        plt.savefig(results_folder + 'figure_clustering_assignments.png')
        plt.close()


        # check selector outputs and intialized classes
        pred_y, tmp_m = model.predict_s_sample(tr_data_x)

        pred_y = pred_y.reshape([-1, 1])[tmp_m.reshape([-1]) == 1]
        print(np.unique(pred_y))

        plt.hist(pred_y[:, 0], bins=15, color='C1', alpha=1.0)
        # plt.show()
        plt.savefig(results_folder + 'figure_clustering_hist.png')
        plt.close()


        tmp_y, tmp_m = model.predict_y_bars(te_data_x)


        y_pred = tmp_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
        y_true = te_data_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]


        AUROC = np.zeros([y_dim])
        AUPRC = np.zeros([y_dim])
        for y_idx in range(y_dim):
            auroc, auprc = f_get_prediction_scores(y_true[:, y_idx], y_pred[:, y_idx])
            AUROC[y_idx] = auroc
            AUPRC[y_idx] = auprc
        
        AUROC_weig, AUPRC_weig = f_get_weighted_prediction_scores(y_true, y_pred)
            
        print('AUROC: {}'.format(AUROC))
        print('AUPRC: {}'.format(AUPRC))
        
        if y_type == 'categorical':
            print('AUROC mean, weighted : {} / {}'.format(np.mean(AUROC), AUROC_weig))
            print('AUPRC mean, weighted: {} / {}'.format(np.mean(AUPRC), AUPRC_weig))

        pred_y, tmp_m = model.predict_s_sample(te_data_x)

        pred_y = (pred_y * tmp_m).reshape([-1, 1])
        pred_y = pred_y[(tmp_m.reshape([-1, 1]) == 1)[:, 0], 0]

        true_y = (te_data_y * np.tile(np.expand_dims(tmp_m, axis=2), [1,1,y_dim])).reshape([-1, y_dim])
        true_y = true_y[(tmp_m.reshape([-1]) == 1)]
        true_y = true_y[:, 0]
        #true_y = np.argmax(true_y, axis=1)

        tmp_nmi    = normalized_mutual_info_score(true_y, pred_y)
        tmp_ri     = adjusted_rand_score(true_y, pred_y)
        tmp_purity = purity_score(true_y, pred_y)


        print('NMI:{:.4f}, RI:{:.4f}, PURITY:{:.4f}'.format(tmp_nmi, tmp_ri, tmp_purity))

        " ------------------------------------------------------------------- "
        cluster_pred_test, m_pred_test = model.predict_s_sample(te_data_x)
        y_pred_test, _ = model.predict_y_bars(te_data_x)
        
        cluster_pred_trajs = cluster_pred_test[:, -1]
        y_pred_trajs       = y_pred_test[:, -1, :]
        
        print("Predicted cluster distribution: \n", 
              np.unique(cluster_pred_trajs, return_counts = True))
        
        embedding_string = "a{}_b{}_s{}_K{}".format(alpha, beta, seed,
                                                    K)
        
        labels_name = "ACTPC-labels.csv"
        y_name      = "ACTPC-y.csv"
        if not os.path.exists(labels_name):
            labels = pd.DataFrame(data = np.nan, index = np.arange(te_data_x.shape[0]),
                                  columns = [embedding_string])
        else:
            labels = pd.read_csv(labels_name,  index_col = 0, header = 0)
            
        if not os.path.exists(y_name):
            y_output = pd.DataFrame(data = np.nan, index = np.arange(te_data_x.shape[0]*4),
                                 columns = [embedding_string])
        else:
            y_output = pd.read_csv(y_name, header = 0, index_col = 0)
            
        labels[embedding_string] = cluster_pred_trajs
        y_output[embedding_string] = y_pred_trajs.reshape(-1)
        labels.to_csv(labels_name, index = True, header = True)
        y_output.to_csv(y_name, index = True, header = True)
        
        


    
    
    
    