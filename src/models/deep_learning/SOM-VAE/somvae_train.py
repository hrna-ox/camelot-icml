"""
Script to train the SOM-VAE model as described in https://arxiv.org/abs/1806.02199
Copyright (c) 2018
Author: Vincent Fortuin
Institution: Biomedical Informatics group, ETH Zurich
License: MIT License

If you want to optimize the hyperparameters using labwatch, you have to install labwatch and SMAC
and comment in the commented out lines.
"""
import uuid
import shutil
from datetime import date

import tensorflow as tf
from tqdm import tqdm
import sacred
from sklearn.model_selection import train_test_split

from labwatch import hyperparameters as hyper

from somvae_model import SOMVAE
from utils_model import *
import utils
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

load_init_config = {"id_column": 'subject_id', "time_column": 'charttime', "feature_set_names": 'vitals', 
                    "fill_limit": None, "norm_method": None,
                    "roughly_balanced": None}
load_config = {"folder_dir": '/home/ds.ccrg.kadooriecentre.org/henrique.aguiar/Desktop/COPD/data/processed/',
    "X_y_name": ('COPD_VLS_process', 'copd_outcomes'), "time_range": (24, 72), "feature_set": 'vit-lab-sta', "include_time": None}


generated_seeds = [1001, 1012,1134, 2475, 6138, 7415, 1663, 7205, 9253, 1782]
possible_clusters = range(2, 50)
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

X_train = tf.expand_dims(X_train, -1)
X_val   = tf.expand_dims(X_val, -1)
X_test  = tf.expand_dims(X_test, -1)
y_train = tf.expand_dims(y_train, -1)
y_val   = tf.expand_dims(y_val, -1)
y_test  = tf.expand_dims(y_test, -1)

ex = sacred.Experiment("hyperopt", interactive = False)
ex.observers.append(sacred.observers.FileStorageObserver.create("../sacred_runs"))
ex.captured_out_filter = sacred.utils.apply_backspaces_and_linefeeds

# ex.observers.append(sacred.observers.MongoObserver.create(db_name="somvae_hyperopt"))

# assistant = LabAssistant(ex, "somvae_hyperopt", optimizer=SMAC, url="localhost:{}".format(db_port))


@ex.config
def ex_config():
    """Sacred configuration for the experiment.
    
    Params:
        num_epochs (int): Number of training epochs.
        patience (int): Patience for the early stopping.
        batch_size (int): Batch size for the training.
        latent_dim (int): Dimensionality of the SOM-VAE's latent space.
        som_dim (list): Dimensionality of the self-organizing map.
        learning_rate (float): Learning rate for the optimization.
        alpha (float): Weight for the commitment loss.
        beta (float): Weight for the SOM loss.
        gamma (float): Weight for the transition probability loss.
        tau (float): Weight for the smoothness loss.
        decay_factor (float): Factor for the learning rate decay.
        name (string): Name of the experiment.
        ex_name (string): Unique name of this particular run.
        logdir (path): Directory for the experiment logs.
        modelpath (path): Path for the model checkpoints.
        interactive (bool): Indicator if there should be an interactive progress bar for the training.
        data_set (string): Data set for the training.
        save_model (bool): Indicator if the model checkpoints should be kept after training and evaluation.
        time_series (bool): Indicator if the model should be trained on linearly interpolated
            MNIST time series.
        mnist (bool): Indicator if the model is trained on MNIST-like data.
    """
    num_epochs = 100
    patience = 100
    batch_size = 64
    latent_dim = 32
    som_dim = [4,4]
    learning_rate = 0.0005
    alpha = 1.0
    beta = 0.9
    gamma = 1.8
    tau = 1.4
    decay_factor = 0.9
    name = ex.get_experiment_info()["name"]
    ex_name = "{}_{}_{}-{}_{}_{}".format(name, latent_dim, som_dim[0], som_dim[1], str(date.today()), uuid.uuid4().hex[:5])
    logdir = "../logs/{}".format(ex_name)
    modelpath = "./models/{}/{}.ckpt".format(ex_name, ex_name)
    interactive = True
    save_model = False
    time_series = False
    seed = 1001


# @assistant.search_space
def search_space():
    num_epochs = 20
    patience = 20
    batch_size = 32
    latent_dim = hyper.UniformInt(lower=64, upper=256, log_scale=True)
    som_dim = [8,8]
    learning_rate = hyper.UniformFloat(lower=0.0001, upper=0.01, log_scale=True)
    alpha = hyper.UniformFloat(lower=0., upper=2.)
    beta = hyper.UniformFloat(lower=0., upper=2.)
    gamma = hyper.UniformFloat(lower=0., upper=2.)
    tau = hyper.UniformFloat(lower=0., upper=2.)
    decay_factor = hyper.UniformFloat(lower=0.8, upper=1.)
    interactive = True

@ex.capture
def get_data_generator(X, y, batch_size, seed = 1001):
    
    "Create generator"
    def batch_generator():
        "Generator for data"
        trajs  = X.copy()
        labels = y.copy()
        
        while True:
            rng   = np.random.default_rng(seed)
            indices = rng.permutation(np.arange(len(trajs)))
            trajs = trajs[indices]
            labels = labels[indices]
            
            for batch_count in range(X.shape[0]//batch_size):
                
                data_batch = X[batch_count*batch_size:(batch_count+1)*batch_size, :, :]
                
                if batch_count + 1 == X.shape[0] //batch_size:    
                    data_batch = X[(batch_count+1) * batch_size:, :, :]   
                    
                yield data_batch
        
    return batch_generator()


@ex.capture
def train_model(model, x, lr_val, num_epochs, patience, batch_size, logdir,
        modelpath, learning_rate, interactive, data, seed):
    """Trains the SOM-VAE model.
    
    Args:
        model (SOM-VAE): SOM-VAE model to train.
        x (tf.Tensor): Input tensor or placeholder.
        lr_val (tf.Tensor): Placeholder for the learning rate value.
        num_epochs (int): Number of epochs to train.
        patience (int): Patience parameter for the early stopping.
        batch_size (int): Batch size for the training generator.
        logdir (path): Directory for saving the logs.
        modelpath (path): Path for saving the model checkpoints.
        learning_rate (float): Learning rate for the optimization.
        interactive (bool): Indicator if we want to have an interactive
            progress bar for training.
    """
    X_train, y_train, X_val, y_val = data
    train_gen = get_data_generator(X_train, y_train, batch_size, seed)
    val_gen   = get_data_generator(X_val, y_val, batch_size, seed)

    num_batches = X_train.shape[0]//batch_size

    saver = tf.train.Saver()
    summaries = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        patience_count = 0
        test_losses = []
#        with LogFileWriter(ex):
            # train_writer = tf.summary.FileWriter(logdir+"/train", sess.graph)
            # test_writer  = tf.summary.FileWriter(logdir+"/test", sess.graph)
        print("Training...")
        train_step_SOMVAE, train_step_prob = model.optimize
        try:
            for epoch in tqdm(range(num_epochs)):
                batch_val = next(val_gen)
                test_loss, summary = sess.run([model.loss, summaries], feed_dict={x: batch_val})
                test_losses.append(test_loss)
                # test_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                if test_losses[-1] == min(test_losses):
                    saver.save(sess, modelpath, global_step=epoch)
                    patience_count = 0
                else:
                    patience_count += 1
                if patience_count >= patience:
                    break
                for i in range(num_batches):
                    batch_data = next(train_gen)
                    if i%100 == 0:
                        train_loss, summary = sess.run([model.loss, summaries], feed_dict={x: batch_data})
                        # train_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                    train_step_SOMVAE.run(feed_dict={x: batch_data, lr_val:learning_rate})
                    train_step_prob.run(feed_dict={x: batch_data, lr_val:learning_rate*100})

        except KeyboardInterrupt:
            pass
        finally:
            saver.save(sess, modelpath)


@ex.capture
def evaluate_model(model, x, modelpath, batch_size, data, embedding_string):
    """Evaluates the performance of the trained model in terms of normalized
    mutual information, purity and mean squared error.
    
    Args:
        model (SOM-VAE): Trained SOM-VAE model to main.py.
        x (tf.Tensor): Input tensor or placeholder.
        modelpath (path): Path from which to restore the model.
        batch_size (int): Batch size for the evaluation.
        
    Returns:
        dict: Dictionary of evaluation results (NMI, Purity, MSE).
    """
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.)

    X_val, y_val = data
    num_batches = len(X_val)//batch_size

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, modelpath)

        test_k_all = []
        test_rec_all = []
        test_mse_all = []
        print("Evaluation...")
        for i in range(num_batches):
            batch_data = X_val[i*batch_size:(i+1)*batch_size, :, :]
            test_k_all.extend(sess.run(model.k, feed_dict={x: batch_data}))
            test_rec = sess.run(model.reconstruction_q, feed_dict={x: batch_data})
            test_rec_all.extend(test_rec)

        print("Evaluation completed.")
        #test_nmi = compute_NMI(test_k_all, y_val[:len(test_k_all), :, :])
        # test_purity = compute_purity(test_k_all, y_val[:len(test_k_all), :, :])

    results = {}
    #results["NMI"] = test_nmi
    #results["Purity"] = test_purity
    #results["MSE"] = test_mse
#    results["optimization_target"] = 1 - test_nmi
    if not os.path.exists("SOM-VAE-labels.csv"):
        pd_labels = pd.DataFrame(data = np.nan, index = np.arange(len(test_k_all)), columns = [embedding_string])
    else:
        pd_labels = pd.read_csv("SOM-VAE-labels.csv", index_col = 0, header = 0)
        
    pd_labels[embedding_string] = test_k_all
    print("Available columns: ", pd_labels.columns)
    print(pd_labels[embedding_string].value_counts())
    
    pd_labels.to_csv('SOM_VAE_labels.csv', index = True, header = True)
    
    # print(" Labels distribution: unique - {} \n counts - {}".format(unique_values, counts))

    return results
 

@ex.automain
def main(latent_dim, som_dim, learning_rate, decay_factor, alpha, beta, gamma, tau, seed, modelpath, save_model):
    """Main method to build a model, train it and main.py it.
    
    Args:
        latent_dim (int): Dimensionality of the SOM-VAE's latent space.
        som_dim (list): Dimensionality of the SOM.
        learning_rate (float): Learning rate for the training.
        decay_factor (float): Factor for the learning rate decay.
        alpha (float): Weight for the commitment loss.
        beta (float): Weight for the SOM loss.
        gamma (float): Weight for the transition probability loss.
        tau (float): Weight for the smoothness loss.
        modelpath (path): Path for the model checkpoints.
        save_model (bool): Indicates if the model should be saved after training and evaluation.
        
    Returns:
        dict: Results of the evaluation (NMI, Purity, MSE).
    """
    # Dimensions for MNIST-like data
    tf.reset_default_graph()
    
    
    input_length = 12
    input_channels = 27
    x = tf.placeholder(tf.float32, shape=[None, 12, 27, 1])

    lr_val = tf.placeholder_with_default(learning_rate, [])

    model = SOMVAE(inputs=x, latent_dim=latent_dim, som_dim=som_dim, learning_rate=lr_val, decay_factor=decay_factor,input_length=input_length, input_channels=input_channels, alpha=alpha, beta=beta, gamma=gamma, tau=tau)
    print("model loaded")
    
    train_model(model, x, lr_val, data = (X_train, y_train, X_val, y_val), seed = seed)

    embedding_string = "a{}_b{}_g{}_t{}_l{}_sd{}{}_s{}".format(alpha, beta, gamma, tau, latent_dim,
                                                               som_dim[0], som_dim[1], seed)
    result = evaluate_model(model, data = (X_test, y_test), embedding_string = embedding_string)

    if not save_model:
        shutil.rmtree(os.path.dirname(modelpath))

    return result



