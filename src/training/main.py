from tqdm import tqdm
import itertools
import os
import utils

import numpy as np
import pandas as pd

np.set_printoptions(precision=3, suppress = True)

# generated_seeds = [1001, 1012,1134, 2475, 6138, 7415, 1663, 7205, 9253, 1782]
P = [0.01]
D = [1]
generated_seeds_5 = [1001, 1002, 1003]

K_set           = [6, 8, 10, 12, 15, 20]
latent_set      = [64, 128, 256]
epochs_set          = [30]
lr_set              = [0.001]
#var_set = ["vit", "vit-sta", "vit-lab", "vit-lab-sta"]
var_set = ["vit-lab-sta"]


scores_df = pd.DataFrame(data = 0, index = ["auc", "f1", "rec"], columns = [])
for  K, latent_dim, seed in tqdm(itertools.product(K_set, latent_set,generated_seeds_5)):

    print("K: {}\nlatent-dim: {}\nseed: {}\nlr: {}\nepochs: {}".format(
        K, latent_dim, seed))
    
    auc, f1, rec = os.system("""python run_model.py --K {} --latent_dim {} --seed {}""".format(
        K, latent_dim, seed))
        
    scores_df["K{}_L{}_S{}".format(K, latent_dim, seed)] = auc, f1, rec

scores_df.to_csv("scores_df", index = True, header = True)

