# Load the dataset.

import pandas as pd
import scipy.io as scio
# the first value passed to the script is the name of the dataset

import sys
dataset = sys.argv[1]

if dataset == "alyawarra":
    mat = scio.loadmat('datasets/irmdata/alyawarradata.mat')
    d = mat['Rs']
    n_archetypes = (17, 17)
    relations = [0, 0]

elif dataset == "uml":
    mat = scio.loadmat('datasets/irmdata/uml.mat')
    d = mat['Rs']
    n_archetypes = (15, 15, 22)
    relations = [0, 0, 1]

else:
    raise ValueError("Unknown dataset")

print(f"Dataset: {dataset}")

# Run the models
from sklearn.metrics import average_precision_score, roc_auc_score
from models import NAA, NSBM
from tqdm.auto import tqdm
from itertools import product
import torch
import numpy as np

# check if cuda is available
device = "cuda" if torch.cuda.is_available() else "cpu"

n_sims = 25
n_reps = 5

models = [NAA, NSBM]
degree_corrections = [False, True]

# Set some 1s to 0s
a = torch.tensor(d).float()

results = None
results_loss = None

for sim_i in tqdm(range(n_sims), leave=False):

    # Clone the original matrix to remove some 1s
    a_zero = a.clone()

    # Get the indices of the 1s in the original matrix
    x, y, z = np.where(a == 1)
    generator = np.random.default_rng(int(sim_i))

    # Select 10% of the 1s
    rep_i = generator.choice(len(x), int(len(x) / 10), replace=False)
    x_to_remove = x[rep_i]
    y_to_remove = y[rep_i]
    z_to_remove = z[rep_i]

    # remove the selected 1s
    for x_i, y_i, z_i in zip(x_to_remove, y_to_remove, z_to_remove):
        # Check if the node is not isolated
        if a_zero[x_i, :].sum() > 1 and a_zero[:, y_i].sum() > 1 and a_zero[:, :, z_i].sum() > 1:
            a_zero[x_i, y_i, z_i] = 0

    # Get the indices of the 0s in the new matrix
    x_zero, y_zero, z_zero = np.where(a_zero == 0)

    # Get the labels of the 0s in the original matrix
    y_true = a[x_zero, y_zero, z_zero].detach().numpy().astype(bool)


    aa_model = None
    for degree_correction, model in tqdm(list(product(degree_corrections, models)), leave=False):
        auc_j = -np.inf
        prauc_j = -np.inf

        best_model = None
        for rep_i in tqdm(range(n_reps), leave=False):
            if model in [NSBM]:
                model_i = model(
                    n_archetypes,
                    a_zero.shape,
                    relations=relations,
                    degree_correction=degree_correction,
                    loss="bernoulli",
                    membership="soft",
                    device=device,
                    aa_model=aa_model)
            else:
                model_i = model(
                    n_archetypes,
                    a_zero.shape,
                    relations=relations,
                    degree_correction=degree_correction,
                    loss="bernoulli",
                    membership="soft",
                    device=device)

            lr = 0.025
            n_epochs = 4_000
            model_i.fit(a_zero, n_epochs=n_epochs, learning_rate=lr)

            a_rec = model_i.estimated_data

            # Select the 0s in the reconstructed matrix
            y_score = a_rec[x_zero, y_zero, z_zero].cpu().detach().numpy()

            # Compute the roc_auc and average_precision_score
            auc_i = roc_auc_score(y_true, y_score)
            prauc_i = average_precision_score(y_true, y_score)

            if auc_j < auc_i:
                auc_j = auc_i
            if prauc_j < prauc_i:
                prauc_j = prauc_i

            if not best_model or best_model.losses[-1] < model_i.losses[-1]:
                best_model = model_i

        if model in [NAA]:
            aa_model = best_model

        # Save the results in a dataframe

        # auc and prauc
        data = {"auc": auc_j, "prauc": prauc_j, "model": model.__name__, "iteration": sim_i, "degree_correction": degree_correction}
        results_i = pd.DataFrame(data=data, index=[0])
        results = pd.concat([results, results_i], ignore_index=True)

        # loss
        data = {"loss": best_model.losses[::20], "step": np.arange(len(best_model.losses))[::20],
                "model": model.__name__, "iteration": sim_i, "degree_correction": degree_correction}
        results_loss_i = pd.DataFrame(data=data)
        results_loss = pd.concat([results_loss, results_loss_i], ignore_index=True)

    results.to_csv(f"results/{dataset}_x_results_auc.csv")
    results_loss.to_csv(f"results/{dataset}_x_results_loss.csv")
