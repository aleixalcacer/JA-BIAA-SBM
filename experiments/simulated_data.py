#!/usr/bin/env python
# coding: utf-8


import sys
sys.path.append('/Users/aleix11alcacer/Projects/archetypes')

import numpy as np
import torch
from models import BiAA, SBM, DBiAA, DSBM
from archetypes.datasets import sort_by_archetype_similarity, make_archetypal_dataset, shuffle_dataset
from archetypes.visualization import heatmap
from itertools import product
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score

n_archetypes = (3, 3)

# define a numpy generator
generator = np.random.default_rng(7)

archetypes = generator.uniform(0, 1, n_archetypes)
uniform_low = generator.uniform(0, 0.2, n_archetypes)
uniform_high = generator.uniform(0.75, 1, n_archetypes)

archetypes = np.where(archetypes < 0.8, uniform_low, uniform_high)

shape = (100, 100)

n_iters = 25
alphas = [0, 0.25, 0.5, 1]
models = [BiAA, SBM, DBiAA, DSBM]
thresholds = [0.9, 0.75, 0.5, 0]
assignments = ["soft", "hard"]

results = None
results_loss = None

for alpha in tqdm(alphas, leave=False):

    generator = np.random.default_rng(0)
    data, labels = make_archetypal_dataset(
        archetypes,
        shape,
        alpha=alpha,
        generator=generator,
        noise=0.1
    )

    data = np.clip(data, 0, 1)

    data = generator.binomial(1, data)  # binarize the data

    # plot data and save it as pdf
    mpl.use('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    heatmap(data, ax=ax)
    fig.savefig(f"figures/simulated_data_{alpha}.pdf")

    data_shuffle, info_shuffle = shuffle_dataset(data, generator=generator)
    a = torch.tensor(data_shuffle, dtype=torch.float32)

    for sim_i in tqdm(range(n_iters), leave=False):
        # Clone the original matrix to remove some 1s
        a_zero = a.clone()

        # Get the indices of the 1s in the original matrix
        x, y = np.where(a == 1)
        generator = np.random.default_rng(int(sim_i))

        # Select 10% of the 1s
        rep_i = generator.choice(len(x), int(len(x) / 10), replace=False)
        x_to_remove = x[rep_i]
        y_to_remove = y[rep_i]

        # remove the selected 1s
        for x_i, y_i in zip(x_to_remove, y_to_remove):
            # Check if the node is not isolated
            if a_zero[x_i, :].sum() > 1 and a_zero[:, y_i].sum() > 1:
                a_zero[x_i, y_i] = 0

        # Get the indices of the 0s in the new matrix
        x_zero, y_zero = np.where(a_zero == 0)

        # Get the labels of the 0s in the original matrix
        y_true = a[x_zero, y_zero].detach().numpy().astype(bool)

        biaa_model = None
        for assignment, model in tqdm(list(product(assignments, models)), leave=False):
            auc_j = -np.inf
            prauc_j = -np.inf
            best_model = None
            for _ in tqdm(range(5), leave=False):
                if model in [BiAA, DBiAA]:
                    model_i = model(n_archetypes, a_zero, likelihood="bernoulli", assignment=assignment,
                                    device="cuda")
                else:
                    model_i = model(n_archetypes, a_zero, likelihood="bernoulli", assignment=assignment,
                                    device="cuda", biaa_model=biaa_model)

                model_i.fit(2_000, learning_rate=0.01, threshold=0)

                # Reconstruct the matrix
                if model in [BiAA, SBM]:
                    a_rec = model_i.A @ model_i.Z @ model_i.D
                else:
                    a_rec = model_i.a[:, None] * model_i.A @ model_i.Z @ model_i.D * model_i.d[None, :]

                # Select the 0s in the reconstructed matrix
                y_score = a_rec[x_zero, y_zero].cpu().detach().numpy()

                # Compute the roc_auc and average_precision_score
                auc_i = roc_auc_score(y_true, y_score)
                prauc_i = average_precision_score(y_true, y_score)

                if auc_j < auc_i:
                    auc_j = auc_i
                if prauc_j < prauc_i:
                    prauc_j = prauc_i

                if not best_model or best_model.losses[-1] < model_i.losses[-1]:
                    best_model = model_i

            if model in [BiAA, DBiAA]:
                biaa_model = best_model

                # Save the results in a dataframe

                # auc and prauc
            data = {"auc": auc_j, "prauc": prauc_j, "model": model.__name__, "assignment": assignment, "alpha": alpha,
                    "iteration": sim_i}
            results_i = pd.DataFrame(data=data, index=[0])
            results = pd.concat([results, results_i], ignore_index=True)

            # loss
            data = {"loss": best_model.losses[::20], "step": np.arange(len(best_model.losses))[::20],
                    "model": model.__name__,
                    "assignment": assignment, "alpha": alpha, "iteration": sim_i}
            results_loss_i = pd.DataFrame(data=data)
            results_loss = pd.concat([results_loss, results_loss_i], ignore_index=True)

        results.to_csv(f"results/simulated_data_x_results_auc.csv")
        results_loss.to_csv(f"results/simulated_data_x_results_loss.csv")
