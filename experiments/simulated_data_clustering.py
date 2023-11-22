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

for i in tqdm(range(n_iters), leave=False):
    for alpha in tqdm(alphas, leave=False):
        generator = np.random.default_rng(i)

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
        if i == 0:
            # run matplotlib in a remote server
            mpl.use('Agg')
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            heatmap(data, ax=ax)
            fig.savefig(f"figures/simulated_data_{alpha}_{i}.pdf")

        data_shuffle, info_shuffle = shuffle_dataset(data, generator=generator)

        data_tensor = torch.tensor(data_shuffle, dtype=torch.float32)

        biaa_model = None

        for assignment, model in tqdm(list(product(assignments, models)), leave=False):
            best_model = None
            for _ in tqdm(range(5), leave=False):
                if model in [BiAA, DBiAA]:
                    model_i = model(n_archetypes, data_tensor, likelihood="bernoulli", assignment=assignment,
                                    device="cpu")
                else:
                    model_i = model(n_archetypes, data_tensor, likelihood="bernoulli", assignment=assignment,
                                    device="cpu",
                                    biaa_model=biaa_model)
                model_i.fit(1_000, learning_rate=0.01, threshold=0)

                if best_model is None or best_model.losses[-1] > model_i.losses[-1]:
                    best_model = model_i

            if model in [BiAA, DBiAA]:
                biaa_model = best_model

            model_alphas = [best_model.A.cpu().detach().numpy(), best_model.D.cpu().detach().numpy().T]
            estimated_archetypes = best_model.Z.cpu().detach().numpy()
            data_sorted, info_sorted = sort_by_archetype_similarity(data_shuffle, model_alphas, estimated_archetypes)

            data = {"loss": best_model.losses[::20],
                    "step": np.arange(len(best_model.losses))[::20],
                    "model": model.__name__,
                    "assignment": assignment,
                    "alpha": alpha,
                    "iteration": i
                    }

            results_loss_i = pd.DataFrame(data=data)
            results_loss = pd.concat([results_loss, results_loss_i], ignore_index=True)

            # select only labels that has a score grater than 0.9
            for threshold in thresholds:
                true_labels = labels[0][info_shuffle["perms"][0]][info_sorted["perms"][0]]
                pred_labels = info_sorted["labels"][0]
                scores = info_sorted["scores"][0]
                pred_labels = pred_labels[scores > threshold]
                true_labels = true_labels[scores > threshold]

                # compare labels using normalized mutual information
                nmi = normalized_mutual_info_score(true_labels, pred_labels)

                # save results as a dataframe
                data = {"model": model.__name__,
                        "assignment": assignment,
                        "alpha": alpha,
                        "threshold": threshold,
                        "nmi": nmi,
                        "iteration": i
                        }
                results_i = pd.DataFrame(data=data, index=[0])
                results = pd.concat([results, results_i], ignore_index=True)

    results.to_csv(f"results/simulated_data_results.csv", index=False)
    results_loss.to_csv(f"results/simulated_data_results_loss.csv", index=False)
