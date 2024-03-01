#%%
import numpy as np
import pandas as pd
import scipy
from archetypes.datasets import sort_by_archetype_similarity

from models import NAA
import torch

mat = scipy.io.loadmat('datasets/irmdata/uml.mat')
d = mat['Rs']

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

data = torch.tensor(d).float().reshape(d.shape).to(device)

n_reps = 10

results = None

for rep_i in range(n_reps):
    best_model = None
    best_score = np.inf
    for _ in range(5):
        model = NAA((15, 15, 21), d.shape, relations=[0, 0, 1], degree_correction=True,
                     membership="soft", loss="bernoulli", device=device)

        model.fit(data, n_epochs=5_000, learning_rate=0.01)

        if model.losses[-1] < best_score:
            best_score = model.losses[-1]
            best_model = model

    model = best_model

    alphas = [a.cpu().detach().numpy() for a in model.A]
    estimated_archetypes = model.Z.cpu().detach().numpy()

    _, info_s = sort_by_archetype_similarity(data[0, :, :], alphas[1:3], estimated_archetypes[0, :, :])

    gnames = np.array([str(*g) for g in mat["gnames"][0]])[info_s["perms"][0]]
    names = np.array([str(*n[0]) for n in mat["names"]])[info_s["perms"][0]]
    labels = info_s["labels"][0]
    scores = info_s["scores"][0]

    df = pd.DataFrame({"gname": gnames, "name": names, "label": labels, "score": scores})

    #%%
    true_label = df.gname.astype("category").cat.codes
    pred_label = df.label.astype("category").cat.codes

    from sklearn.metrics import adjusted_rand_score

    rand_score = adjusted_rand_score(true_label, pred_label)

    results_i = pd.DataFrame({"rep_i": rep_i, "rand_score": rand_score}, index=[0])
    results = pd.concat([results, results_i], axis=0)

    # save results
    results.to_csv("results/uml_adj_rand_score.csv", index=False)
