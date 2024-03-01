import numpy as np
import scipy

mat = scipy.io.loadmat('datasets/irmdata/alyawarradata.mat')
d = mat['Rs']

import sys
sys.path.append("/Users/aleix11alcacer/Projects/archetypes")
#%%
from archetypes.algorithms.torch import NAA
from models import BiAA, DBiAA
import torch
from archetypes.datasets import sort_by_archetype_similarity
import pandas as pd

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

data = torch.tensor(d).float().reshape(d.shape).to(device)

n_rep = 10
results = None

for rep_i in range(n_rep):
    best_model = None
    best_score = np.inf
    for _ in range(5):
        model = NAA((16, 16), d.shape, relations=[0, 0], degree_correction=False, membership="soft", device=device,
                    loss="bernoulli")

        model.fit(data, n_epochs=2_000, learning_rate=0.05)

        if model.losses[-1] < best_score:
            best_score = model.losses[-1]
            best_model = model

    model = best_model

    df = pd.DataFrame(mat['features'][:, [3, 4, 8]], columns = ["Gender", "Age", "Kinship"])
    group = model.A[1].cpu().detach().numpy().argmax(axis=1)
    df["group"] = group
    # cut age into 6 bins: 0-7, 8-14, 15-29, 30-44, 45-59, 60-99
    df["Age"] = pd.cut(df["Age"], bins=[0, 7, 14, 29, 44, 59, 99])
    # mutate gender from 0,1 to M,F
    df["Gender"][df["Gender"] == 1] = "M"
    df["Gender"][df["Gender"] == 2] = "F"

    # rename groups to be ordered by kinship
    df = df.sort_values(by=["Kinship"])

    # rename group number to letter by occurrence order

    g_id = df["group"].unique()
    # map group id to letter
    g_id_map = {g_id[i]: chr(i + 65) for i in range(len(g_id))}
    df["group"] = df["group"].map(g_id_map)


    df2 = df.copy()

    df2["Age2"] = df2["Age"].apply(lambda x: 0 if x.right > 44 else 1)

    df2["Gender2"] = df2["Gender"].apply(lambda x: 0 if x == "M" else 1)

    # combine Age2, Gender2 and Kindship into a single column as a tuple

    df2["group2"] = df2[["Age2", "Gender2", "Kinship"]].apply(tuple, axis=1)

    # mutate group2 to numbers from 0 to 16
    df2["group2"] = df2["group2"].astype('category').cat.codes

    df2["group"] = df2["group"].astype('category').cat.codes
    #%%
    from sklearn.metrics import adjusted_rand_score

    rand_score = adjusted_rand_score(df2["group"], df2["group2"])

    results_i = pd.DataFrame({"rep_i": rep_i, "rand_score": rand_score}, index=[0])

    results = pd.concat([results, results_i], axis=0)

    # save results
    results.to_csv("results/alyawarra_adj_rand_score.csv", index=False)
