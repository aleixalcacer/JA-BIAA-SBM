# Load the dataset.

import pandas as pd

# the first value passed to the script is the name of the dataset

import sys
dataset = sys.argv[1]

if dataset == "restaurants":
    df = pd.read_csv("datasets/restaurants/rating_final.csv")
    df = df[df.rating != 0]

elif dataset == "nips":
    import scipy.io as scio

    mat = scio.loadmat('datasets/nips/nips12raw_str602.mat')

    df = pd.DataFrame(columns=['paper_id', 'author_id'])

    for paper_id, authors_id in enumerate(mat["apapers"]):
        authors_id = authors_id[0][0]
        for author_id in authors_id:
            df_i = pd.DataFrame({"paper_id": f"p{paper_id}", "author_id": f"a{author_id}"}, index=[paper_id])
            df = pd.concat([df, df_i], ignore_index=True)

elif dataset == "drugs":
    df = pd.read_csv("datasets/drugs/ChSe-Decagon_monopharmacy.csv")
else:
    raise ValueError("Unknown dataset")


print(f"Dataset: {dataset}")

# Create a bipartite graph and extract the biadjacency matrix
from networkx.algorithms import bipartite
import networkx as nx
import numpy as np

bottom_nodes = df.iloc[:, 0].unique()
top_nodes = df.iloc[:, 1].unique()

# generator = np.random.default_rng(0)
# bottom_nodes = generator.choice(bottom_nodes, 500, replace=False)
# top_nodes = generator.choice(top_nodes, 500, replace=False)

B = nx.Graph()
B.add_nodes_from(bottom_nodes, bipartite=0)
B.add_nodes_from(top_nodes, bipartite=1)

B.add_edges_from([(row.iloc[0], row.iloc[1]) for idx, row in df.iterrows()])

# display #nodes (both subsets) and #edges
print("Nodes:", B.number_of_nodes())
print("Edges:", B.number_of_edges())


a_np = bipartite.biadjacency_matrix(B, bottom_nodes, top_nodes).todense().astype(np.float32)

print(a_np.shape)

# Run the models
from sklearn.metrics import average_precision_score, roc_auc_score
from models import BiAA, SBM, DBiAA, DSBM
from tqdm.auto import tqdm
from itertools import product
import torch
import numpy as np

k_min, k_max = 2, 10

n_sims = 10
n_reps = 5

models = [BiAA, SBM, DBiAA, DSBM]
assignments = ["soft", "hard"]

# Set some 1s to 0s
a = torch.tensor(a_np)

results = None
results_loss = None

for sim_i in tqdm(range(n_sims), leave=False):

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

    for k in tqdm(range(k_min, k_max), leave=False):

        biaa_model = None
        for assignment, model in tqdm(list(product(assignments, models)), leave=False):
            auc_j = -np.inf
            prauc_j = -np.inf

            best_model = None
            for rep_i in tqdm(range(n_reps), leave=False):
                if model in [BiAA, DBiAA]:
                    model_i = model((k, k), a_zero, likelihood="bernoulli", assignment=assignment, device="cuda")
                else:
                    model_i = model((k, k), a_zero, likelihood="bernoulli", assignment=assignment, device="cuda",
                                    biaa_model=biaa_model)

                lr = 0.025
                n_epochs = 4_000
                model_i.fit(n_epochs, learning_rate=lr, threshold=0)

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
            data = {"auc": auc_j, "prauc": prauc_j, "model": model.__name__, "assignment": assignment, "k": k,
                    "iteration": sim_i}
            results_i = pd.DataFrame(data=data, index=[0])
            results = pd.concat([results, results_i], ignore_index=True)

            # loss
            data = {"loss": best_model.losses[::20], "step": np.arange(len(best_model.losses))[::20],
                    "model": model.__name__,
                    "assignment": assignment, "k": k, "iteration": sim_i}
            results_loss_i = pd.DataFrame(data=data)
            results_loss = pd.concat([results_loss, results_loss_i], ignore_index=True)

    results.to_csv(f"results/{dataset}_x_results_auc.csv")
    results_loss.to_csv(f"results/{dataset}_x_results_loss.csv")
