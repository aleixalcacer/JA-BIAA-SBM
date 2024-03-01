from archetypes.datasets import make_archetypal_dataset
import numpy as np
from time import time
from archetypes.algorithms import BiAA
from archetypes.algorithms.torch import BiAA as BiAA_nn
import torch
import pandas as pd

dataframe = None
for n_archetypes_i in [5, 10, 50, 100]:
    print(n_archetypes_i)
    for n_elements in [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]:
        print(n_elements)
        shape = (int(np.sqrt(n_elements)), int(np.sqrt(n_elements)))
        n_archetypes = (n_archetypes_i, n_archetypes_i)

        # check that shape[0] > n_archetypes[0] and shape[1] > n_archetypes[1]:
        if shape[0] < n_archetypes_i or shape[1] < n_archetypes_i:
            continue

        archetypes = np.random.uniform(0, 1, n_archetypes)

        data, _ = make_archetypal_dataset(archetypes, shape, noise=0.1, generator=n_elements)
        data = data.astype(np.float32)

        if n_elements > 10_000:
            dataframe_i = None
        else:
            start = time()
            model_biaa = BiAA(n_archetypes, max_iter=10_000, tol=0)
            model_biaa.fit(data)
            stop = time()

            biaa_time = stop - start

            dataframe_i = pd.DataFrame({
                "n_elements": [n_elements],
                "n_archetypes": [n_archetypes_i],
                "model": ["BiAA"],
                "device": ["cpu"],
                "time": [biaa_time]
            })

        # PyTorch based algorithms

        data_torch = torch.tensor(data, dtype=torch.float32)

        start = time()
        model_biaa_nn = BiAA_nn(n_archetypes, *shape, device="cpu")
        model_biaa_nn.train(data_torch, 10_000, learning_rate=0.05)
        stop = time()

        biaa_nn_time = stop - start

        dataframe_i = pd.concat([dataframe_i, pd.DataFrame({
            "n_elements": [n_elements],
            "n_archetypes": [n_archetypes_i],
            "model": ["BiAA (gradient-based)"],
            "device": ["cpu"],
            "time": [biaa_nn_time]
        })], ignore_index=True)

        start = time()
        model_biaa_nn = BiAA_nn(n_archetypes, *shape, device="cuda")
        model_biaa_nn.train(data_torch, 10_000, learning_rate=0.05)
        stop = time()

        biaa_nn_cuda_time = stop - start

        dataframe_i = pd.concat([dataframe_i, pd.DataFrame({
            "n_elements": [n_elements],
            "n_archetypes": [n_archetypes_i],
            "model": ["BiAA (gradient-based)"],
            "device": ["cuda"],
            "time": [biaa_nn_cuda_time]
        })], ignore_index=True)

        dataframe = pd.concat([dataframe, dataframe_i], ignore_index=True)

        # save dataframe

        dataframe.to_csv("performance_results.csv", index=False)
