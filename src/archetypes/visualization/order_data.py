import pandas as pd

def order_data_hard(data, model):
    return order_data_soft(data, model, max(data.shape))


def order_data_soft(data, model, t=5):
    i = [i for i in range(model.A.shape[0])]
    max, argmax = model.A.max(1)

    df_data = {
        "i": i,
        "max": max.detach().numpy(),
        "argmax": argmax.detach().numpy(),
    }

    i = pd.DataFrame(df_data) \
        .sort_values(["argmax", "max"], ascending=[True, False]) \
        .groupby("argmax") \
        .head(t)

    iind = i["i"].to_numpy()

    j = [j for j in range(model.D.shape[1])]
    max, argmax = model.D.max(0)

    df_data = {
        "j": j,
        "max": max.detach().numpy(),
        "argmax": argmax.detach().numpy(),
    }

    j = pd.DataFrame(df_data) \
        .sort_values(["argmax", "max"], ascending=[True, False]) \
        .groupby("argmax") \
        .head(t)

    jind = j["j"].to_numpy()

    ordered_data = data[iind, :][:, jind]

    hlines = [0] + list(i.groupby("argmax").count().cumsum()["i"].to_numpy())

    vlines = [0] + list(j.groupby("argmax").count().cumsum()["j"].to_numpy())

    return ordered_data, hlines, vlines