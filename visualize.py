#!/usr/bin/env python3

"""
VISUALIZATION
"""
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from IPython.core.debugger import set_trace
import seaborn as sns

sns.set()


def augment_data(loss, N):
    """
    create additional fake data similar to loss with gaussian noise
    """
    return np.abs((np.random.normal(0, 0.0003, [N, 23]) + loss).T)


def plot_losses_fits(losses, axes, colors):
    """
    Plot the fitted both on loglog and linear plots on top of the data.
    
    Args:
        losses(np.array of float): M X N_epochs with M the number of runs in the CV for the given parameter
        colors
    """
    # color 1 for the losses, color 2 for the fit
    color1, color2 = colors
    lin_ax = axes[0]
    log_ax = axes[1]

    log_ax.set_yscale("log")
    log_ax.set_xscale("log")

    # in solid line
    mean_loss = np.mean(losses, axis=1)

    # we recreate the epochs
    x = np.expand_dims(np.arange(ex_loss.shape[0]) + 1, axis=1)
    x_log = np.log(x)
    y_log = np.log(mean_loss)

    # the regression works in the logspace
    reg = LinearRegression().fit(x_log, y_log)
    score = reg.score(x_log, y_log)
    y_pred_log = reg.predict(x_log)

    # we recover the equivalent on a linear scale
    y_pred = np.exp(y_pred_log)

    # plot the empirical curves
    lin_ax.set_title("Linear plot of the loss")
    lin_ax.set_xlabel("Epochs")
    lin_ax.set_ylabel("Cross Entropy Loss")

    lin_ax.plot(x, losses, color=color1, alpha=0.2)
    lin_ax.plot(x, mean_loss, color=color1, label="mean_runs")
    lin_ax.plot(x, y_pred, color=color2, label="fitted exp curve")
    lin_ax.legend()
    log_ax.plot(x, losses, color=color1, alpha=0.2)
    log_ax.plot(x, mean_loss, color=color1)

    # plot the fitted curve
    log_ax.set_title("Logarithmic plot of the loss")
    log_ax.set_xlabel("Epochs")
    log_ax.set_ylabel("Cross Entropy Loss")

    log_ax.plot(x, y_pred, color=color2)

    return reg


if __name__ == "__main__":
    # TODO: dynmamic file loading
    df = pd.read_json("./results/log_images_results.json")
    # normalizes the structure
    df_res = pd.json_normalize(df.results)
    df_bestparam = pd.json_normalize(df.best_param)
    # reunify the dataframe
    df = pd.concat([df, df_res, df_bestparam], axis=1)
    df.drop(columns=["results", "best_param"], inplace=True)
    # we take the first parameter combination (arbitrary)
    # TODO: ex_loss should become an array of array
    ex_loss = np.array(df.iloc[0].train_loss)
    # TODO: get rid of the augment_data
    N = 3  # number of simulated runs
    fake_losses = augment_data(ex_loss, N)
    fake_losses2 = augment_data(ex_loss + 0.002, N)

    fig, axes = plt.subplots(2, figsize=(10, 10))
    col1 = ("#3e9eab", "#db2e59")
    reg1 = plot_losses_fits(fake_losses, axes, col1)
    col2 = ("#3e57ab", "#ed7002")
    reg2 = plot_losses_fits(fake_losses2, axes, col2)

    print(score1, score2)
