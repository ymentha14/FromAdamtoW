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
import numpy as np
import matplotlib

sns.set()

coladam = "#3e9eab"
# color for adamW
coladamW = "#db2e59"
# color for SGD
colSGD = "#ffa200"

def augment_data(loss,N):
    """
    create additional fake data similar to loss with gaussian noise
    
    Args:
        loss(list of float): losses from one run
        N(int): number of simulated runs to create
    """
    return np.abs((np.random.normal(0,0.0003,[N,23]) + loss).T)

def get_inputs_linreg(losses):
    """
    return the x y necessary to fit a linear regression
    
    Args:
        losses (list of list of int): all nested lists don't necessarily have the same shape because of early stopping: list of the metric for a given
        combination of hyperparameters
        
    Returns:
        X,y (np.array): data fitted for the input of Linear Regression fitter
    """
    losses = np.asarray(losses)
    X = [np.arange(len(run))+1 for run in losses]
    X = np.concatenate(X,axis=None)
    X = np.expand_dims(X,axis = 1)
    #X = np.expand_dims(X.flatten(),axis = 1)
    y = np.concatenate(losses,axis=None)
    return X,y

def filter_df(df):
    """
    return the same df while dropping any row whose validation loss contains Nan values
    
    Args:
        df (pd.DataFrame): info issue from the json file
    """
    df = df[df['test_losses'].isnull()]
    drop_indexes = df[df.val_losses.apply(lambda x:any([None in y for y in x]))].index
    return df.drop(drop_indexes)
    

def filter_df_test(df):
    """
    return the same df while dropping any row whose validation loss contains Nan values
    
    Args:
        df (pd.DataFrame): info issue from the json file
    """
    df = df[~df['test_losses'].isnull()]
    drop_indexes = df[df.test_losses.apply(lambda x:any([None in y for y in x]))].index
    return df.drop(drop_indexes)
    
    
def plot_losses_fits(losses, 
                     ax, 
                     colors,
                     plot_runs=True,
                     plot_fit=False,
                     plot_mean=False,
                     fit_type="log",
                     label=""):
    """
    Plot the fitted both on loglog and linear plots on top of the data.
    
    Args:
        ax (plt.ax): matplotlib ax
        colors (tuple of str): 2 colors to use, one for the runs, the other for the fit
        plot_runs (bool): plot the runs
        plot_fit (bool): plot the fitted curve
        plot_mean (bool): plot the mean of the runs
        losses(np.array of float): M X N_epochs with M the number of runs in the CV for the given parameter
        fit_type (bool): whether to fit the regression in a log-log or lin-log space
        label (str): label to give to the losses on the plot 
    """
    # color 1 for the losses, color 2 for the fit
    color1,color2 = colors
    
    # we set the limit of the plot to the longest run
    max_n_epochs = max(len(run) for run in losses)
    
    mean_loss = []
    for i in range(max_n_epochs):
        presents = []
        for run in losses:
            # in solid line
            if (i < len(run)):
                presents.append(run[i])
        mean_loss.append(np.mean(presents))
    x_mean = np.expand_dims(np.arange(max_n_epochs)+1,axis=1)
    
    # we recreate the epochs
    x,y = get_inputs_linreg(losses)
    y_log = np.log(y) 
    if (fit_type == "log"):
        x_log = np.log(x) 
        x_mean_log = np.log(x_mean)
    else:
        x_log = x
        x_mean_log = x_mean
    
    # the regression works in the logspace
    reg = LinearRegression().fit(x_log,y_log)
    score = reg.score(x_log,y_log)
    y_pred_log = reg.predict(x_mean_log)
    
    # we recover the equivalent on a linear scale
    y_pred  = np.exp(y_pred_log)
    

    # plot the empirical curves
    if plot_runs:
        for i,loss in enumerate(losses):
            ax.plot(np.arange(len(loss))+1,
                    loss,
                    color=color1,
                    alpha=0.1,
                    label= label if i == 0 and not plot_mean else None)
    if plot_fit:
        ax.plot(x_mean,y_pred,color=color2,label = f'fitted exp curve {label}')
    if plot_mean:
        ax.plot(x_mean,mean_loss,color=color1,label = f'Mean {label}')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Cross Entropy Loss")
    ax.legend(handlelength=5, handleheight=3)    
    
    return reg

def get_concat_losses(df,train : bool):
    """
    concat the losses for the given df into a single list
    
    Args:
        df (pd.DataFrame): info issue from the json file
        train (Bool): train or validation set
    """
    concat_losses = []
    tot_losses = df.train_losses if train else df.val_losses
    for losses in tot_losses.values:
        concat_losses = concat_losses + losses
    return concat_losses


def get_concat_test_losses(df,train : bool):
    """
    concat the losses for the given df into a single list
    
    Args:
        df (pd.DataFrame): info issue from the json file
        train (Bool): train or validation set
    """
    concat_losses = []
    tot_losses = df.train_losses if train else df.test_losses
    for losses in tot_losses.values:
        concat_losses = concat_losses + losses
    return concat_losses

def plot_grid_search(df,
                     ax,
                     plot_SGD=True,
                     plot_Adam=True,
                     plot_AdamW=True,
                     plot_runs=True,
                     plot_fit=False,
                     plot_mean=False,
                     fit_type="log",
                     train=True):
    """
    plot the runs of the computed grid search according to many display parameters
    
    Args:
        df (pd.DataFrame): info issue from the json file
        ax (plt.ax): matplotlib ax
        plot_XXX (bool): whether to plot the XXX optimizer
        plot_runs (bool): plot the runs
        plot_fit (bool): plot the fitted curve
        plot_mean (bool): plot the mean of the runs
        fit_type (bool): whether to fit the regression in a log-log or lin-log space
        train (Bool): train or validation set    
    """
    
    
    args = [plot_runs,plot_fit,plot_mean,fit_type]
    if plot_SGD:
        sgd_df = df[df.optimizer == 'SGD']
        sgd_concat_losses = get_concat_losses(sgd_df,train)
        col1 = (colSGD,"#00ff3c")
        reg1 = plot_losses_fits(sgd_concat_losses, ax, col1,*args,label="SGD")
    if plot_Adam:
        adam_df = df[df.optimizer == 'Adam']
        adam_concat_losses = get_concat_losses(adam_df,train)
        col2 = (coladam,"#08f0fc")
        reg2 = plot_losses_fits(adam_concat_losses, ax,col2,*args,label="Adam")
    if plot_AdamW:
        adamW_df = df[df.optimizer == 'AdamW']
        adamW_concat_losses = get_concat_losses(adamW_df,train)
        col3 = (coladamW, "yellow")
        reg3 = plot_losses_fits(adamW_concat_losses, ax,col3,*args,label="AdamW")
    bottom,top = ax.get_ylim()
    ax.set_yticks(np.linspace(bottom,top,5))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    

def plot_grid_search_test(df,
                         ax,
                         plot_SGD=True,
                         plot_Adam=True,
                         plot_AdamW=True,
                         plot_runs=True,
                         plot_fit=False,
                         plot_mean=False,
                         fit_type="log",
                         train=True):
    """
    plot the runs of the computed grid search according to many display parameters
    
    Args:
        df (pd.DataFrame): info issue from the json file
        ax (plt.ax): matplotlib ax
        plot_XXX (bool): whether to plot the XXX optimizer
        plot_runs (bool): plot the runs
        plot_fit (bool): plot the fitted curve
        plot_mean (bool): plot the mean of the runs
        fit_type (bool): whether to fit the regression in a log-log or lin-log space
        train (Bool): train or validation set    
    """
    
    
    args = [plot_runs,plot_fit,plot_mean,fit_type]
    if plot_SGD:
        sgd_df = df[df.optimizer == 'SGD']
        sgd_concat_losses = get_concat_test_losses(sgd_df,train)
        col1 = (colSGD,"#00ff3c")
        reg1 = plot_losses_fits(sgd_concat_losses, ax, col1,*args,label="SGD")
    if plot_Adam:
        adam_df = df[df.optimizer == 'Adam']
        adam_concat_losses = get_concat_test_losses(adam_df,train)
        col2 = (coladam,"#08f0fc")
        reg2 = plot_losses_fits(adam_concat_losses, ax,col2,*args,label="Adam")
    if plot_AdamW:
        adamW_df = df[df.optimizer == 'AdamW']
        adamW_concat_losses = get_concat_test_losses(adamW_df,train)
        col3 = (coladamW, "yellow")
        reg3 = plot_losses_fits(adamW_concat_losses, ax,col3,*args,label="AdamW")
    bottom,top = ax.get_ylim()
    ax.set_yticks(np.linspace(bottom,top,5))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

