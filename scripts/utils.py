
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar
import time

class ProgressBar:
    def __init__(self, max_value):
        time.sleep(0.5)
        self.bar = progressbar.ProgressBar(
            min_value=0,
            max_value=max_value,
            initial_value=0,
            widgets = [progressbar.SimpleProgress(), 
                       progressbar.Bar(), 
                       progressbar.Percentage()])
        self.bar.update(0)
        self.counter = 0
    
    def update(self):
        self.bar.update(self.counter + 1)
        self.counter += 1
        
    def finish(self):
        self.bar.finish()       

def flatten(x):
    return [z for y in x for z in y]

def plot_grid(df, n_cols, figsize):
    n_rows = int(np.ceil(len(df.columns)/n_cols))
    df.plot(subplots=True, layout=(n_rows, n_cols), figsize=figsize)

def plot_fcst(fcst, train=None, test=None, ax=None):
    lower = fcst.yhat_lower.interpolate()
    upper = fcst.yhat_upper.interpolate()
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if train is not None:
        train.plot(style="k.", ax=ax)
    if test is not None:
        test.plot(style="r.", ax=ax)
    fcst.yhat.plot(ax=ax)
    ax.fill_between(fcst.index, y1=lower, y2=upper, alpha=0.3)

def get_amount_info(df):
    amount_info = df.notna().sum().sort_values() / len(df)
    return amount_info

def get_forecastables(df, T=0.5, N=None):
    amount_info = get_amount_info(df)
    forecastable = (amount_info > T)
    if N is not None:
        forecastable = forecastable.tail(N)
    return df.loc[:, forecastable.index].copy()
