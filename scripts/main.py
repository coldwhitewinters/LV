#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet_forecaster import ProphetForecaster

plt.rcParams["figure.figsize"] = (14,4)

def flatten(x):
    return [z for y in x for z in y ]

def get_forecastables(df, T=0.5):
    amount_info = df.notna().sum() / len(df)
    return df.loc[:, amount_info > T].copy()

def plot_fcst(fcst, train=None, test=None):
    ax = fcst.yhat.plot()
    lower = fcst.yhat_lower.interpolate()
    upper = fcst.yhat_upper.interpolate()
    plt.fill_between(fcst.index, y1=lower, y2=upper, alpha=0.3)
    if train is not None:
        train.plot(style="k.")
    if test is not None:
        test.plot(style="r.")
    plt.show()

data = pd.read_csv(
    "../data/base_limpia.csv", 
    parse_dates=["tiempo", "fecha pedido", "fecha liq"]
    )

u_producto = pd.pivot_table(
    data, 
    values="u pedidas", 
    index="tiempo", 
    columns="producto", 
    aggfunc="sum"
    ).asfreq("D")

forecastables = get_forecastables(u_producto)
forecastables = forecastables.iloc[:, :3]
train, test = forecastables[:"2018-12-31"], forecastables["2019-01-01":]
model = ProphetForecaster()
model.fit(train)
fcst = model.predict(steps=365).asfreq("D")

prod_id = 16061
plot_fcst(fcst[prod_id], train[prod_id], test[prod_id])
    
    
    
