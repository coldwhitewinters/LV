#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from fbprophet import Prophet

pd.plotting.register_matplotlib_converters()

class ProphetForecaster:
    def __init__(
        self,
        use_boxcox=True,
        growth="linear",
        changepoints=None,
        n_changepoints=25,
        changepoint_range=0.8,
        yearly_seasonality="auto",
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=None,
        seasonality_mode="additive",
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        interval_width=0.8,
        uncertainty_samples=1000,
        stan_backend=None):
    
        self.models = dict()
        self.fcst = dict()
        self.lmbda_boxcox = dict()
        self.use_boxcox = use_boxcox
        self.prophet_config = {
            "growth":growth,
            "changepoints":changepoints,
            "n_changepoints":n_changepoints,
            "changepoint_range":changepoint_range,
            "yearly_seasonality":yearly_seasonality,
            "weekly_seasonality":weekly_seasonality,
            "daily_seasonality":daily_seasonality,
            "holidays":holidays,
            "seasonality_mode":seasonality_mode,
            "seasonality_prior_scale":seasonality_prior_scale,
            "holidays_prior_scale":holidays_prior_scale,
            "changepoint_prior_scale":changepoint_prior_scale,
            "mcmc_samples":mcmc_samples,
            "interval_width":interval_width,
            "uncertainty_samples":uncertainty_samples,
            "stan_backend":stan_backend }

    def fit(self, train_df):
        for product in train_df.columns:
            target = train_df[product].dropna()
            if self.use_boxcox:
                idx = target.index
                target, self.lmbda_boxcox[product] = boxcox(target)
                target = pd.Series(target, index=idx)
            target.index.name = "ds"
            target.name = "y"
            target = target.reset_index()
            self.models[product] = Prophet(**self.prophet_config)
            self.models[product].fit(target)
        return self.models
            
    def predict(self, steps=365, freq="D"):
        for product, model in self.models.items():
            future = model.make_future_dataframe(steps, freq=freq)
            pred = model.predict(future).set_index("ds")
            pred = pred[["yhat_lower", "yhat", "yhat_upper"]]
            self.fcst[product] = pred
            if self.use_boxcox:
                self.fcst[product] = inv_boxcox(
                    self.fcst[product], 
                    self.lmbda_boxcox[product])
        return pd.concat(self.fcst, axis=1)
