# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 23:31:11 2020

@author: ariel
"""

import numpy as np
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import pmdarima as pm
from progress_bar import ProgressBar


class ARIMAForecaster:
    def __init__(
        self,
        use_boxcox=True,
        n_fourier_terms=10,
        seasonality=[365.25],
        interval_width=0.8):
    
        self.models = dict()
        self.fcst = dict()
        self.lmbda_boxcox = dict()
        self.use_boxcox = use_boxcox
        self.n_fourier_terms = n_fourier_terms
        self.seasonality = seasonality
        self.interval_width = interval_width
        

    def fit(self, train_df):
        self.ds = pd.Series(train_df.index)
        print("Fitting...")
        progress_bar = ProgressBar(len(train_df.columns))
        
        for item in train_df.columns:
            target = train_df[item].interpolate().bfill()
            if self.use_boxcox:
                idx = target.index
                target, self.lmbda_boxcox[item] = boxcox(target)
                target = pd.Series(target, index=idx)
            target.index.name = "ds"
            target.name = "y"     
            self.models[item] = pm.auto_arima(
                target,
                seasonal=False,
                exogenous=fourier(
                    len(target), 
                    seasonality=self.seasonality, 
                    n_terms=self.n_fourier_terms), 
                method="bfgs",
                suppress_warnings=True)
            progress_bar.update()
        progress_bar.finish()
        return self.models
            
    def predict(self, steps=365):
        print("Forecasting...")
        progress_bar = ProgressBar(len(self.models.items()))
        for item, model in self.models.items():
            pred = model.predict(
                exogenous=fourier(
                    steps, 
                    seasonality=self.seasonality, 
                    n_terms=self.n_fourier_terms),
                n_periods=steps, 
                return_conf_int=True,
                alpha=(1.0 - self.interval_width))
            fcst = pd.DataFrame()
            fcst["yhat_lower"] = pred[1][:,0]
            fcst["yhat"] = pred[0]
            fcst["yhat_upper"] = pred[1][:,1]
            self.fcst[item] = fcst
            if self.use_boxcox:
                self.fcst[item] = inv_boxcox(
                    self.fcst[item], 
                    self.lmbda_boxcox[item])
            progress_bar.update()
        progress_bar.finish()
        return pd.concat(self.fcst, axis=1)
    
    
def fourier(steps, seasonality, n_terms=10):
    coeff_list = []
    t = np.arange(0, steps)
    for period in seasonality:
        coeff_M = np.zeros((steps, 2*n_terms))
        for k in range(n_terms):
            coeff_M[:, 2*k] = np.sin(2*np.pi*(k+1)*t/period)
            coeff_M[:, 2*k+1] = np.cos(2*np.pi*(k+1)*t/period)
        coeff_list.append(coeff_M)
    coeff = np.concatenate(coeff_list, axis=1)
    return coeff