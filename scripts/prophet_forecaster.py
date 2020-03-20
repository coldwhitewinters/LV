
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from fbprophet import Prophet
pd.plotting.register_matplotlib_converters()
from progress_bar import ProgressBar

class ProphetForecaster:
    def __init__(self, use_boxcox=True, prophet_config=dict()):
        self.models = dict()
        self.fcst = dict()
        self.lmbda_boxcox = dict()
        self.use_boxcox = use_boxcox
        self.prophet_config = prophet_config

    def fit(self, train_df):
        print("Fitting...")
        progress_bar = ProgressBar(len(train_df.columns))
        
        for item in train_df.columns:
            target = train_df[item].dropna()
            if self.use_boxcox:
                idx = target.index
                target, self.lmbda_boxcox[item] = boxcox(target)
                target = pd.Series(target, index=idx)
            target.index.name = "ds"
            target.name = "y"
            target = target.reset_index()
            self.models[item] = Prophet(**self.prophet_config)
            self.models[item].fit(target)
            progress_bar.update()
        progress_bar.finish()
        return self.models
            
    def predict(self, steps=365, freq="D"):
        print("Forecasting...")
        progress_bar = ProgressBar(len(self.models.items()))
        for item, model in self.models.items():
            future = model.make_future_dataframe(steps, freq=freq)
            pred = model.predict(future).set_index("ds")
            pred = pred[["yhat", "yhat_lower", "yhat_upper"]]
            self.fcst[item] = pred
            if self.use_boxcox:
                self.fcst[item] = inv_boxcox(
                    self.fcst[item], 
                    self.lmbda_boxcox[item])
            progress_bar.update()
        progress_bar.finish()
        fcst_df = pd.concat(self.fcst, axis=1).sort_index(axis=1)
        return fcst_df
