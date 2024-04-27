import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import pandas_datareader as web
from dataset.utils import *

class FRED():
    def __init__(self, date_time, symbol):
        super().__init__()
        self.date_time = date_time
        self.symbol = symbol
    
    def _get_fred_dataset(self):
        dataset = web.DataReader(self.symbol, 'fred',start=pd.to_datetime(self.date_time))
        # dataset = reader.read()
        # Get numeric values from the pandas dataframe
        xs = dataset.index
        ys = dataset[self.symbol]
        return xs, ys
    
    def _plot_fred_dataset(self, save_path_plot):
        plt.figure(figsize=(12,6))
        xs, ys = self._get_fred_dataset()
        plt.plot(xs, ys, label=self.symbol)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f"BRENT crude oil prices from FRED about {self.symbol}")
        plt.legend()
        plt.savefig(f"{save_path_plot}/{self.symbol}.png")
        plt.show()
    
    def _process_fred_dataset(self):
        xs, ys = self._get_fred_dataset()
        ys_imputed = basic_imputation(ys)
        return xs, ys_imputed
    
    def _plot_fred_dataset_imputed(self, save_path_plot):
        plt.figure(figsize=(12,6))
        xs, ys_imputed = self._process_fred_dataset()
        plt.plot(xs, ys_imputed, label=self.symbol)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f"BRENT crude oil prices from FRED about {self.symbol} with imputation")
        plt.legend()
        plt.savefig(f"{save_path_plot}/{self.symbol}_imputed.png")
        plt.show()