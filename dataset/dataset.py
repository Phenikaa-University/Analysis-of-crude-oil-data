import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import pandas_datareader as web
from dataset.utils import *

class FREDtrain():
    def __init__(self, opt):
        super(FREDtrain, self).__init__()
        self.date_time = opt.date_time
        self.symbol = opt.symbol
        
    def load_data(self, opt):
        if opt.mode == "stl_sr":
            batch_sample, batch_label = self._split_sequence(opt)
        elif opt.mode == "stl":
            batch_sample, batch_label = self._split_sequence(opt)
        batch_sample = np.expand_dims(batch_sample, axis=2)
        return batch_sample, batch_label
    
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
    
    def _get_residual_com(self):
        xs, ys = self._process_fred_dataset()
        y_df = pd.DataFrame({self.symbol: ys}, index=xs)
        y_decom = seasonal_decompose(y_df, model='multiplicative')
        y_res = np.copy(y_decom.resid)
        # Padding missing "nan" datapoints with 1 at the beginning and end of residual sequence y_res
        y_res[0:2]=1
        y_res[y_res.shape[0]-2:y_res.shape[0]]=1
        return xs, y_res
    
    def _plot_fred_dataset_residual(self, save_path_plot):
        plt.figure(figsize=(12,6))
        xs, y_res = self._get_residual_com()
        plt.plot(xs, y_res, label=self.symbol)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f"BRENT crude oil prices from FRED about {self.symbol} with residual")
        plt.legend()
        plt.savefig(f"{save_path_plot}/{self.symbol}_residual.png")
        plt.show()
    
    def _get_score_sr(self):
        xs, y_res = self._get_residual_com()
        scores = generate_spectral_score(y_res.tolist())
        return xs, scores
    
    def _plot_fred_dataset_score_sr(self, save_path_plot):
        xs, scores = self._get_score_sr()
        plt.figure(figsize=(12,6))
        plt.plot(xs, scores, label=self.symbol)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f"BRENT crude oil prices from FRED about {self.symbol} with spectral residual score")
        plt.legend()
        plt.savefig(f"{save_path_plot}/{self.symbol}_score_sr.png")
        plt.show()
        
    def _stl_sr_transform(self, slide_window: int):
        xs, score = self._get_score_sr()
        y_df = pd.DataFrame({self.symbol: score + slide_window}, index=xs)
        y_decom = seasonal_decompose(y_df, model='multiplicative')
        y_trans = np.copy(y_decom.resid)
        # Padding missing "nan" datapoints with 1 at the beginning and end of residual sequence y_res
        y_trans[0:2]=1
        y_trans[y_trans.shape[0]-2:y_trans.shape[0]]=1
        return xs, y_trans
    
    def _plot_fred_dataset_stl_sr_transform(self, slide_window: int, save_path_plot):
        xs, y_trans = self._stl_sr_transform(slide_window)
        plt.figure(figsize=(12,6))
        plt.plot(xs, y_trans, label=self.symbol)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f"BRENT crude oil prices from FRED about {self.symbol} with STL-SR transform: {slide_window} window")
        plt.legend()
        plt.savefig(f"{save_path_plot}/{self.symbol}_stl_sr_transform.png")
        
    def _split_sequence(self, opt):
        if opt.mode == "stl_sr":
            xs, y_trans = self._stl_sr_transform(opt.slide_window)
        elif opt.mode == "stl":
            xs, y_trans = self._get_residual_com()
        X, y = list(), list()
        y_trans = list(y_trans)
        for i in range(len(y_trans)):
            end_ix = i + opt.window_size
            out_end_ix = end_ix + opt.pred_window
            if out_end_ix > len(y_trans):
                break
            seq_x, seq_y = y_trans[i:end_ix], y_trans[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
class FREDtest():
    def __init__(self, opt):
        super(FREDtest, self).__init__()
        self.date_time = opt.date_time
        self.symbol = opt.symbol
    
    def _get_fred_dataset(self):
        dataset = web.DataReader(self.symbol, 'fred',start=pd.to_datetime(self.date_time))
        # dataset = reader.read()
        # Get numeric values from the pandas dataframe
        xs = dataset.index
        ys = dataset[self.symbol]
        return xs, ys
    
    def load_data(self, opt):
        xs, ys = self._get_fred_dataset()
        ys_test_imputed = self.basic_imputation(ys)
        ys_corrupted, positions = generate_point_outliers(
            raw_data=ys_test_imputed,
            anomaly_fraction=0.005,
            window_size=100,
            pointwise_deviation=3.5,
            rng_seed=2
        )
        ys_corr_res = self.get_residual_com(ys_corrupted, xs)
        if opt.mode == "stl_sr":
            y = self.stl_sr_transform(ys_corr_res, xs)
            batch_sample, batch_label = self.split_sequence(list(y), window=opt.window_size, pred_window=opt.pred_window)
            batch_sample = np.expand_dims(batch_sample, axis=2)
        elif opt.mode == "stl": 
            batch_sample, batch_label = self.split_sequence(list(ys_corr_res), window=opt.window_size, pred_window=opt.pred_window)
        batch_sample = np.expand_dims(batch_sample, axis=2)
        return batch_sample, batch_label, ys_corrupted, positions
    
    
    
    #STL_SR transform includes 2 STL steps and 1 SR step in the middle; SR stands for spectral residual
    def  stl_sr_transform(self, y,x):
        """Parameters
        ----------
        y: 1d array, dtype=float 
            time-series sequence data
        x: DatetimeIndex, dtype='datetime64[ns]',
            Datetime indices of elements of y
            
        Returns
        -------
        y_trans: 1d array, dtype=float 
            transformed sequence with STL_SR

        """
        y_res = self.get_residual_com(y,x)
        scores = generate_spectral_score(y_res.tolist()) #defined in utils.py
        y_trans = self.get_residual_com(scores+10,x)
        return y_trans
    
    #get residual component from sequence y, using STL decomposition
    def get_residual_com(self, y,index):
        y_df = pd.DataFrame({self.symbol: y}, index=index)
        y_decom = seasonal_decompose(y_df, model='multiplicative')
        y_res= np.copy(y_decom.resid) 
        #padding missing "nan" datapoints with 1 at the beginning and end of residual sequence y_res
        y_res[0:2]=1
        y_res[y_res.shape[0]-2:y_res.shape[0]]=1
        return y_res
    
    def basic_imputation(self, ys):
        ys_imputed = np.copy(ys)
        b = np.isnan(ys_imputed)
        nan_positions = np.argwhere(b)
        segments = []
        for i in range(nan_positions.shape[0]):
            start = nan_positions[i]
            end = start
            while np.isnan(ys_imputed[end]) and end < ys.shape[0]:
                end = end + 1
            segments.append((start,end))
        for (start,end) in segments:
            y_start = ys_imputed[start-1]
            y_end = ys_imputed[end]
            linear = linear_impute(y_start=y_start,y_end=y_end,npoints=int((end+1)-start))
            # Ensure linear is a 1-dimensional array
            linear = np.array(linear).flatten()

            # Ensure the slice of ys_imputed matches the shape of linear
            assert len(ys_imputed[int(start):int(end)]) == len(linear)

            ys_imputed[int(start):int(end)] = linear
        return ys_imputed
    
    # split a univariate sequence into samples
    def split_sequence(self, sequence, window, pred_window):
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + window
            out_end_ix = end_ix + pred_window
            if out_end_ix > len(sequence):
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)