import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import pandas_datareader as web
from dataset.utils import *

'''Data collection and imputation'''

def get_fred_dataset(opt):
    symbol = opt.symbol 
    reader = web.fred.FredReader(symbol, start=pd.to_datetime(opt.date_time))
    dataset = reader.read()
    #get numeric values from the pandas dataframe
    xs = dataset.index
    ys = dataset[symbol]
    return xs,ys

def get_residual_com(opt, y,index):
    y_df = pd.DataFrame({opt.symbol: y}, index=index)
    y_decom = seasonal_decompose(y_df, model='additive')
    y_res= np.copy(y_decom.resid) 
    #padding missing "nan" datapoints with 1 at the beginning and end of residual sequence y_res
    y_res[0:2]=1
    y_res[y_res.shape[0]-2:y_res.shape[0]]=1
    return y_res

# split a univariate sequence into samples
def split_sequence(opt, sequence):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + opt.window_size
        out_end_ix = end_ix + opt.pred_window
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# STL_SR transform includes 2 STL steps and 1 SR step in the middle; SR stands for spectral residual
def  stl_sr_transform(opt, y,x):
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
    y_res = get_residual_com(opt, y,x)
    scores = generate_spectral_score(y_res.tolist()) #defined in utils.py
    y_trans = get_residual_com(opt, scores+10,x)
    return y_trans