""" Anomaly detection in time series data with CNN+STL
"""

#basic imports
from utils import *
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import pandas_datareader as web
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout # type: ignore
from statsmodels.tsa.seasonal import seasonal_decompose

"""Hyperparameters"""
window = 100              # History window (number of time stamps taken into account) i.e., filter(kernel) size       
pred_window = 1          # Prediction window (number of time stampes required to be predicted)

#cnn structure parameters
n_features = 1           # Univariate time series
kernel_size = 2          # Size of filter in conv layers
num_filt_1 = 16          # Number of filters in first conv layer
num_filt_2 = 16          # Number of filters in second conv layer
num_nrn_dl = 20          # Number of neurons in dense layer
num_nrn_ol = pred_window # Number of neurons in output layer

conv_strides = 1
pool_size_1 = 2          # Length of window of pooling layer 1
pool_size_2 = 2          # Length of window of pooling layer 2
pool_strides_1 = 2       # Stride of window of pooling layer 1
pool_strides_2 = 2       # Stride of window of pooling layer 2

#training parameters
epochs = 100
batch_size = 32
dropout_rate = 0.1       # Dropout rate in the fully connected layer
learning_rate = 0.0002


'''Data collection and imputation'''
def get_fred_dataset(symbol,date_time):
    symbol = symbol 
    reader = web.fred.FredReader(symbol, start=pd.to_datetime(date_time))
    dataset = reader.read()
    #get numeric values from the pandas dataframe
    xs = dataset.index
    ys = dataset[symbol]
    return xs,ys

symbol = "DCOILBRENTEU"

#get residual component from sequence y, using STL decomposition
def get_residual_com(y,index):
    y_df = pd.DataFrame({symbol: y}, index=index)
    y_decom = seasonal_decompose(y_df, model='multiplicative')
    y_res= np.copy(y_decom.resid) 
    #padding missing "nan" datapoints with 1 at the beginning and end of residual sequence y_res
    y_res[0:2]=1
    y_res[y_res.shape[0]-2:y_res.shape[0]]=1
    return y_res
# split a univariate sequence into samples
def split_sequence(sequence):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + window
        out_end_ix = end_ix + pred_window
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# build CNN model 
def cnn_stl_structure():
    model = Sequential()
    model.add(Conv1D(filters=num_filt_1,
                     kernel_size=kernel_size,
                     strides=conv_strides,
                     padding='valid',
                     activation='relu',
                     input_shape=(window, n_features)))
    model.add(MaxPooling1D(pool_size=pool_size_1)) 
    model.add(Conv1D(filters=num_filt_2,
                     kernel_size=kernel_size,
                     strides=conv_strides,
                     padding='valid',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size_2))
    model.add(Flatten())
    model.add(Dense(units=num_nrn_dl, activation='relu')) 
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=num_nrn_ol))
    return model
def  stl_sr_transform(y,x):
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
    y_res = get_residual_com(y,x)
    scores = generate_spectral_score(y_res.tolist()) #defined in utils.py
    y_trans = get_residual_com(scores+10,x)
    return y_trans


#build cnn_stl detector using the trained model above, i.e., cnn_stl_1.h5 (save weights of cnn_stl)
def cnn_stl_detector(y,detection_threshold,weight_dir):
    model = cnn_stl_structure() 
    model.load_weights(weight_dir)
    
    batch_sample, batch_label = split_sequence(list(y))
    batch_sample = np.expand_dims(batch_sample, axis=2)
    
    y_pred = model.predict(batch_sample, verbose=1)
    y_pred = y_pred.reshape(y[window:].shape)
    y_true = y[window:].astype('float32')
    dist = (y_true-y_pred)*(y_true-y_pred) #distance or difference between the predicted and the truth
    
    anomalies = np.where(dist>detection_threshold)[0]+window #get an array of anomaly indices
    return anomalies

def cnn_structure():
    model = Sequential()
    model.add(Conv1D(filters=num_filt_1,
                     kernel_size=kernel_size,
                     strides=conv_strides,
                     padding='valid',
                     activation='relu',
                     input_shape=(window, n_features)))
    model.add(MaxPooling1D(pool_size=pool_size_1)) 
    model.add(Conv1D(filters=num_filt_2,
                     kernel_size=kernel_size,
                     strides=conv_strides,
                     padding='valid',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size_2))
    model.add(Flatten())
    model.add(Dense(units=num_nrn_dl, activation='relu')) 
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=num_nrn_ol))
    return model


def cnn_stl_sr(y,x,detection_threshold,weight_dir):
    """Parameters
    ----------
    y: 1d array, dtype=float 
        time-series sequence 
    x: DatetimeIndex, dtype='datetime64[ns]'
        Datetime indices of elements of y
    detection_threshold: dtype=float
        threshold for distance between y true and y predicted  
    weight_dir: string
        directory of saved CNN model
        
    Returns
    -------
    anomalies: 1d array, dtype=int 
         indices of detected anomalies

    """   
    #apply STL_SR transform to input sequence y
    y = stl_sr_transform(y,x)
    
    #create and load CNN model
    model = cnn_structure()
    model.load_weights(weight_dir)
    
    #split y into batches
    batch_sample, batch_label = split_sequence(list(y))
    batch_sample = np.expand_dims(batch_sample, axis=2)
    
    #prediction and anomaly estimation
    y_pred = model.predict(batch_sample, verbose=1)
    y_pred = y_pred.reshape(y[window:].shape)
    y_true = y[window:].astype('float32')
    dist = (y_true-y_pred)*(y_true-y_pred)
    anomalies = np.where(dist>detection_threshold)[0]+window
    
    return anomalies

#moving median detector
def moving_median(dataset,window_size,detection_threshold):
        """
        Proposed detection of pointwise anomalies in timeseries

        Parameters
        ----------
        dataset : 1d array, dims=(N,) ,dtype=float
            original raw timeseries
        Returns
        -------
        anomalies: 1d array, dtype=int
            1d array of anomaly indices
        """
        #not use z_normalization
        def transform(dataset):
            return sliding_window(dataset,window_size,None) 
        
        windowed = transform(dataset)
        anomalies = np.empty(0)
        for i in range(windowed.shape[0]):
            window = windowed[i,:] 
            #using median moving instead of average moving
            anoms = np.where(np.absolute(window) > detection_threshold*np.median(window))[0] + i 
            anomalies = np.concatenate((anomalies,anoms),axis=0)

        anomalies = np.unique(anomalies)
        return np.int64(anomalies)
    
def moving_average(dataset,window_size,detection_threshold):
        """
        Detection of pointwise anomalies in timeseries

        Parameters
        ----------
        dataset : 1d array, dims=(N,) ,dtype=float
            original raw timeseries
        Returns
        -------
        anomalies: 1d array, dtype=int
            1d array of anomaly indices
        """
        def transform(dataset):
            return sliding_window(dataset,window_size,z_normalization)
        
        windowed = transform(dataset)
        anomalies = np.empty(0)
        for i in range(windowed.shape[0]): 
            window = windowed[i,:]
            anoms = np.where(np.absolute(window) > detection_threshold)[0] + i
            #anoms = np.where(np.absolute(window) > np.mean(window)+detection_threshold*np.std(window))[0] + i
            anomalies = np.concatenate((anomalies,anoms),axis=0)

        anomalies = np.unique(anomalies)
        return np.int64(anomalies)
    
def sliding_window(dataset,window_size,normalization=None):
    """
    Sliding window transformation with optional per-window normalization

    Parameters
    ----------
    dataset : 1d array, dims = (N,)
        input 1d dataset to transform
    window_size : int
        size of sliding window
    normalization : func
        normalization to apply to each window

    Returns
    -------
    windows : 2d array, dims = (N-window_size,window_size)
    """
    window_size = int(window_size)

    assert(window_size >= 1)
    N_windows = dataset.shape[0] - window_size #8349
    windows = np.empty((N_windows,window_size)) # zero matrix 8349x100
    windows[:] = np.nan # nan matrix 8349x100
    for i in range(N_windows):
        arr = dataset[i:i+window_size] # (100,)
        if normalization is None:
          windows[i,:] = arr
        else:
            windows[i,:] = normalization(arr)
    return windows

def basic_imputation(ys):
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

#linear impute between two points
def linear_impute(y_start,y_end,npoints):
    m = (y_end - y_start)/(npoints)
    c = y_start
    values = [m*x + c for x in range(1,npoints)]
    return values