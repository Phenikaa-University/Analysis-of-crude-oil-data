import numpy as np 
import matplotlib.pyplot as plt 
import cvxpy as cvx

EPS = 1e-8
score_window = 100
mag_window = 3
look_ahead=5
extend_num=5

def generate_point_outliers(raw_data,anomaly_fraction=0.005,window_size=100,pointwise_deviation=3.5,rng_seed=0):
    """
    Generate point anomalies in timeseries
    
    Parameters
    ----------
    raw_data : 1d array
        raw timeseries data of values
    anomaly_fraction : float
        fraction of anomalous datapoints to generate
        typical values are in the range 0.0001 (0.01%)
        to 0.01 (1%) as anomalies are rare
    
    window_size : int
        size of window over which we compute mean and stdev
        
    pointwise_deviation : float
        number of stdevs to increase/reduce anomalous point relative to segment
    rng_seed : int
        seed for random number generator to make results reproducible
        
    Returns
    -------
    anomalous_data : 1d array
        Array of timeseries corrupted by generated anomalies
    positions : 1d array
        Array of locations of the generated anomalies in the dataset 
    
    """
    np.random.seed(rng_seed)
    anomalous_data = np.copy(raw_data) #array([18.63, 18.45, 18.55, ..., 58.01, 59.13, 59.46])
    
    n = raw_data.shape[0] #8449
    n_outliers = int(np.ceil(n*anomaly_fraction)) # 43
    
    positions = np.random.randint(0,n,n_outliers)
    
    if window_size %2 == 0:
        window_size = window_size + 1 # 101 make it odd so that it can be symmetric about the point!
    
    #number of standard deviations lift point
    for pn in positions:
        window = anomalous_data[max(0,int(pn-(window_size-1)/2)):min(n,int(pn+(window_size-1)/2))] # 2682:2782 when pn=2732
        mu = np.mean(window)
        std = np.std(window)
        sign = np.random.randint(0,1)
        if sign == 0:
            anomalous_data[pn] = mu + pointwise_deviation*std
        elif sign == 1:
            anomalous_data[pn] = mu - pointwise_deviation*std
            
    return anomalous_data,positions


#linear impute between two points
def linear_impute(y_start,y_end,npoints):
    m = (y_end - y_start)/(npoints)
    c = y_start
    values = [m*x + c for x in range(1,npoints)]
    return values

#linear impute series with gaps of arbitrary size
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
        ys_imputed[int(start):int(end)] = linear
    return ys_imputed
#linear impute series with gaps of arbitrary size
def anomaly_imputation(ys,positions):
    ys_imputed = np.copy(ys)
    segments = []
    for i in range(positions.shape[0]):
        start = positions[i]
        end = start+1
        segments.append((start,end))
    for (start,end) in segments:
        y_start = ys_imputed[start-1]
        y_end = ys_imputed[end]
        linear = linear_impute(y_start=y_start,y_end=y_end,npoints=int((end+1)-start))
        ys_imputed[start:end] = linear
    return ys_imputed

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


def exact_detection_function(detected,truth):
    """
    Compute true positives (TPs), false positives (FP), false negatives (FP) for exact matches only
        using set differencing. Useful as test case
    Parameters
    ----------
    detected: 1d array, dtype=int 
        indices of dataset labelled anomalous by detector
    truth: 1d array, dtype=int,
        ground truth anomalous indices

    Returns
    -------
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    """
    detected = np.unique(detected) 
    truth = np.unique(truth)
   
    true_positives = set(detected).intersection(set(truth))
    false_positives = set(detected).difference(set(truth))
    false_negatives = set(truth).difference(set(detected))
    total_non_anomalies = len(truth) - len(true_positives)

    tp = len(true_positives) # 36
    fp = len(false_positives) # 21
    fn = len(false_negatives) # 7
    tn = total_non_anomalies - len(false_positives)

    return tp,fp,fn,tn


#define scoring metrics
def precision(tp,fp,fn):
    if tp+fp == 0:
        return 0
    return tp/(tp+fp) # 36/57

def recall(tp,fp,fn):
    if tp+fn == 0:
        return 0
    return tp/(tp+fn) # 36/43

def f_beta_measure(tp,fp,fn,beta=1):
    """
    Generalized F-measure allowing weighting of precision and recall

    beta: float, 0 <= beta <= 1
        weigthing factor
    """
    prec = precision(tp,fp,fn)
    rec = recall(tp,fp,fn)

    f_beta = ((1+beta**2) * (prec * rec))/(beta**2 * prec + rec) # ab/(a+b) if beta=1
    return f_beta


def display_scores(detected,positions):
    """
    Display precision, recall and F-beta scores of anomaly detection
    
    Parameters
    ----------
    detected: 1d array, dtype = int
        detected anomaly indices
    positions: 1d array, dtype = int
        true amomaly indices
        
    """
    tp,fp,fn = exact_detection_function(detected=detected,truth=positions)
    print("true positives:",tp,"false positives:",fp,"false_negatives:",fn)
    print("precision:",precision(tp=tp,fp=fp,fn=fn))
    print("recall:",recall(tp=tp,fp=fp,fn=fn))
    print("f_beta_measure",f_beta_measure(tp=tp,fp=fp,fn=fn,beta=1))
    

def generate_spectral_score(series):
    extended_series = extend_series(series) #len 4005
    mag = spectral_residual_transform(extended_series)[:len(series)]
    ave_mag = average_filter(mag, n=score_window)
    ave_mag[np.where(ave_mag <= EPS)] = EPS

    return abs(mag - ave_mag) / ave_mag

def spectral_residual_transform(values):
    """
    This method transform a time series into spectral residual series
    :param values: list.
        a list of float values.
    :return: mag: list.
        a list of float values as the spectral residual values
    """

    trans = np.fft.fft(values)
    mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)
    eps_index = np.where(mag <= EPS)[0]
    mag[eps_index] = EPS

    mag_log = np.log(mag)
    mag_log[eps_index] = 0

    spectral = np.exp(mag_log - average_filter(mag_log, n=mag_window))

    trans.real = trans.real * spectral / mag
    trans.imag = trans.imag * spectral / mag
    trans.real[eps_index] = 0
    trans.imag[eps_index] = 0

    wave_r = np.fft.ifft(trans)
    mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)
    return mag

def average_filter(values, n=3): #moving average
    """
    Calculate the sliding window average for the give time series.
    Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
    :param values: list.
        a list of float numbers
    :param n: int, default 3.
        window size.
    :return res: list.
        a list of value after the average_filter process. 
    """

    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res

def extend_series(values, extend_num=5, look_ahead=5):

    if look_ahead < 1:
        raise ValueError('look_ahead must be at least 1')

    extension = [predict_next(values[-look_ahead - 2:-1])] * extend_num
    return values + extension

def predict_next(values):
    """
    Predicts the next value by sum up the slope of the last value with previous values.
    Mathematically, g = 1/m * sum_{i=1}^{m} g(x_n, x_{n-i}), x_{n+1} = x_{n-m+1} + g * m,
    where g(x_i,x_j) = (x_i - x_j) / (i - j), n=m=5??
    :param values: list.
        a list of float numbers.
    :return : float.
        the predicted next value.
    """

    if len(values) <= 1:
        raise ValueError(f'data should contain at least 2 numbers')

    v_last = values[-1]
    n = len(values) 
    
    slopes = [(v_last - v) / (n - 1 - i) for i, v in enumerate(values[:-1])]

    return values[1] + sum(slopes) #? values[0] + sum(slopes) or just use values[1:5]