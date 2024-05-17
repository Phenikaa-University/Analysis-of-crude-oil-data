import numpy as np 
import matplotlib.pyplot as plt 
import cvxpy as cvx

EPS = 1e-8
score_window = 100
mag_window = 3
threshold = 3
look_ahead=5
extend_num=5

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

    """
    detected = np.unique(detected) 
    truth = np.unique(truth)
   
    true_positives = set(detected).intersection(set(truth))
    false_positives = set(detected).difference(set(truth))
    false_negatives = set(truth).difference(set(detected))

    tp = len(true_positives) # 36
    fp = len(false_positives) # 21
    fn = len(false_negatives) # 7

    return tp,fp,fn


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

def difference(dataset):
    """
    First order differencing of 1D data
    
    Parameters
    ----------
    dataset : 1d array, dtype=float, dims = (N,)

    Returns
    -------
    diff : 1d array, dtype = float, dims(N-1,)
    """
    diff = dataset[1:] - dataset[0:-1]
    assert(dataset.shape[0]-1 == diff.shape[0])

    return diff

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

def extend_series(values, extend_num=5, look_ahead=5):

    if look_ahead < 1:
        raise ValueError('look_ahead must be at least 1')

    extension = [predict_next(values[-look_ahead - 2:-1])] * extend_num
    return values + extension

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

def _makearray(a):
    new = np.asarray(a)
    wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
    return new, wrap

def compress(indices,dist =1):
    indices,_ = _makearray(indices)
    inds = [indices[0]]

    for i in range(1,indices.shape[0]):
        if np.absolute(indices[i]- indices[i-1]) > dist:
            inds.append(indices[i])
    inds = np.array(inds)
    return inds

def denoised_rescaling(signal,weight=50.0):
    denoised_signal,_,_,f1,f2 = convex_smooth(signal=signal,weight=weight,objective_type="quadratic",normalise=False)
    output = signal/denoised_signal
    return output,denoised_signal,f1,f2


def convex_smooth(signal,weight, objective_type="quadratic",normalise = True):
    """
    Smoothing signal based on weight parameter
       

    Parameters
    ----------

    signal : 1d array
        noisy signal to be smoothed
    weight : float
        regularization term weight
        
    objective_type : string
        determines type of regularization objective [values: quadratic,total_variation] 
    

    Returns
    -------

    x_out : 1d array, dtype = float
        smoothed data
    problem_value : float
        final value of objective function
    problem_status : string
        status of the problem solution, indicates whether optimization converged or failed
    """
    print("setting up cvx problem")
    signal = np.array(signal,dtype=float)
    assert(weight >= 0)
    signal_max = np.max(signal)

    #max value normalization
    if normalise==True:
        signal=signal/signal_max

    dims = signal.shape[0]
    x = cvx.Variable(dims)

    #create difference matrix
    D = np.identity(dims, int)
    for i in range(1,dims):
        D[i][i-1] = -1

    #introduce regularization objective
    if objective_type == "total_variation":
        f2 = cvx.norm(D*x,1)
    elif objective_type == "quadratic":
        f2 = cvx.sum_squares(D*x)
    else:
        raise ValueError("Only allowed values: [ total_variation | quadratic ]")
    
    #define primary objective
    f1 = cvx.sum_squares(x-signal)
    #scalarize multi-objective problem
    objective = cvx.Minimize(f1 + weight*f2)

    #solve
    prob = cvx.Problem(objective)
    prob.solve()
    x_out =  np.asarray(x.value)
    x_out = x_out.reshape(signal.shape)
    
    if normalise==True: 
        x_out=x_out*signal_max

    #compute both objectives:
    f1_value = np.sum(np.power(x_out - signal,2))
    if objective_type == "quadratic":
        f2_value = np.sum(np.power(x_out[1:] - x_out[0:-1],2))
    elif objective_type == "total_variation":
        f2_value = np.sum(np.absolute(x_out[1:] - x_out[0:-1]))
    
    #return results
    problem_value = prob.value 
    problem_status = prob.status
    return x_out, problem_value, problem_status,f1_value,f2_value

def tk_transform(dataset):

    """
    step 1: perform z-normalization of differenced dataset.
        In this way we try to amplify effect of pointwise noise
    """
    anomalous_detrended = z_normalization(difference(dataset))
    """
    step 2: We know that pointwise discontinuity will produce two large differences with values either side of it
        These will be of opposite sign, so take absolute value. Then run a window of length 2 across dataset
        If discontinuity is "high" relative to other fluctuations then it's amplitude grows faster than for 
        non-anomalous points, lifting anomalies higher than inliers and enabled thresholding.

        This assumes data is normally distributed and variance is stationary in time.

    """
    anomalous_convolved = np.convolve([0.5,0.5],np.absolute(anomalous_detrended))
    
    """
    step 3: trying to remove effects of time-varying variance by dividing through by smoothed value of differences
    """
    anomalous_thresholdable,_,_,_ = denoised_rescaling(anomalous_convolved,weight=160)
    return anomalous_thresholdable

def threshold_anomaly_detector(signal,threshold):
    """
    Performs global thresholding of signal. 
    Remove consecutive values above threshold as we assume anomalies are pointwise, independent and uncorrelated.
    """
    indices = np.where(signal > threshold)[0]

    indices = compress(indices)
    return indices

def z_normalization(dataset): # for each (100,) input
    """
    Z normalization

    General form: (xs - loc(xs))/scale
    Where:
        loc(x) = mean(x)
        scale(x) = std(x)
    Parameters
    ----------
    dataset: 1d array, dtype=float, dims=(N,)
    
    Returns
    -------
    normed : 1d array, dtype =float, dims=(N,)
    """ 

    loc = np.mean(dataset)
    scale = np.std(dataset)
    znormed = (dataset - loc)/scale
    return znormed