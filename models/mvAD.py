import numpy as np

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