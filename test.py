from dataset.dataset import *
from dataset.utils import *
import argparse
from models.cnnAD import cnn_structure
from models.mvAD import moving_average, moving_median
from utils import *
import os

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_time', type=str, required=True)
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--model', type=str, default='mm')
    

    parser.add_argument('--detection_threshold', type=float, default=0.085)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--mode', type=str, default='stl_sr')
    
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--slide_window', type=int, default=20)
    parser.add_argument('--pred_window', type=int, default=1)
    
    parser.add_argument('--n_features', type=int, default=1)
    parser.add_argument('--num_filt_1', type=int, default=16)
    parser.add_argument('--num_filt_2', type=int, default=16)
    parser.add_argument('--num_nrn_dl', type=int, default=20)
    parser.add_argument('--num_nrn_ol', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--conv_strides', type=int, default=1)
    parser.add_argument('--pool_size_1', type=int, default=2)
    parser.add_argument('--pool_size_2', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    
    
    opt = parser.parse_args()
    return opt

def cnn_stl_detector(opt, y,detection_threshold):
    model = cnn_structure(opt) 
    model.load_weights("./checkpoints/cnn_sr.weights.h5")
    
    batch_sample, batch_label = split_sequence(opt, list(y))
    batch_sample = np.expand_dims(batch_sample, axis=2)
    
    y_pred = model.predict(batch_sample, verbose=1)
    y_pred = y_pred.reshape(y[opt.window_size:].shape)
    y_true = y[opt.window_size:].astype('float32')
    dist = (y_true-y_pred)*(y_true-y_pred) #distance or difference between the predicted and the truth
    
    anomalies = np.where(dist>detection_threshold)[0]+opt.window_size #get an array of anomaly indices
    return anomalies

def cnn_stl_sr(opt, y, x, detection_threshold):
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
    y = stl_sr_transform(opt, y,x)
    
    #create and load CNN model
    model = cnn_structure(opt)
    model.load_weights("./checkpoints/cnn_stl_sr.weights.h5")
    
    #split y into batches
    batch_sample, batch_label = split_sequence(opt, list(y))
    batch_sample = np.expand_dims(batch_sample, axis=2)
    
    #prediction and anomaly estimation
    y_pred = model.predict(batch_sample, verbose=1)
    y_pred = y_pred.reshape(y[opt.window_size:].shape)
    y_true = y[opt.window_size:].astype('float32')
    dist = (y_true-y_pred)*(y_true-y_pred)
    anomalies = np.where(dist>detection_threshold)[0]+opt.window_size
    
    return anomalies 
    
def main():
    opt = get_opt()
    print(opt)
    
    """Testing with corrupted data that contains anomalies"""
    #collect test dataset which can be train data corrupted with anomalies, or any totally new corrupted data
    #No. 2 Heating Oil Prices: New York Harbor ------------------------------------------(DHOILNYH)
    #CBOE Crude Oil ETF Volatility Index ------------------------------------------------(OVXCLS)
    #Conventional Gasoline Prices: New York Harbor, Regular -----------------------------(DGASNYH)
    
    xs_test, ys_test = get_fred_dataset(opt)
    ys_test_imputed = basic_imputation(ys_test)
    
    ys_corrputed, positions = generate_point_outliers(
        raw_data=ys_test_imputed,
        anomaly_fraction=0.005,
        window_size=opt.window_size,
        pointwise_deviation=3.5,
        rng_seed=2
    )
    
    # Get residual from y_corrputed which will be used as input sequence for the model
    ys_corr_res = get_residual_com(opt, ys_corrputed, xs_test)
    
    '''Detect anomalies with cnn_stl and print results, for one test only - rng_seed=0'''

    print(f"========= {opt.symbol}"+ " dataset ================")
    # Load model
    if opt.mode == "stl":
        thresholds = np.linspace(0.005, 0.025, 10)
        tprs = []
        fprs = []
        for threshold in thresholds:
            detected = cnn_stl_detector(opt, ys_corr_res,threshold)
            tp,fp,fn, tn = exact_detection_function(detected=detected,truth=positions)
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            tprs.append(tp)
            fprs.append(fp)
        """Visualize the ROC curve with cnn_stl detector"""
        plt.plot(fprs, tprs)
        for i, threshold in enumerate(thresholds):
            plt.scatter(fprs[i], tprs[i], label=f'threshold: {threshold:.5f}', s=10)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve {opt.symbol} dataset")
        plt.legend(loc='lower right')
            
    if opt.mode == "stl_sr":
        thresholds = np.linspace(0.05, 0.1, 10)
        tprs = []
        fprs = []
        for threshold in thresholds:
            detected = cnn_stl_sr(opt, ys_corr_res, xs_test,threshold)
            tp,fp,fn, tn = exact_detection_function(detected=detected,truth=positions)
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            tprs.append(tp)
            fprs.append(fp)
        """Visualize the ROC curve with cnn_stl detector"""
        plt.plot(fprs, tprs)
        for i, threshold in enumerate(thresholds):
            plt.scatter(fprs[i], tprs[i], label=f'threshold: {threshold:.5f}', s=10)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve {opt.symbol} dataset")
        plt.legend(loc='lower right')
    
    if opt.mode == "stl":
        save_path = "plot/results/cnn_stl/"
        detected = cnn_stl_detector(opt, ys_corr_res,detection_threshold=0.01167)
    elif opt.mode == "stl_sr":
        save_path = "plot/results/cnn_stl_sr/"
        detected = cnn_stl_detector(opt, ys_corr_res,detection_threshold=0.06667)
    elif opt.mode == "ma":
        save_path = "plot/results/moving_average/"
        detected = moving_average(ys_corrputed, opt.window_size, detection_threshold=3)
    elif opt.mode == "mm":
        save_path = "plot/results/moving_median/"
        detected = moving_median(ys_corrputed, opt.window_size, detection_threshold=3)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f"{save_path}{opt.symbol}_ROC_curve.png")
    tp,fp,fn, tn = exact_detection_function(detected=detected,truth=positions)
    display_metrics(tp,fp,fn)
    """Visualize the detected animalies with cnn_stl detector"""
    
    fig,ax = plt.subplots(1,figsize=(12,9))
    plt.plot(ys_corrputed,label="corrupted data")
    plt.plot(positions,ys_corrputed[positions],'rx', markersize=8, label="true anomalies")
    plt.plot(detected,ys_corrputed[detected],'ko', markersize=4, label="detected anomalies")
    plt.legend()
    if opt.mode == "stl":
        plt.title(f"Detects anomalies using {opt.symbol} dataset with CNN_STL")
        plt.savefig(f"plot/results/cnn_stl/{opt.symbol}_detected_anomalies_cnn_stl.png")
    elif opt.mode == "stl_sr":
        plt.title(f"Detects anomalies using {opt.symbol} dataset with CNN_STL_SR")
        plt.savefig(f"plot/results/cnn_stl/{opt.symbol}_detected_anomalies_cnn_stl_sr.png")
    elif opt.mode == "ma":
        plt.title(f"Detects anomalies using {opt.symbol} dataset with Moving Average")
        plt.savefig(f"plot/results/moving_average/{opt.symbol}_detected_anomalies_moving_average.png")
    elif opt.mode == "mm":
        plt.title(f"Detects anomalies using {opt.symbol} dataset with Moving Median")
        plt.savefig(f"plot/results/moving_median/{opt.symbol}_detected_anomalies_moving_median.png")


if __name__ == '__main__':
    main()