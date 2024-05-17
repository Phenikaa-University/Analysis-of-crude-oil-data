from dataset.dataset import FREDtest, FREDtrain
import argparse
from models.cnnAD import CNNAnomalyDetector
from models.mvAD import moving_average, moving_median
from utils import *

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_time', type=str, required=True)
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--model', type=str, default='mm')
    

    parser.add_argument('--detection_threshold', type=float, default=0.085)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
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
    
    
    opt = parser.parse_args()
    return opt
    
def main():
    opt = get_opt()
    print(opt)
    
    test_data = FREDtest(opt)
    batch_sample, batch_label, ys_corr, positions = test_data.load_data(opt)
    
    if opt.model == "cnn":
        model = CNNAnomalyDetector(opt)
        model.load_weights("checkpoints/cnn_stl_sr.weights.h5")
        detected = model.detect_anomaly(ys_corr ,batch_sample, opt.detection_threshold)
    elif opt.model == "ma":
        detected = moving_average(ys_corr, window_size=opt.window_size, detection_threshold=opt.detection_threshold)
    elif opt.model == "mm":
        detected = moving_median(ys_corr, window_size=opt.window_size, detection_threshold=opt.detection_threshold)
    tp,fp,fn = exact_detection_function(detected=detected,truth=positions)
    print("true positives:",tp,"false positives:",fp,"false_negatives:",fn)
    print("precision:",precision(tp=tp,fp=fp,fn=fn))
    print("recall:",recall(tp=tp,fp=fp,fn=fn))
    print("f_beta_measure",f_beta_measure(tp=tp,fp=fp,fn=fn,beta=1))

if __name__ == '__main__':
    main()