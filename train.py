from dataset.dataset import FREDtest, FREDtrain
import argparse
from models.cnnAD import CNNAnomalyDetector

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_time', type=str, required=True)
    parser.add_argument('--symbol', type=str, required=True)
    
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
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
    
    train_data = FREDtrain(opt)
    batch_sample, batch_label = train_data.load_data(opt)
    # print dtype of batch_sample and batch_label
    print(batch_sample.dtype)
    model = CNNAnomalyDetector(opt)
    model.train(batch_sample=batch_sample, batch_label=batch_label, opt=opt)

if __name__ == '__main__':
    main()