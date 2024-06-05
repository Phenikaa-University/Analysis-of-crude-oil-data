from dataset.dataset import *
from dataset.utils import *
import argparse
from models.cnnAD import cnn_structure
from tensorflow.keras.optimizers import Adam # type: ignore

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
    
    xs, ys = get_fred_dataset(opt)
    # Fill in the missing values in the dataset with linear interpolation
    ys_imputed = basic_imputation(ys)
    '''Define training data which is normal data (without anomalies)'''
    # Get the residual component of the time series
    ys_res = get_residual_com(opt, ys_imputed, xs)
    # Split into samples
    batch_sample, batch_label = split_sequence(opt, list(ys_res))
    batch_sample = np.expand_dims(batch_sample, axis=2)
    
    model = cnn_structure(opt)
    '''Training procedure'''
    model.compile(optimizer=Adam(learning_rate=opt.lr),
                loss='mean_absolute_error')
    model_fit = model.fit(batch_sample,
                        batch_label,
                        epochs=opt.num_epochs,
                        batch_size=opt.batch_size,
                        verbose=1)
    model.save_weights(f'{opt.checkpoint_dir}cnn_sr.weights.h5')
if __name__ == '__main__':
    main()