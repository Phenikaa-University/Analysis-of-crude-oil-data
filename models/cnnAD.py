"""Building CNN model for anomaly detection"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout # type: ignore

class CNNAnomalyDetector:
    def __init__(self, opt):
        super().__init__()
        self.window = opt.window_size
        self.n_features = opt.n_features
        self.num_filt_1 = opt.num_filt_1
        self.num_filt_2 = opt.num_filt_2
        self.num_nrn_dl = opt.num_nrn_dl
        self.num_nrn_ol = opt.num_nrn_ol
        self.kernel_size = opt.kernel_size
        self.conv_strides = opt.conv_strides
        self.pool_size_1 = opt.pool_size_1
        self.pool_size_2 = opt.pool_size_2
        self.dropout_rate = opt.dropout_rate
        self.model = self._build_cnn_model()
        self.loss = []

    def _build_cnn_model(self):
        model = Sequential()
        model.add(Conv1D(filters=self.num_filt_1,
                         kernel_size=self.kernel_size,
                         strides=self.conv_strides,
                         padding='valid',
                         activation='relu',
                         input_shape=(self.window, self.n_features)))
        model.add(MaxPooling1D(pool_size=self.pool_size_1))
        model.add(Conv1D(filters=self.num_filt_2,
                         kernel_size=self.kernel_size,
                         strides=self.conv_strides,
                         padding='valid',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=self.pool_size_2))
        model.add(Flatten())
        model.add(Dense(units=self.num_nrn_dl, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(units=self.num_nrn_ol))
        return model

    def train(self, batch_sample, batch_label, opt):
        optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr)
        self.model.compile(optimizer=optimizer, loss='mean_absolute_error')
        self.model.fit(batch_sample, batch_label, epochs=opt.num_epochs, batch_size=opt.batch_size)
        if opt.mode == 'stl_sr':
            self.model.save_weights(f'{opt.checkpoint_dir}cnn_stl_sr.weights.h5')
        elif opt.mode == "stl":
            self.model.save_weights(f'{opt.checkpoint_dir}cnn_sr.weights.h5')
    def predict(self, batch_sample):
        return self.model.predict(batch_sample, verbose=1)
    
    def detect_anomaly(self, ys_cor, batch_sample, threshold):
        prediction = self.predict(batch_sample)
        y_pred = prediction.reshape(ys_cor[self.window:].shape)
        y_true = ys_cor[self.window:].astype('float32')
        dist = (y_true - y_pred) ** 2
        anomalies = np.where(dist > threshold)[0] * self.window
        return anomalies
        
    def load_weights(self, path):
        self.model.load_weights(path)
        
# build CNN model 
def cnn_structure(opt):
    model = Sequential()
    model.add(Conv1D(filters=opt.num_filt_1,
                     kernel_size=opt.kernel_size,
                     strides=opt.conv_strides,
                     padding='valid',
                     activation='relu',
                     input_shape=(opt.window_size, opt.n_features)))
    model.add(MaxPooling1D(pool_size=opt.pool_size_1)) 
    model.add(Conv1D(filters=opt.num_filt_2,
                     kernel_size=opt.kernel_size,
                     strides=opt.conv_strides,
                     padding='valid',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=opt.pool_size_2))
    model.add(Flatten())
    model.add(Dense(units=opt.num_nrn_dl, activation='relu')) 
    model.add(Dropout(opt.dropout_rate))
    model.add(Dense(units=opt.num_nrn_ol))
    return model