"""Building CNN model for anomaly detection"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout # type: ignore

class CNNAnomalyDetector:
    def __init__(self, window, n_features, num_filt_1, num_filt_2, num_nrn_dl, num_nrn_ol,
                 kernel_size, conv_strides, pool_size_1, pool_size_2, dropout_rate):
        self.window = window
        self.n_features = n_features
        self.num_filt_1 = num_filt_1
        self.num_filt_2 = num_filt_2
        self.num_nrn_dl = num_nrn_dl
        self.num_nrn_ol = num_nrn_ol
        self.kernel_size = kernel_size
        self.conv_strides = conv_strides
        self.pool_size_1 = pool_size_1
        self.pool_size_2 = pool_size_2
        self.dropout_rate = dropout_rate
        self.model = self._build_cnn_model()

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

    def train(self, batch_sample, batch_label, num_epochs, batch_size, learning_rate, save_path, stl_sr=True):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_absolute_error')
        self.model.fit(batch_sample, batch_label, epochs=num_epochs, batch_size=batch_size)
        if stl_sr:
            self.model.save_weights(f'{save_path}/cnn_stl_sr.weights.h5')
        else:
            self.model.save_weights(f'{save_path}/cnn_sr.weights.h5')
    
    def predict(self, batch_sample):
        return self.model.predict(batch_sample, verbose=1)
    
    def detect_anomaly(self, batch_sample, threshold):
        prediction = self.predict(batch_sample)
        y_pred = prediction.reshape(prediction[self.window:].shape)
        y_true = prediction[self.window:].astype('float32')
        dist = (y_true - y_pred) ** 2
        anomalies = np.where(dist > threshold)[0] * self.window
        return anomalies
    
    
        
    