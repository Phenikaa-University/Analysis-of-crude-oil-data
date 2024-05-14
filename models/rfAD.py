"""Create Random Forest model for Anomaly Detection"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RFAnomalyDetector:
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)
        self.score = []

    def train(self, batch_sample, batch_label):
        n_samples, n_time_steps, n_features = batch_sample.shape
        batch_sample_2d = batch_sample.reshape(n_samples, n_time_steps * n_features)
        self.model.fit(batch_sample_2d, batch_label)
    
    def predict(self, batch_sample):
        n_samples, n_time_steps, n_features = batch_sample.shape
        batch_sample_2d = batch_sample.reshape(n_samples, n_time_steps * n_features)
        return self.model.predict(batch_sample_2d)
    
    def detect_anomaly(self, batch_sample, threshold):
        prediction = self.predict(batch_sample)
        y_pred = prediction.reshape(prediction.shape)
        y_true = batch_sample.astype('float32')
        dist = (y_true - y_pred) ** 2
        anomalies = np.where(dist > threshold)[0]
        return anomalies