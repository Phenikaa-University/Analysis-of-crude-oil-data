from dataset.dataset import *
from dataset.utils import *
import argparse
from models.cnnAD import cnn_structure
from models.mvAD import moving_average, moving_median
from utils import *
import os
import streamlit as st
from test import cnn_stl_detector, cnn_stl_sr


symbol_test = st.sidebar.selectbox("Input Symbol example",  
                       ("DCOILBRENTEU", "DHOILNYH", "OVXCLS", "GVZCLS", "VIXCLS"))


detected_type = st.sidebar.selectbox("select model", ("cnn_stl_detector", "cnn_stl_sr", "moving_average"))

def main():
  
    st.title("Anomaly detection in time series")
    model()


def model():
    date_time_test = "1800-1-1" 
    xs_test,ys_test = get_fred_dataset(symbol_test,date_time_test)
    ys_test_imputed = basic_imputation(ys_test)
    
    #add anomalies to imputed test data to obtain corrupted data which is used for testing
    ys_corrupted,positions = generate_point_outliers(raw_data=ys_test_imputed,
        anomaly_fraction=0.005,
        window_size=100,
        pointwise_deviation=3.5, 
        rng_seed=2)
    
    #get residual from y_corrupted which will be used as input sequence of detector
    ys_corr_res = get_residual_com(ys_corrupted,xs_test)
    
    

    #detection_threshold is selected in [0.005,0.025]
    
    if detected_type == "cnn_stl_detector" :
        #load weight directory for cnn_stl model
        weight_dir = './saved_models/cnn_stl_1.h5'
        detected = cnn_stl_detector(ys_corr_res,0.045,weight_dir)
    elif detected_type == "cnn_stl_sr":
        #load weight directory for cnn_stl model
        weight_dir = './saved_models/cnn_stl_9.h5'
        detected = cnn_stl_sr(ys_corrupted, xs_test,0.08,weight_dir)
    else:
        detected = moving_average(ys_corrupted,100,3)
    
    tp,fp,fn = exact_detection_function(detected=detected,truth=positions)
    st.write("----------------------------------------------------------------")
    st.write("true positives:",tp,"| false positives:",fp,"| false_negatives:",fn)
    st.write("----------------------------------------------------------------")
    st.write("precision:",precision(tp=tp,fp=fp,fn=fn))
    st.write("recall:",recall(tp=tp,fp=fp,fn=fn))
    st.write("f1:",f_beta_measure(tp=tp,fp=fp,fn=fn,beta=1))
    st.write("----------------------------------------------------------------")
    
    
    
    
    st.write("Visualize the detected anomalies")
    fig,ax = plt.subplots(1,figsize=(12,9))
    plt.plot(ys_corrupted,label="corrupted data")
    plt.plot(positions,ys_corrupted[positions],'rx', markersize=8, label="true anomalies")
    plt.plot(detected,ys_corrupted[detected],'ko', markersize=4, label="detected anomalies")
    st.pyplot(fig)

main()