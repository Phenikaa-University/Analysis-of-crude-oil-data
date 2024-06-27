import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st

from models.utils import *
from dataset.utils import generate_point_outliers, exact_detection_function, precision, recall, f_beta_measure

st.set_page_config("Anomaly Detection Demo", "📈")

# Typing input data in streamlit
st.title("Anomaly Detection Demo")
def get_input_st():
    symbol = st.selectbox("Select the symbol", ["DCOILBRENTEU", "DHOILNYH", "OVXCLS", "DGASNYH"])
    date_time = st.text_input("Enter the start date in the format YYYY-MM-DD", "1880-01-01")
    mode = st.selectbox("Select the mode", ["cnn_stl_sr", "ma_stl_sr", "mm_stl_sr"])
    threshold = st.number_input("Enter the detection threshold", 0.0, 10.0, 0.5)
    window_size = st.selectbox("Select the window size", [100, 200, 300, 400, 500])
    return symbol, date_time, mode, threshold, window_size

def main():
    symbol, date_time, model, threshold, window_size = get_input_st()
    window_size = int(window_size)
    # Tạo một nút để chạy phân tích
    if st.button("Detect Anomalies"):

        xs_test, ys_test = get_fred_dataset(symbol=symbol, date_time=date_time)
        ys_test_imputed = basic_imputation(ys_test)
        
        ys_corrputed, positions = generate_point_outliers(
            raw_data=ys_test_imputed,
            anomaly_fraction=0.005,
            window_size=window_size,
            pointwise_deviation=3.5,
            rng_seed=2
        )
        ys_corr_res = get_residual_com(ys_corrputed, xs_test)

        # Phát hiện dị thường dựa trên model đã chọn
        if model == "cnn_stl_sr":
            detected = cnn_stl_sr(ys_corr_res, x=xs_test, detection_threshold=threshold, weight_dir="checkpoints/cnn_stl_sr.weights.h5")
        elif model == "ma_stl_sr":
            detected = moving_median(ys_corr_res, window_size, detection_threshold=threshold)
        elif model == "mm_stl_sr":
            detected = moving_average(ys_corr_res, window_size, detection_threshold=threshold)

        # Tính toán và hiển thị các chỉ số đánh giá
        tp, fp, fn, tn = exact_detection_function(detected=detected, truth=positions)
        st.write("Evaluation Metrics:")
        st.write(f"Precision: {precision(tp, fp, fn):.4f}")
        st.write(f"Recall: {recall(tp, fp, fn):.4f}")
        st.write(f"F1-score: {f_beta_measure(tp, fp, fn, beta=1):.4f}")

        # Vẽ và hiển thị biểu đồ
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(ys_corrputed, label="corrupted data")
        ax.plot(positions, ys_corrputed[positions], 'rx', markersize=8, label="true anomalies")
        ax.plot(detected, ys_corrputed[detected], 'ko', markersize=4, label="detected anomalies")
        ax.legend()
        ax.set_title(f"Anomaly Detection using {model} on {symbol} dataset")
        plt.savefig(f"plot/results/{model}_{symbol}.png")
        st.pyplot(fig)

if __name__ == "__main__":
    main()