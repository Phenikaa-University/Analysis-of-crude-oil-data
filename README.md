# Analysis-of-crude-oil-data
Đề tài bài tập lớn môn Lập trình AI.

<p align="center">
    <br>
    <img src="results/cnn_stl_sr/DHOILNYHdetected_anomalies_cnn_stl_sr.png">
    <br>
<p>

We present a novel approach for anomaly detection in time-series data by combining three modern models, including Convolutional Neural Network (CNN), Seasonal and Trend decomposition using Loess (STL), and Spectral Residual (SR). By preprocessing the data using the STL and SR models, we extract the important points in the time-series and feed them into the CNN model to detect anomalies. This enables us to effectively identify anomalies within the data set.

## Author

[Vuong Tuan Cuong](https://cngvng.github.io/) - ID: 21011490 

## Requirements

```
    conda create -n ai-dev python=3.8.16
```

```
    pip install -r requirements.txt
```

## Inference 

The checkpoints from folder `checkpoints`:
- For model CNN_STL saved in `checkpoints/cnn_stl.weights.h5`:

```
    bash scripts/test_cnn_stl.sh
```

- For model CNN_STL_SR saved in `checkpoints/cnn_stl_sr.weights.h5`

```
    bash scripts/test_cnn_stl_sr.sh
```

## Training

- For model CNN_STL:

```
    bash scripts/train_cnn_stl.sh
```

- For model CNN_STL_SR:

```
    bash scripts/train_cnn_stl_sr.sh
```