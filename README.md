<h1 align="center">
    Phân tích dữ liệu giá dầu thô
</h1>

<div align="center">
  
  <a href="https://github.com/Phenikaa-University/Analysis-of-crude-oil-data">![link to main GitHub showing Stars number](https://img.shields.io/github/stars/Phenikaa-University/Analysis-of-crude-oil-data?style=social)</a>
  <a href="https://github.com/Phenikaa-University/Analysis-of-crude-oil-data">![link to main GitHub showing Forks number](https://img.shields.io/github/forks/Phenikaa-University/Analysis-of-crude-oil-data?style=social)</a>
  <a href="https://twitter.com/cngvng413">![X (formerly Twitter) URL](https://img.shields.io/twitter/follow/cngvng413)</a>
 
</div>

<p align="center">
    <br>
    <img src="plot/results/cnn_stl/DCOILBRENTEU_detected_anomalies_cnn_stl_sr.png">
    <br>
<p>

Báo cáo thị trường dầu thô là một nguồn thông tin quan trọng về thị trường dầu thô. Báo cáo này cung cấp thông tin tổng quan về thị trường, giúp dự đoán xu hướng giá thị trường dầu thô trong tương lai, đánh giá các yếu tố rủi ro và hỗ trợ các nhà đầu tư trong việc đưa ra quyết định. Trong bài tập lớn này, chúng tôi tập trung vào việc phân tích dữ liệu giá dầu thô dựa trên cơ sở dữ liệu FRED để đưa ra các cách phân tích và đánh giá đồng thời dự đoán các yếu tố tác động gây nên sự bất thường trong biến đổi giá cả của thị trường dầu mỏ.


## Author

<div align="center">

|  Họ và tên | MSSV | Lớp |
| -------- | -------- | -------- |
| [Vương Tuấn Cường](https://cngvng.github.io/)  | 21011490    | K15-KHMT    |

</div>

## Requirements

```
    conda create -n ai-dev python=3.8.16
```

```
    pip install -r requirements.txt
```

## Demo

```
    streamlit run st.py
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