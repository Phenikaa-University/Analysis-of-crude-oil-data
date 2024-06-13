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
    <img src="plot/results/cnn_stl_sr/DHOILNYHdetected_anomalies_cnn_stl_sr.png">
    <br>
<p>

The final exam is designed to expand on the concepts covered in the midterm exam and provide a deeper understanding of the subject matter. Specifically, we plan to introduce a Convolutional Neural Network (CNN) model, which will replace the traditional Neural Network (NN) model used in the midterm exam. The CNN model is a powerful tool that is commonly used in image processing and has shown promising results in a variety of applications. Additionally, we will incorporate the Spectral Residual (SR) method to enhance the data preprocessing phase of the model. This method has been demonstrated to effectively remove noise and artifacts from images, resulting in improved accuracy and performances.


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