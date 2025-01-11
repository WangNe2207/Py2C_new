




# Py2C

This is the official Py2C tool of paper "High-Level Synthesis from Python to RTL for Deep
Neural Networks on SoC System"

## Version Requirements  
- Tensorflow = 2.10  
- Keras = 2.10  
## User manual  

## Dataset Preparing  

### Image
1. 3D(Cifar10, ImageNet):
Print the values ​​of the image from left to right, top to bottom and red channel to blue channel

3. 1D(Fashion MNIST)

## Inference

1. Modify value in file **config.yaml**:
   - test_ckpt_path: path to checkpoint used to test
2. We release [checkpoints](https://drive.google.com/drive/folders/1YfN9upk4ZPwADSsbL5BjUvl3GYzrVBux?usp=sharing) corresponding to each classification task in the PTB-XL dataset reported in the paper.
3. Run file **test.py** with specific arguments to inference:
   - exp_type: experiment type (selected from **["super", "sub", "rhythm", "all", "diag", "form", "cpsc"]**)
   - Note: the value of exp_type must correspond to the checkpoint.
   - Example: to test model for "SUPER" task in PTB-XL dataset, run the command
   ```commandline
   python test.py --exp_type super
   ```

## Citation

HƯỚNG DẪN SỬ DỤNG PY2C:

bước 1: ở main.py tại dòng: pyc_lib = Py2C(".....h5"), hãy điền vào phần .... file h5 của bạn 

bước 2: chính cái param ở file py2c.py theo model muốn chạy(hướng dẫn bên dưới)

bước 3: chạy main.py

Bước 4: code sẽ xuất ra các đoạn code cpp và header file. Sau đó cho đoạn code được xuất ra vào project c++ để chạy(thay đổi hằng numberofpicture = số lượng input, hằng d = dimension của input.

LƯU Ý: 

Chỉ hỗ trợ model được xây dựng bởi Keras Functional API

