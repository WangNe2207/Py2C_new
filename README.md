




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
Print the values ​​of the image *from left to right, top to bottom and red channel to blue channel*
![Screenshot 2025-01-10 110027](https://github.com/user-attachments/assets/f6dcc3d7-5b43-4746-acd1-5930a76bf4d6)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Red&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Green&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Blue  
txt: ![My Project](https://img.shields.io/badge/0.1_0.2_0.3_0.4_0.5_0.6_0.7_0.8_0.9-red)![My Project](https://img.shields.io/badge/1_1.1_1.2_1.3_1.4_1.5_1.6_1.7_1.8-green)![My Project](https://img.shields.io/badge/1.9_2_2.1_2.2_2.3_2.4_2.5_2.6_2.7-blue)

3. 1D(Fashion MNIST):  
Print the values ​​of the image *from left to right, top to bottom*  
![Screenshot 2025-01-10 114657](https://github.com/user-attachments/assets/3fe3c109-20a8-469c-944b-24bb555a5cce)  
txt: ![My Project](https://img.shields.io/badge/0.1_0.2_0.3_0.4_0.5_0.6_0.7_0.8_0.9-white)
### Signal(Ptbxl, HAR):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Print the values ​​of the signal set *in order of type*  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![Screenshot 2025-01-10 115943](https://github.com/user-attachments/assets/5e732a49-883d-43ab-ac65-192aab20158f)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;I&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;II&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;III&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;VR&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;VL&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;VF&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;V1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;V2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;V3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;V4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;V5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;V6  
txt: ![My Project](https://img.shields.io/badge/0.1_1.3_..._2.5_3.7-747474)![My Project](https://img.shields.io/badge/0.2_1.4_..._2.6_3.8-7030A0)![My Project](https://img.shields.io/badge/0.3_1.5_..._2.7_3.9-00359E)![My Project](https://img.shields.io/badge/0.4_1.6_..._2.8_4-0070C0)![My Project](https://img.shields.io/badge/0.5_1.7_..._2.9_4.1-00B0F0)![My Project](https://img.shields.io/badge/0.6_1.8_..._3_4.2-00B050)![My Project](https://img.shields.io/badge/0.7_1.9_..._3.1_4.3-92D050)![My Project](https://img.shields.io/badge/0.8_2_..._3.2_4.4-FFFF00)![My Project](https://img.shields.io/badge/0.9_2.1_..._3.3_4.5-FFC000)![My Project](https://img.shields.io/badge/1_2.2_..._3.4_4.6-FF0000)![My Project](https://img.shields.io/badge/1.1_2.3_..._3.5_4.7-C00000)![My Project](https://img.shields.io/badge/1.2_2.4_..._3.6_4.8-white)  


## Parameter Set up  
### Py2C set up

- Step 1: in main.py at line: pyc_lib = Py2C("....."), fill .... part with path to your file h5   

- Step 2: set up param ở file py2c.py suit for your model
- 

- bước 3: chạy main.py

- Bước 4: code sẽ xuất ra các đoạn code cpp và header file. Sau đó cho đoạn code được xuất ra vào project c++ để chạy(thay đổi hằng numberofpicture = số lượng input, hằng d = dimension của input.


## Citation

HƯỚNG DẪN SỬ DỤNG PY2C:


LƯU Ý: 

Chỉ hỗ trợ model được xây dựng bởi Keras Functional API

