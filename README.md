# Py2C

This is the official Py2C tool of the paper "High-Level Synthesis from Python to RTL for Deep
Neural Networks on SoC System"

## Version Requirements  
- Tensorflow = 2.10 (**equal to 2.10**)
- Keras = 2.10  (**equal to 2.10**)
## User manual  

## Dataset Preparing  
Because Py2C's output will contain a C++ testbench, we need to convert the dataset we use into Txt format to run the C++ testbench. Depending on which dataset you use, there will be different ways to convert it.
### Image
1. 3D(Cifar10, ImageNet):
- Print the values ​​of the image *from left to right, top to bottom, and red channel to blue channel*  
![Screenshot 2025-01-10 110027](https://github.com/user-attachments/assets/f6dcc3d7-5b43-4746-acd1-5930a76bf4d6)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Red&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Green&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Blue  
txt: ![My Project](https://img.shields.io/badge/0.1_0.2_0.3_0.4_0.5_0.6_0.7_0.8_0.9-red)![My Project](https://img.shields.io/badge/1_1.1_1.2_1.3_1.4_1.5_1.6_1.7_1.8-green)![My Project](https://img.shields.io/badge/1.9_2_2.1_2.2_2.3_2.4_2.5_2.6_2.7-blue)

2. 1D(Fashion MNIST):  
- Print the values ​​of the image *from left to right, top to bottom*  
![Screenshot 2025-01-10 114657](https://github.com/user-attachments/assets/3fe3c109-20a8-469c-944b-24bb555a5cce)  
txt: ![My Project](https://img.shields.io/badge/0.1_0.2_0.3_0.4_0.5_0.6_0.7_0.8_0.9-white)
### Signal(Ptbxl, HAR):  
- Print the values ​​of the signal set *in order of type*  
Example using Ptbxl dataset[1]  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![Screenshot 2025-01-10 115943](https://github.com/user-attachments/assets/5e732a49-883d-43ab-ac65-192aab20158f)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;I&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;II&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;III&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;VR&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;VL&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;VF&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;V1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;V2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;V3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;V4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;V5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;V6  
txt: ![My Project](https://img.shields.io/badge/0.1_1.3_..._2.5_3.7-747474)![My Project](https://img.shields.io/badge/0.2_1.4_..._2.6_3.8-7030A0)![My Project](https://img.shields.io/badge/0.3_1.5_..._2.7_3.9-00359E)![My Project](https://img.shields.io/badge/0.4_1.6_..._2.8_4-0070C0)![My Project](https://img.shields.io/badge/0.5_1.7_..._2.9_4.1-00B0F0)![My Project](https://img.shields.io/badge/0.6_1.8_..._3_4.2-00B050)![My Project](https://img.shields.io/badge/0.7_1.9_..._3.1_4.3-92D050)![My Project](https://img.shields.io/badge/0.8_2_..._3.2_4.4-FFFF00)![My Project](https://img.shields.io/badge/0.9_2.1_..._3.3_4.5-FFC000)![My Project](https://img.shields.io/badge/1_2.2_..._3.4_4.6-FF0000)![My Project](https://img.shields.io/badge/1.1_2.3_..._3.5_4.7-C00000)![My Project](https://img.shields.io/badge/1.2_2.4_..._3.6_4.8-white)  


## Parameter Set up  
### Py2C set up
- Step 1: in main.py at line: pyc_lib = Py2C("....."), fill .... part with the path to your file h5   
- Step 2: set up param ở file py2c.py suit for your model
  - *model_path* (string) is the path of h5 model file (Note: This function supports CNN and ANN)
  - *type* (string) is the type of data such as int, float, fxp (default: "fxp")
  - *fxp_para* is the parameter of fxp if you choose. It has 2 parameters (x,y) with x as the sum of bits showing data and y as the integral part of the data
  - *choose_only_output* is used to choose the type of output of your model. If your model's output is a number, then set it *True*. If your model's output is an array, then set it *False*. For example:  
    - if you use "sigmoid" for the activation function of your last dense layer, you will have output as an array. So set *choose_only_output* equal to *False*.  
    - if you use "softmax" for the activation function of your last dense layer, you will have output as a number. So set *choose_only_output* equal to *True*.  
  - *ide* is used to choose what kind of IDE that you use. if you use Visual Studio, set ide = "vs". If you use Visual Studio Code or something else you can ignore it
  - *num_of_output* is the number of output your model has. For example: to use inceptionV1 in Py2C tool, you need to set up *num_of_output* equal to 3  
    ![Image](https://github.com/user-attachments/assets/f273f789-266a-4d34-b086-3166949de4a5)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;InceptionV1 architecture  
- Step 3: Run main.py  
### C++ code  

After running Py2C properly, you will have 10 files in total.  
- Testbench (CNN_tb.cpp): it's testbench of all C++ code we have. we will run it in a C++ compiler to make sure that our C++ code is correct.  
- C++ code (CNN.cpp, CNN.h, Conv.cpp, Conv.h, Dense.cpp, Dense.h, Pool.cpp, Pool.h): These are the files that we will put into Vitis HLS.  
- Weight (Float_Weight.txt): this file contains all parameters of our model.  
### Run C++ testbench 

- Step 1: Open CNN_tb.cpp
- Step 2: Inline "#define NumberOfPicture ...", fill the... part with the number of your input.  
- Step 3: In lines "FILE* Input = fopen("X.txt", "r");" and "FILE* Output = fopen("Y.txt", "r");", Replaces X.txt with your txt input, Y.txt with your text label.
- Step 4: Run C++ code.

## Note:
- Our tool only supports DNN model built by Keras Functional API and h5 checkpoint.
- Important: Your h5 checkpoint must be built on the same Keras's and TensorFlow's versions. otherwise, it will have the possibility to cause errors. then, your h5 checkpoint's version must be 2.10  

## Our Experiment
- We've already done some experiments using HAR[3] and Cifar10[1] dataset combine with VGG16[4], Resnet34[5], InceptionV1[6] model. You can access it through the link below:  
Link: https://drive.google.com/file/d/1xchjuCPuoTrw1F1vuslt91zmcvxyeu8_/view?usp=drive_link  

## Citation
<!--[1] Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.1). PhysioNet. https://doi.org/10.13026/x4td-x982.  
[2] Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.  
[3] Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013, April). A public domain dataset for human activity recognition using smartphones. In Esann (Vol. 3, p. 3).  
[4] Simonyan, K. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.  
[5] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).  
[6] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).  
