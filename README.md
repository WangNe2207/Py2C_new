# PY2C

This is the official implementation of paper "High-Level Synthesis from Python to RTL for Deep Neural Networks on SoC System"

## User manual
### Prepare txt Dataset
   - **Image input**:
      + 3D(Cifar10, Cifar100, ImageNet): print all values of a image in channel order.  
   Example:  
   ![ImageOfReadme_0](https://github.com/user-attachments/assets/fd4355ce-67bf-4266-9e89-2e0b4f58f3a4)

      + 1D(Fashion-MNIST): print all values of a image.  
   Example:  
   ![ImageOfReadme_1](https://github.com/user-attachments/assets/25206884-03e9-480d-a861-73717b686f58)

   - **Signal input(PTB-XL, HAR)**:  
   Example:  
![ImageOfReadme_2](https://github.com/user-attachments/assets/25e17c1b-d39f-4223-a14f-d7d860b26c0f)

   - **Label:** print all label in the same order with input

### Using Py2C:  
   **Step 1:** in main.py at line: pyc_lib = Py2C("....."), fill in the .... part the path to your H5 file  
   **Step 2:** edit the parameters in py2c.py file according to your model  
   **Step 3:** run main.py  
### Edit C++ Code:  
   **Step 1:** In CNN_tb.cpp: replaces constant numberofpicture = number of inputs, constant d = dimension of input  
   **Step 2:** In CNN_tb.cpp: in line "FILE* Input = fopen("X.txt", "r");" and "FILE* Output = fopen("Y.txt", "r");" .replaces "X.txt" and "Y.txt" with the name of your txt files  
   **Step 3:** run C code  
