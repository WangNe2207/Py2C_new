Version Requirements

Tensorflow = 2.10
Keras = 2.10

HƯỚNG DẪN SỬ DỤNG PY2C
bước 1: ở main.py tại dòng: pyc_lib = Py2C(".....h5"), hãy điền vào phần .... file h5 của bạn 

bước 2: chính cái param ở file py2c.py theo model muốn chạy(hướng dẫn bên dưới)

bước 3: chạy main.py

Bước 4: code sẽ xuất ra các đoạn code cpp và header file. Sau đó cho đoạn code được xuất ra vào project c++ để chạy(thay đổi hằng numberofpicture = số lượng input, hằng d = dimension của input.

LƯU Ý: 

Chỉ hỗ trợ model được xây dựng bởi Keras Functional API


    """
    The Py2C class has four inputs and nine functions.

        With input:
        - model_path (string) is path of h5 model file (Note: This function support CNN and ANN)
        - type (string) is type of data such as int, float, fxp (default: "fxp")
        - fxp_para is parameter of fxp if you choose. It has 2 parameters (x,y) with x is sum of bits showing a data and y is integral part of the data
        - if choose_only_output is False, output C code model show full array. Else it will show the only variable being argmax of array
        - ide is kind of IDE that you use to run C code. if you use Visual Studio, set ide = "vs". if you use Visual Studio Code or something else you can ignore it
        With function:
        - set_Fxp_Param function is to set fxp parameter again
        - convert2C function is to convert the loaded model into C code and it store in self array
        - WriteCfile function is to write C code from convert2C into .cc and .hh file
        - del_one_file function is to delete the particular file
        - del_any_file function is to delete any file in the particular array
        - del_all_file function is to delete all of .cc and .hh file, which has created
        - Write_Float_Weights_File function is to create Float Weights file
        - Write_IEEE754_32bits_Weights_File function is to create IEEE754 32bits Weights file
        - Write_FixedPoint_Weights_File function is to create Fixed Point Weights file
    """
