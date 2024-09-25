Version Requirements

Tensorflow = 2.10
Keras = 2.10


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
