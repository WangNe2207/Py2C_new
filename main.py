from Py2C import Py2C
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.__version__)
pyc_lib = Py2C("Inceptionv1.h5")
pyc_lib.convert2C()
pyc_lib.WriteCfile()
# pyc_lib.del_one_file("CNN.hh")
# pyc_lib.del_any_file(path_w)
# pyc_lib.del_all_file()
# pyc_lib.set_Fxp_Param((16,6))
# pyc_lib.Write_IEEE754_32bits_Weights_File()
pyc_lib.Write_Float_Weights_File()
# pyc_lib.Write_FixedPoint_Weights_File()
