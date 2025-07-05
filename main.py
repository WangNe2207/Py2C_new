from Py2C import Py2C
pyc_lib = Py2C(model_path="scripted_model_VGG16.pt",
               torch=True,
               input_size=(9, 128), # only use in Pytorch model
               type="float",
               fxp_para=(32, 16), #only use in  fixed-point datatype
               num_of_output=1,
               choose_only_output=True,
               ide="vs")
pyc_lib.convert2C()
pyc_lib.WriteCfile()
pyc_lib.Write_Float_Weights_File()
# pyc_lib.del_one_file("CNN.hh")
# pyc_lib.del_any_file(path_w)
# pyc_lib.del_all_file()
# pyc_lib.set_Fxp_Param((16,6))
# pyc_lib.Write_IEEE754_32bits_Weights_File()
# pyc_lib.Write_FixedPoint_Weights_File()
