import os
import struct

import numpy as np
import tensorflow as tf
import math


class Py2C:
    """
    The Py2C class has four inputs and nine functions.

        With input:
        - model_path (string) is path of h5 model file (Note: This function support CNN and ANN)
        - type (string) is type of data such as int, float, fxp (default: "fxp")
        - fxp_para is parameter of fxp if you choose. It has 2 parameters (x,y) with x is sum of bits showing a data and y is integral part of the data
        - if choose_only_output is False, output C code model show full array. Else it will show the only variable being argmax of array
        - ide is kind of IDE that you use. if you use Visual Studio, set ide = "vs". if you use Visual Studio Code or something else you can ignore it
        - num_of_output is number of output your model have.
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

    def __init__(self, model_path, type="float", fxp_para=(21, 8),num_of_output=1, choose_only_output=True, ide="vs"):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        assert fxp_para[0] > 0 and fxp_para[1] > 0, "the 1st or the 2nd Fxp Parameter must be more than zero!!!"
        assert fxp_para[0] >= fxp_para[1], "the 1st Fxp Parameter must be equal or more than the 2nd one!!!"
        if type == "fxp":
            self.fxp_para = fxp_para
        else:
            self.fxp_para = None
        self.choose_only_output = choose_only_output
        self.type = type
        self.config = self.model.get_config()
        self.index = 0
        self.ide = ide
        self.num_of_output = num_of_output

        self.base_include = ""
        self.fxp_include = "#include <ap_axi_sdata.h>\ntypedef ap_fixed<" + str(fxp_para[0]) + "," + str(
            fxp_para[1]) + "> fxp;\n"
        self.CNN_include = "#include \"Conv.h\"\n#include \"Pool.h\"\n#include \"Dense.h\"\n#include <algorithm>\n#include <string.h>\n"
        self.source_Conv_cc = ""

        self.act_arr = ""
        self.fxp_inc = ""
        self.base_inc = ""
        self.full_source_Conv_cc = []
        self.full_source_Conv_hh = []
        self.full_source_Pool_cc = []
        self.full_source_Pool_hh = []
        self.full_source_Dense_cc = []
        self.full_source_Dense_hh = []
        self.full_source_CNN_cc = []
        self.Weights = []
        self.source_CNN = ""
        self.source_CNN_hh = ""
        self.source_CNN_tb = ""
        self.index = 0
        self.index2D = 0
        self.indexBatch = 0
        self.indexActi = 0
        self.index_P = 0
        self.index_P2D = 0
        self.index_D = 0
        self.indexAdd = 0
        self.indexConcatenate = 0
        self.index_Flatten = 0
        self.index_GlbAvgPool = 0
        self.index_GlbMaxPool = 0
        self.index_output = 0
        self.cnt_param = 0
        self.count_dense_output=0
        self.call = ["",""]
        self.call_function = ""
        self.out = ["", ""]
        self.full_source_CNN_cc.append(["", "&InModel[0]"])
        self.path_w = ["Conv.cpp", "Conv.h", "Pool.cpp", "Pool.h", "Dense.cpp", "Dense.h", "CNN.cpp", "CNN.h",
                       "CNN_tb.cpp"]
        print("Model Information")
        # self.model.summary()

    def set_Fxp_Param(self, fxp_para):
        assert fxp_para[0] > 0 and fxp_para[1] > 0, "the 1st or the 2nd Fxp Parameter must be more than zero!!!"
        assert fxp_para[0] >= fxp_para[1], "the 1st Fxp Parameter must be equal or more than the 2nd one!!!"
        self.fxp_para = fxp_para

    def convert2C(self):
        ######################################### WRITING PHARSE #######################################################
        if (len(self.model.input_shape) == 4):
            input_shape = (self.model.input_shape[1], self.model.input_shape[2], self.model.input_shape[3])
            depth = min(input_shape)
            depth_index = input_shape.index(depth)
            if (depth_index == 0): depth_index = 1
            if (depth_index == 2): depth_index = 3

            if depth_index == 1:
                height_index = 2
                width_index = 3
            if depth_index == 3:
                height_index = 1
                width_index = 2
            assert ((depth_index == 1) or (depth_index == 3)), "set up height_index, width_index and depth_index wrong!!!!"
        else:
            depth_index = 0
            height_index = 1
            width_index = 2


        for i in range(len(self.config["layers"])):
            found = 0
            layer = self.config["layers"][i]['class_name']
            if layer.find("Conv2D") >= 0 and layer.find("conv2d_input") < 0:
                found = 1
                activation = self.config["layers"][i]['config']['activation']
                in_shape = (
                self.model.layers[i].input.shape[depth_index], self.model.layers[i].input.shape[height_index],
                self.model.layers[i].input.shape[width_index])
                out_shape = (
                self.model.layers[i].output.shape[depth_index], self.model.layers[i].output.shape[height_index],
                self.model.layers[i].output.shape[width_index])
                kernel_shape = (
                    self.model.layers[i].get_weights()[0].shape[3], self.model.layers[i].get_weights()[0].shape[2],
                    self.model.layers[i].get_weights()[0].shape[0], self.model.layers[i].get_weights()[0].shape[1])
                h = np.transpose(self.model.layers[i].get_weights()[0], (3, 2, 0, 1)).reshape(
                    kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3])
                for k in h:
                    self.Weights.append(k)
                if self.model.layers[i].bias is None:
                    if activation == "sigmoid":
                        self.act_arr = "Output_Conv[" + str(out_shape[1]) + "*" + str(out_shape[2]) + "*n+" + str(out_shape[2]) + "*x+y]=1/(1 + exp(-s));"
                    elif activation == "tanh":
                        self.act_arr = "Output_Conv[" + str(out_shape[1]) + "*" + str(out_shape[2]) + "*n+" + str(out_shape[2]) + "*x+y]=(2/(1 + exp(-2*s)))-1;"
                    elif activation == "relu":
                        self.act_arr = "if (s < 0) Output_Conv[" + str(out_shape[1]) + "*" + str(out_shape[2]) + "*n+" + str(out_shape[2]) + "*x+y]=0; else Output_Conv[" + str(out_shape[1]) + "*" + str(out_shape[2]) + "*n+" + str(out_shape[2]) + "*x+y]=s;"
                    else:
                        self.act_arr = "Output_Conv[" + str(out_shape[1]) + "*" + str(out_shape[2]) + "*n+" + str(
                            out_shape[2]) + "*x+y]=s;"
                else:
                    for k in self.model.layers[i].get_weights()[1]:
                        self.Weights.append(k)
                    if activation == "sigmoid":
                        self.act_arr = "Output_Conv[" + str(out_shape[1]) + "*" + str(out_shape[2]) + "*n+" + str(out_shape[2]) + "*x+y]=1/(1 + exp(-(s+bias[n])));"
                    elif activation == "tanh":
                        self.act_arr = "Output_Conv[" + str(out_shape[1]) + "*" + str(out_shape[2]) + "*n+" + str(out_shape[2]) + "*x+y]=(2/(1 + exp(-2*(s+bias[n]))))-1;"
                    elif activation == "relu":
                        self.act_arr = "if ((s+bias[n])<0) Output_Conv[" + str(out_shape[1]) + "*" + str(
                            out_shape[2]) + "*n+" + str(out_shape[2]) + "*x+y]=0; else Output_Conv[" + str(
                            out_shape[1]) + "*" + str(out_shape[2]) + "*n+" + str(out_shape[2]) + "*x+y]=s+bias[n];"
                    else:
                        self.act_arr = "Output_Conv[" + str(out_shape[1]) + "*" + str(out_shape[2]) + "*n+" + str(
                            out_shape[2]) + "*x+y]=s+bias[n];"

                if self.type == "fxp" and self.index2D == 0:
                    self.fxp_inc = self.fxp_include
                else:
                    self.fxp_inc = ""
                if activation == "sigmoid" or activation == "tanh":
                    self.fxp_inc += "#include <cmath>\n"
                else:
                    self.fxp_inc += ""
                stride = self.model.layers[i].strides[0]
                if self.config["layers"][i]['config']['padding'] == 'same':
                    padding = (((out_shape[1] - 1) * stride - in_shape[1] + kernel_shape[2]) / 2,
                               ((out_shape[2] - 1) * stride - in_shape[2] + kernel_shape[3]) / 2)

                    in_shape_if_padding = (in_shape[0], (in_shape[1] + 2 * padding[0]), (in_shape[2] + 2 * padding[1]))
                    source_pad_conv_cc = self.fxp_inc + "void Padding_Conv2D_" + str(
                        self.index2D) + "(" + self.type + " input_Pad_Conv[" + str(
                        in_shape[0] * in_shape[1] * in_shape[2]) + "], " + self.type + " output_Pad_Conv[" + str(int(
                        in_shape_if_padding[0] * in_shape_if_padding[1] * in_shape_if_padding[
                            2])) + "]){\n\tloop_for_3_channel_pad_" + str(
                        self.index2D) + ":\n\tfor (int c = 0; c < " + str(
                        in_shape_if_padding[0]) + "; c++){" + "\n\t\tloop_for_channel_pad_" + str(
                        self.index2D) + ":\n\t\tfor (int n = 0; n < " + str(
                        int(in_shape_if_padding[1])) + "; n++){\n\t\t\tloop_for_weight_pad_" + str(
                        self.index2D) + ":\n\t\t\tfor (int i = 0; i < " + str(
                        int(in_shape_if_padding[2])) + "; i++){\n\t\t\t\t"
                    if padding[0] < 1:
                        source_pad_conv_cc += "if (n >= " + str(in_shape[2]) + ") output_Pad_Conv[" + str(
                            int(in_shape_if_padding[1])) + " * " + str(int(in_shape_if_padding[2])) + " * c + " + str(
                            int(in_shape_if_padding[2])) + " * n + i]=0;\n\t\t\t\t else \n\t\t\t\t\tif (i >= " + str(
                            in_shape[2]) + ") output_Pad_Conv[" + str(int(in_shape_if_padding[1])) + " * " + str(
                            int(in_shape_if_padding[2])) + " * c + " + str(
                            int(in_shape_if_padding[2])) + " * n + i]=0; else output_Pad_Conv[" + str(
                            int(in_shape_if_padding[1])) + " * " + str(int(in_shape_if_padding[2])) + " * c + " + str(
                            int(in_shape_if_padding[2])) + " * n + i] = input_Pad_Conv[" + str(
                            in_shape[1]) + " * " + str(in_shape[2]) + " * c + " + str(
                            in_shape[2]) + " * n + i];\n\t\t\t}\n\t\t}\n\t}\n}\n"
                    else:
                        source_pad_conv_cc += "if (n < " + str(int(0 + (math.floor(padding[0])))) + " || n >= " + str(
                            int(in_shape[2] + (math.floor(padding[0])))) + ") output_Pad_Conv[" + str(
                            int(in_shape_if_padding[1])) + " * " + str(int(in_shape_if_padding[2])) + " * c + " + str(
                            int(in_shape_if_padding[2])) + " * n + i]=0;\n\t\t\t\t else \n\t\t\t\t\tif (i < " + str(
                            int(0 + (math.floor(padding[0])))) + " || i >= " + str(
                            int(in_shape[2] + (math.floor(padding[0])))) + ") output_Pad_Conv[" + str(
                            int(in_shape_if_padding[1])) + " * " + str(int(in_shape_if_padding[2])) + " * c + " + str(
                            int(in_shape_if_padding[2])) + " * n + i]=0; else output_Pad_Conv[" + str(
                            int(in_shape_if_padding[1])) + " * " + str(int(in_shape_if_padding[2])) + " * c + " + str(
                            int(in_shape_if_padding[2])) + " * n + i] = input_Pad_Conv[" + str(
                            in_shape[1]) + " * " + str(in_shape[2]) + " * c + " + str(in_shape[2]) + " * (n - " + str(
                            int(math.floor(padding[1]))) + ") + i - " + str(int(math.floor(padding[1]))) + "];\n\t\t\t}\n\t\t}\n\t}\n}\n"
                    source_pad_conv_hh = self.fxp_inc + "void Padding_Conv2D_" + str(
                        self.index2D) + "(" + self.type + " input_Pad_Conv[" + str(
                        in_shape[0] * in_shape[1] * in_shape[2]) + "], " + self.type + " output_Pad_Conv[" + str(
                        in_shape[0] * int(in_shape[1] + 2 * padding[0]) * int(in_shape[2] + 2 * padding[1])) + "]);\n"
                    self.full_source_Conv_cc.append(source_pad_conv_cc)
                    self.full_source_Conv_hh.append(source_pad_conv_hh)
                    self.call_function += "\t" + self.type + " OutPadConv" + str(self.index2D) + "[" + str(
                        in_shape[0] * int(in_shape[1] + 2 * padding[0]) * int(in_shape[2] + 2 * padding[1])) + "];\n"
                    self.full_source_CNN_cc.append(
                        ["\tPadding_Conv2D_" + str(self.index2D) + "(", "OutPadConv" + str(self.index2D), "",
                         ""])
                    self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(
                        out_shape[0] * out_shape[1] * out_shape[2]) + "];\n"
                    if self.model.layers[i].bias is None:
                        source_Conv_cc = (self.fxp_inc + "void Conv2D_" + str(self.index2D) + "(" + self.type + " Input_Conv[" + str(int(in_shape_if_padding[0] * in_shape_if_padding[1] * in_shape_if_padding[2])) + "]," + self.type + " Output_Conv[" + str(out_shape[0] * out_shape[1] * out_shape[2]) + "], " + self.type + " kernel[" + str(kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]) + "])")
                    else:
                        source_Conv_cc = (self.fxp_inc + "void Conv2D_" + str(self.index2D) + "(" + self.type + " Input_Conv[" + str(int(in_shape_if_padding[0] * in_shape_if_padding[1] * in_shape_if_padding[2])) + "]," + self.type + " Output_Conv[" + str(out_shape[0] * out_shape[1] * out_shape[2]) + "], " + self.type + " bias[" + str(out_shape[0]) + "], " + self.type + " kernel[" + str(kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]) + "])")

                    source_Conv_cc+= ("{\n\tloop_for_channel2D_" + str(self.index2D) + ":\n\tint stride = " + str(stride) + ";\n\tfor (int n = 0; n < " + str(
                        out_shape[0]) + "; n++){\n\t\tloop_for_bp2D_" + str(
                        self.index2D) + ":\n\t\tfor (int x = 0; x < " + str(
                        out_shape[1]) + "; x++){\n\t\t\tloop_for_ap2D_" + str(
                        self.index2D) + ":\n\t\t\tfor (int y = 0; y < " + str(
                        out_shape[2]) + "; y++){\n\t\t\t\t" + self.type + " s = 0;\n\t\t\t\tloop_for_fc_" + str(
                        self.index2D) + ":\n\t\t\t\tfor (int k = 0; k < " + str(
                        kernel_shape[1]) + "; k++){\n\t\t\t\t\tloop_for_fb_" + str(
                        self.index2D) + ":\n\t\t\t\t\tfor (int i = 0; i < " + str(
                        kernel_shape[2]) + "; i++){\n\t\t\t\t\t\tloop_for_fa_" + str(
                        self.index2D) + ":\n\t\t\t\t\t\tfor (int j = 0; j < " + str(
                        kernel_shape[3]) + "; j++){\n\t\t\t\t\t\t\ts=s+(kernel[" + str(kernel_shape[1]) + "*" + str(
                        kernel_shape[2]) + "*" + str(
                        kernel_shape[2]) + "*n+" + str(kernel_shape[2]) + "*" + str(
                        kernel_shape[3]) + "*k+" + str(
                        kernel_shape[3]) + "*i+j])*(Input_Conv[" + str(int(in_shape_if_padding[1])) + "*" + str(
                        int(in_shape_if_padding[2])) + "*k+" + str(int(in_shape_if_padding[
                                                                           2])) + "*(i+x*stride)+j+y*stride]);}\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t\t" + self.act_arr + "\n\t\t\t}\n\t\t}\n\t}\n}\n")

                    if self.model.layers[i].bias is None:
                        source_Conv_hh = self.fxp_inc + "void Conv2D_" + str(self.index2D) + "(" + self.type + " Input_Conv[" + str(int(in_shape_if_padding[0] * in_shape_if_padding[1] * in_shape_if_padding[2])) + "]," + self.type + " Output_Conv[" + str(out_shape[0] * out_shape[1] * out_shape[2]) + "], " + self.type + " kernel[" + str(kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]) + "]);\n"
                    else:
                        source_Conv_hh = self.fxp_inc + "void Conv2D_" + str(
                            self.index2D) + "(" + self.type + " Input_Conv[" + str(int(
                            in_shape_if_padding[0] * in_shape_if_padding[1] * in_shape_if_padding[
                                2])) + "]," + self.type + " Output_Conv[" + str(
                            out_shape[0] * out_shape[1] * out_shape[2]) + "], " + self.type + " bias[" + str(
                            out_shape[0]) + "], " + self.type + " kernel[" + str(
                            kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]) + "]);\n"

                else:
                    self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(
                        out_shape[0] * out_shape[1] * out_shape[2]) + "];\n"
                    if self.model.layers[i].bias is None:
                        source_Conv_cc = (self.fxp_inc + "void Conv2D_" + str(self.index2D) + "(" + self.type + " Input_Conv[" + str(in_shape[0] * in_shape[1] * in_shape[2]) + "]," + self.type + " Output_Conv[" + str(out_shape[0] * out_shape[1] * out_shape[2]) + "], " + self.type + " kernel[" + str(kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]) + "])")
                    else:
                        source_Conv_cc = (self.fxp_inc + "void Conv2D_" + str(self.index2D) + "(" + self.type + " Input_Conv[" + str(in_shape[0] * in_shape[1] * in_shape[2]) + "]," + self.type + " Output_Conv[" + str(out_shape[0] * out_shape[1] * out_shape[2]) + "], " + self.type + " bias[" + str(out_shape[0]) + "], " + self.type + " kernel[" + str(kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]) + "])")

                    source_Conv_cc+= ("{\n\tint stride = " + str(stride) + ";\n\tloop_for_channel2D_" + str(self.index2D) + ":\n\tfor (int n = 0; n < " + str(
                        out_shape[0]) + "; n++){\n\t\tloop_for_bp2D_" + str(
                        self.index2D) + ":\n\t\tfor (int x = 0; x < " + str(
                        out_shape[1]) + "; x++){\n\t\t\tloop_for_ap2D_" + str(
                        self.index2D) + ":\n\t\t\tfor (int y = 0; y < " + str(
                        out_shape[2]) + "; y++){\n\t\t\t\t" + self.type + " s = 0;\n\t\t\t\tloop_for_fc_" + str(
                        self.index2D) + ":\n\t\t\t\tfor (int k = 0; k < " + str(
                        kernel_shape[1]) + "; k++){\n\t\t\t\t\tloop_for_fb_" + str(
                        self.index2D) + ":\n\t\t\t\t\tfor (int i = 0; i < " + str(
                        kernel_shape[2]) + "; i++){\n\t\t\t\t\t\tloop_for_fa_" + str(
                        self.index2D) + ":\n\t\t\t\t\t\tfor (int j = 0; j < " + str(
                        kernel_shape[3]) + "; j++){\n\t\t\t\t\t\t\ts=s+(kernel[" + str(kernel_shape[1]) + "*" + str(
                        kernel_shape[2]) + "*" + str(
                        kernel_shape[2]) + "*n+" + str(kernel_shape[2]) + "*" + str(
                        kernel_shape[3]) + "*k+" + str(
                        kernel_shape[3]) + "*i+j])*(Input_Conv[" + str(
                        in_shape[1]) + "*" + str(
                        in_shape[2]) + "*k+" + str(
                        in_shape[
                            2]) + "*(i+x*stride)+j+y*stride]);}\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t\t" + self.act_arr + "\n\t\t\t}\n\t\t}\n\t}\n}\n")

                    if self.model.layers[i].bias is None:
                        source_Conv_hh = self.fxp_inc + "void Conv2D_" + str(self.index2D) + "(" + self.type + " Input_Conv[" + str(in_shape[0] * in_shape[1] * in_shape[2]) + "]," + self.type + " Output_Conv[" + str(out_shape[0] * out_shape[1] * out_shape[2]) + "], " + self.type + " kernel[" + str(kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]) + "]);\n"
                    else:
                        source_Conv_hh = self.fxp_inc + "void Conv2D_" + str(
                            self.index2D) + "(" + self.type + " Input_Conv[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "]," + self.type + " Output_Conv[" + str(
                            out_shape[0] * out_shape[1] * out_shape[2]) + "], " + self.type + " bias[" + str(
                            out_shape[0]) + "], " + self.type + " kernel[" + str(
                            kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]) + "]);\n"

                self.full_source_Conv_cc.append(source_Conv_cc)
                self.full_source_Conv_hh.append(source_Conv_hh)
                if self.model.layers[i].bias is None:
                    self.full_source_CNN_cc.append(["\tConv2D_" + str(self.index2D) + "(", self.config["layers"][i]['config']['name'], "&Weights[" + str(self.cnt_param) + "]"])
                    self.cnt_param += kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]
                else:
                    self.full_source_CNN_cc.append(
                        ["\tConv2D_" + str(self.index2D) + "(", self.config["layers"][i]['config']['name'],"&Weights[" + str(self.cnt_param + kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]) + "]", "&Weights[" + str(self.cnt_param) + "]"])
                    self.cnt_param += kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3] + out_shape[0]
                self.index2D += 1

            # convert conv1d layer into c array that act like an conv1d layer
            if layer.find("Conv1D") >= 0 and layer.find("conv1d_input") < 0:
                found = 1
                activation = self.config["layers"][i]['config']['activation']
                in_shape = (self.model.layers[i].input.shape[width_index], self.model.layers[i].input.shape[height_index])
                out_shape = (self.model.layers[i].output.shape[width_index], self.model.layers[i].output.shape[height_index])
                kernel_shape = self.model.layers[i].get_weights()[0].T.shape
                h = self.model.layers[i].get_weights()[0].T.reshape(
                    kernel_shape[0] * kernel_shape[1] * kernel_shape[2])

                for k in h:
                    self.Weights.append(k)
                if self.model.layers[i].bias is None:
                    if activation == "sigmoid":
                        self.act_arr = "Output_Conv[" + str(out_shape[1]) + "*n+y]=1/(1 + exp(-s));"
                    elif activation == "tanh":
                        self.act_arr = "Output_Conv[" + str(out_shape[1]) + "*n+y]=(2/(1 + exp(-2*s)))-1;"
                    elif activation == "relu":
                        self.act_arr = "if (s < 0) Output_Conv[" + str(out_shape[1]) + "*n+y]=0; else Output_Conv[" + str(out_shape[1]) + "*n+y]=s;"
                    else:
                        self.act_arr = "Output_Conv[" + str(out_shape[1]) + "*n+y]=s;"
                else:
                    for k in self.model.layers[i].get_weights()[1]:
                        self.Weights.append(k)
                    if activation == "sigmoid":
                        self.act_arr = "Output_Conv[" + str(out_shape[1]) + "*n+y]=1/(1 + exp(-(s+bias[n])));"
                    elif activation == "tanh":
                        self.act_arr = "Output_Conv[" + str(out_shape[1]) + "*n+y]=(2/(1 + exp(-2*(s+bias[n]))))-1;"
                    elif activation == "relu":
                        self.act_arr = "if ((s+bias[n])<0) Output_Conv[" + str(out_shape[1]) + "*n+y]=0; else Output_Conv[" + str(out_shape[1]) + "*n+y]=s+bias[n];"
                    else:
                        self.act_arr = "Output_Conv[" + str(out_shape[1]) + "*n+y]=s+bias[n];"

                if self.type == "fxp" and self.index == 0:
                    self.fxp_inc = self.fxp_include
                else:
                    self.fxp_inc = ""
                if activation == "sigmoid" or activation == "tanh":
                    self.fxp_inc += "#include <cmath>\n"
                else:
                    self.fxp_inc += ""
                stride = self.model.layers[i].strides[0]
                if self.config["layers"][i]['config']['padding'] == 'same':
                    padding_left = math.floor(((out_shape[1] - 1) * stride - (in_shape[1] - (kernel_shape[2] - 1) - 1)) / 2)
                    padding_right = math.ceil(((out_shape[1] - 1) * stride - (in_shape[1] - (kernel_shape[2] - 1) - 1)) / 2)
                    in_shape_if_padding = (in_shape[0], (in_shape[1] + padding_left + padding_right))
                    source_pad_conv_cc = self.fxp_inc + "void Padding_Conv1D_" + str(
                        self.index) + "(" + self.type + " input_Pad_Conv[" + str(
                        in_shape[0] * in_shape[1]) + "], " + self.type + " output_Pad_Conv[" + str(
                        in_shape_if_padding[0] * in_shape_if_padding[1]) + "]){\n\tloop_for_3_channel_pad_" + str(
                        self.index) + ":\n\tfor (int c = 0; c < " + str(
                        in_shape_if_padding[0]) + "; c++){" + "\n\t\tloop_for_channel_pad_" + str(
                        self.index) + ":\n\t\tfor (int n = 0; n < " + str(
                        in_shape_if_padding[1]) + "; n++){\n\t\t\tif (n < " + str(0 + padding_left) + " || n >= " + str(
                        in_shape_if_padding[1] - padding_right) + ") output_Pad_Conv[" + str(
                        in_shape_if_padding[1]) + " * c + n]=0; else output_Pad_Conv[" + str(
                        in_shape_if_padding[1]) + " * c + n] = input_Pad_Conv[" + str(
                        in_shape[1]) + " * c + n - " + str(padding_left) + "];\n\t\t}\n\t}\n}\n"
                    source_pad_conv_hh = self.fxp_inc + "void Padding_Conv1D_" + str(
                        self.index) + "(" + self.type + " input_Pad_Conv[" + str(
                        in_shape[0] * in_shape[1]) + "], " + self.type + " output_Pad_Conv[" + str(
                        in_shape_if_padding[0] * in_shape_if_padding[1]) + "]);\n"
                    self.full_source_Conv_cc.append(source_pad_conv_cc)
                    self.full_source_Conv_hh.append(source_pad_conv_hh)
                    self.call_function += "\t" + self.type + " OutPadConv" + str(self.index) + "[" + str(
                        in_shape_if_padding[0] * in_shape_if_padding[1]) + "];\n"
                    self.full_source_CNN_cc.append(
                        ["\tPadding_Conv1D_" + str(self.index) + "(", "OutPadConv" + str(self.index), "",
                         ""])

                    self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(
                        out_shape[0] *
                        out_shape[1]) + "];\n"
                    if self.model.layers[i].bias is None:
                        source_Conv_cc = (self.fxp_inc + "void Conv1D_" + str(self.index) + "(" + self.type + " Input_Conv[" + str(in_shape_if_padding[0] * in_shape_if_padding[1]) + "]," + self.type + " Output_Conv[" + str(out_shape[0] * out_shape[1]) + "], " + self.type + " kernel[" + str(kernel_shape[0] * kernel_shape[1] * kernel_shape[2]) + "])")
                    else:
                        source_Conv_cc = (self.fxp_inc + "void Conv1D_" + str(self.index) + "(" + self.type + " Input_Conv[" + str(in_shape_if_padding[0] * in_shape_if_padding[1]) + "]," + self.type + " Output_Conv[" + str(out_shape[0] * out_shape[1]) + "], " + self.type + " bias[" + str(out_shape[0]) + "], " + self.type + " kernel[" + str(kernel_shape[0] * kernel_shape[1] * kernel_shape[2]) + "])")

                    source_Conv_cc += ("{\n\tloop_for_channel_" + str(
                        self.index) + ":\n\tint stride = " + str(stride) + ";\n\tfor (int n = 0; n < " + str(
                        out_shape[0]) + "; n++){\n\t\tloop_for_ap_" + str(
                        self.index) + ":\n\t\tfor (int y = 0; y < " + str(
                        out_shape[1]) + "; y++){\n\t\t\t" + self.type + " s = 0;\n\t\t\tloop_for_fc_" + str(
                        self.index) + ":\n\t\t\tfor (int k = 0; k < " + str(
                        kernel_shape[1]) + "; k++){\n\t\t\t\tloop_for_fa_" + str(
                        self.index) + ":\n\t\t\t\tfor (int j = 0; j < " + str(
                        kernel_shape[2]) + "; j++){\n\t\t\t\t\ts=s+(kernel[" + str(kernel_shape[1]) + "*" + str(
                        kernel_shape[2]) + "*n+" + str(kernel_shape[2]) + "*k+j])*(Input_Conv[" + str(
                        in_shape_if_padding[
                            1]) + "*k+j+y*stride]);}\n\t\t\t}\n\t\t\t" + self.act_arr + "\n\t\t}\n\t}\n}\n")
                    if self.model.layers[i].bias is None:
                        source_Conv_hh = self.fxp_inc + "void Conv1D_" + str(self.index) + "(" + self.type + " Input_Conv[" + str(in_shape_if_padding[0] * in_shape_if_padding[1]) + "]," + self.type + " Output_Conv[" + str(out_shape[0] * out_shape[1]) + "], " + self.type + " kernel[" + str(kernel_shape[0] * kernel_shape[1] * kernel_shape[2]) + "]);\n"
                    else:
                        source_Conv_hh = self.fxp_inc + "void Conv1D_" + str(
                            self.index) + "(" + self.type + " Input_Conv[" + str(
                            in_shape_if_padding[0] * in_shape_if_padding[1]) + "]," + self.type + " Output_Conv[" + str(
                            out_shape[0] * out_shape[1]) + "], " + self.type + " bias[" + str(
                            out_shape[0]) + "], " + self.type + " kernel[" + str(
                            kernel_shape[0] * kernel_shape[1] * kernel_shape[2]) + "]);\n"
                else:
                    self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(
                        out_shape[0] *
                        out_shape[1]) + "];\n"
                    if self.model.layers[i].bias is None:
                        source_Conv_cc = (self.fxp_inc + "void Conv1D_" + str(self.index) + "(" + self.type + " Input_Conv[" + str(in_shape[0] * in_shape[1]) + "]," + self.type + " Output_Conv[" + str(out_shape[0] * out_shape[1]) + "], " + self.type + " kernel[" + str(kernel_shape[0] * kernel_shape[1] * kernel_shape[2]) + "])")
                    else:
                        source_Conv_cc = (self.fxp_inc + "void Conv1D_" + str(self.index) + "(" + self.type + " Input_Conv[" + str(in_shape[0] * in_shape[1]) + "]," + self.type + " Output_Conv[" + str(out_shape[0] * out_shape[1]) + "], " + self.type + " bias[" + str(out_shape[0]) + "], " + self.type + " kernel[" + str(kernel_shape[0] * kernel_shape[1] * kernel_shape[2]) + "])")

                    source_Conv_cc += ("{\n\tloop_for_channel_" + str(
                        self.index) + ":\n\tint stride = " + str(stride) + ";\n\tfor (int n = 0; n < " + str(
                        out_shape[0]) + "; n++){\n\t\tloop_for_ap_" + str(
                        self.index) + ":\n\t\tfor (int y = 0; y < " + str(
                        out_shape[1]) + "; y++){\n\t\t\t" + self.type + " s = 0;\n\t\t\tloop_for_fc_" + str(
                        self.index) + ":\n\t\t\tfor (int k = 0; k < " + str(
                        kernel_shape[1]) + "; k++){\n\t\t\t\tloop_for_fa_" + str(
                        self.index) + ":\n\t\t\t\tfor (int j = 0; j < " + str(
                        kernel_shape[2]) + "; j++){\n\t\t\t\t\ts=s+(kernel[" + str(kernel_shape[1]) + "*" + str(
                        kernel_shape[2]) + "*n+" + str(
                        kernel_shape[
                            2]) + "*k+j])*(Input_Conv[" + str(
                        in_shape[1]) + "*k+j+y*stride]);}\n\t\t\t}\n\t\t\t" + self.act_arr + "\n\t\t}\n\t}\n}\n")

                    if self.model.layers[i].bias is None:
                        source_Conv_hh = self.fxp_inc + "void Conv1D_" + str(self.index) + "(" + self.type + " Input_Conv[" + str(in_shape[0] * in_shape[1]) + "]," + self.type + " Output_Conv[" + str(out_shape[0] * out_shape[1]) + "], " + self.type + " kernel[" + str(kernel_shape[0] * kernel_shape[1] * kernel_shape[2]) + "]);\n"
                    else:
                        source_Conv_hh = self.fxp_inc + "void Conv1D_" + str(
                            self.index) + "(" + self.type + " Input_Conv[" + str(
                            in_shape[0] * in_shape[1]) + "]," + self.type + " Output_Conv[" + str(
                            out_shape[0] * out_shape[1]) + "], " + self.type + " bias[" + str(
                            out_shape[0]) + "], " + self.type + " kernel[" + str(
                            kernel_shape[0] * kernel_shape[1] * kernel_shape[2]) + "]);\n"

                self.full_source_Conv_cc.append(source_Conv_cc)
                self.full_source_Conv_hh.append(source_Conv_hh)
                if self.model.layers[i].bias is None:
                    self.full_source_CNN_cc.append(
                        ["\tConv1D_" + str(self.index) + "(", self.config["layers"][i]['config']['name'], "&Weights[" + str(self.cnt_param) + "]"])
                    self.cnt_param += kernel_shape[0] * kernel_shape[1] * kernel_shape[2]
                else:
                    self.full_source_CNN_cc.append(
                        ["\tConv1D_" + str(self.index) + "(", self.config["layers"][i]['config']['name'], "&Weights[" + str(
                            self.cnt_param + kernel_shape[0] * kernel_shape[1] * kernel_shape[2]) + "]",
                         "&Weights[" + str(self.cnt_param) + "]"])
                    self.cnt_param += kernel_shape[0] * kernel_shape[1] * kernel_shape[2] + out_shape[0]

                self.index += 1

            if (layer.find("Add") >= 0) and (layer.find("Pad") < 0):
                found = 1
                if self.type == "fxp" and self.index_P2D == 0:
                    self.fxp_inc = self.fxp_include
                else:
                    self.fxp_inc = ""
                if len(self.model.layers[i].input_shape[0]) == 4:
                    NumOfInput = len(self.model.layers[i].input_shape)
                    source_add = ""
                    argument_add = ""
                    size = self.model.layers[i].input_shape[0][1] * self.model.layers[i].input_shape[0][2] * \
                           self.model.layers[i].input_shape[0][3]
                    for k in range(0, NumOfInput):
                        source_add += "input_" + str(k) + "[i]"
                        argument_add += str(self.type) + " input_" + str(k) + "[" + str(size) + "], "
                        if k < (NumOfInput - 1):
                            source_add += " + "
                elif len(self.model.layers[i].input_shape[0]) == 3:
                    NumOfInput = len(self.model.layers[i].input_shape)
                    source_add = ""
                    argument_add = ""
                    size = self.model.layers[i].input_shape[0][1] * self.model.layers[i].input_shape[0][2]
                    for k in range(0, NumOfInput):
                        source_add += "input_" + str(k) + "[i]"
                        argument_add += str(self.type) + " input_" + str(k) + "[" + str(size) + "], "
                        if k < (NumOfInput - 1):
                            source_add += " + "
                else:
                    assert ((len(self.model.layers[i].input_shape[0]) == 3) or (len(self.model.layers[i].input_shape[0]) == 4)), "add layer hasn't supported 1 dimension yet"

                self.call_function += "\t" + str(self.type) + " " + self.config["layers"][i]['config'][
                    'name'] + "[" + str(size) + "];\n"
                source_Conv_cc = "void Add_" + str(self.indexAdd) + "(" + argument_add + str(
                    self.type) + " output[" + str(size) + "]) {" + "\n\tfor (int i = 0; i < " + str(
                    size) + "; i++){\n\t\toutput[i] = " + source_add + ";\n\t}\n}\n"
                source_Conv_hh = "void Add_" + str(self.indexAdd) + "(" + argument_add + str(
                    self.type) + " output[" + str(size) + "]);\n"
                self.full_source_CNN_cc.append(
                    ["\tAdd_" + str(self.indexAdd) + "(", self.config["layers"][i]['config']['name']])
                self.full_source_Conv_cc.append(source_Conv_cc)
                self.full_source_Conv_hh.append(source_Conv_hh)
                self.indexAdd += 1

            if layer.find("Concatenate") >= 0:
                found = 1
                if self.type == "fxp" and self.index_P2D == 0:
                    self.fxp_inc = self.fxp_include
                else:
                    self.fxp_inc = ""
                dimesion_input = len(self.model.layers[i].input_shape[0])
                axis = self.model.layers[i].axis
                assert (((dimesion_input == 3) and (axis == -1)) or ((dimesion_input == 4) and (axis == 3))) , "concatenate layer only supported 2 dimesion input combine with axis = -1 and 3 dimesion input combine with axis = 3 yet!!!"

                if len(self.model.layers[i].input_shape[0]) == 4:
                    NumOfInput = len(self.model.layers[i].input_shape)
                    source = ""
                    argument = ""
                    num_param = 0
                    for k in range(0, NumOfInput):
                        size = self.model.layers[i].input_shape[k][1] * self.model.layers[i].input_shape[k][2] * self.model.layers[i].input_shape[k][3]
                        source += "\tfor (int i = 0; i < " + str(size) + ";i++){\n\t\toutput[" + str(num_param) + " + i] = input_" + str(k) + "[i];\n\t}\n"
                        argument += str(self.type) + " input_" + str(k) + "[" + str(size) + "], "
                        num_param += size
                    self.call_function += "\t" + str(self.type) + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(self.model.layers[i].output_shape[1] * self.model.layers[i].output_shape[2] * self.model.layers[i].output_shape[3]) + "];\n"


                if len(self.model.layers[i].input_shape[0]) == 3:
                    NumOfInput = len(self.model.layers[i].input_shape)
                    source = ""
                    argument = ""
                    num_param = 0
                    for k in range(0, NumOfInput):
                        size = self.model.layers[i].input_shape[k][1] * self.model.layers[i].input_shape[k][2]
                        source += "\tfor (int i = 0; i < " + str(size) + ";i++){\n\t\toutput[" + str(num_param) + " + i] = input_" + str(k) + "[i];\n\t}\n"
                        argument += str(self.type) + " input_" + str(k) + "[" + str(size) + "], "
                        num_param += size
                    self.call_function += "\t" + str(self.type) + " " + self.config["layers"][i]['config']['name'] + "[" + str(self.model.layers[i].output_shape[1] * self.model.layers[i].output_shape[2]) + "];\n"

                source_Conv_cc = self.fxp_inc + " void Concatenate_" + str(
                    self.indexConcatenate) + "(" + argument + str(self.type) + " output[" + str(
                    self.model.layers[i].output_shape[1] * self.model.layers[i].output_shape[
                        2]) + "]) {\n" + source + "\n}\n"
                source_Conv_hh = "void Concatenate_" + str(self.indexConcatenate) + "(" + argument + str(
                    self.type) + " output[" + str(
                    self.model.layers[i].output_shape[1] * self.model.layers[i].output_shape[2]) + "]);\n"
                self.full_source_CNN_cc.append(
                    ["\tConcatenate_" + str(self.indexConcatenate) + "(", self.config["layers"][i]['config']['name']])
                self.full_source_Conv_cc.append(source_Conv_cc)
                self.full_source_Conv_hh.append(source_Conv_hh)
                self.indexConcatenate += 1

            if layer.find("BatchNormalization") >= 0:
                found = 1
                if self.type == "fxp" and self.index_P2D == 0:
                    self.fxp_inc = self.fxp_include
                else:
                    self.fxp_inc = ""
                # 3D, In convolutional layer
                if (len(self.model.layers[i].input.shape) == 4):
                    in_shape = (
                    self.model.layers[i].input.shape[depth_index], self.model.layers[i].input.shape[height_index],
                    self.model.layers[i].input.shape[width_index])
                    out_shape = (
                    self.model.layers[i].output.shape[depth_index], self.model.layers[i].output.shape[height_index],
                    self.model.layers[i].output.shape[width_index])
                    gamma = self.model.layers[i].get_weights()[0]
                    beta = self.model.layers[i].get_weights()[1]
                    moving_mean = self.model.layers[i].get_weights()[2]
                    moving_variance = self.model.layers[i].get_weights()[3]
                    for k in gamma:
                        self.Weights.append(k)
                    for k in beta:
                        self.Weights.append(k)
                    for k in moving_mean:
                        self.Weights.append(k)
                    for k in moving_variance:
                        self.Weights.append(k)

                    self.call_function += "\t" + str(self.type) + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(in_shape[0] * in_shape[1] * in_shape[2]) + "];\n"
                    type_sqrt = ""
                    if self.type == "fxp":
                        source_Conv_cc = "#include <hls_math.h>\n"
                        type_sqrt = "hls::sqrt"
                    else:
                        source_Conv_cc = "#include <cmath>\n"
                        type_sqrt = "sqrt"

                    if (depth_index == 3) :
                        source_Conv_cc += self.fxp_inc + " void BatchNorm2D_" + str(self.indexBatch) + "(" + str(
                            self.type) + " Input_BatchNorm[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "], " + str(
                            self.type) + " Output_BatchNorm[" + str(
                            out_shape[0] * out_shape[1] * out_shape[2]) + "], " + str(self.type) + " gamma[" + str(
                            len(gamma)) + "], " + str(self.type) + " beta[" + str(len(beta)) + "], " + str(
                            self.type) + " MovMean[" + str(len(moving_mean)) + "], " + str(
                            self.type) + " MovVar[" + str(
                            len(moving_variance)) + "]) {" + "\n\t" + self.type + " eps = " + str(
                            self.model.layers[i].epsilon) + ";\n\t for(int i = 0; i < " + str(
                            in_shape[0]) + "; i++){\n\t\tfor(int j = 0; j < " + str(
                            in_shape[1] * in_shape[2]) + "; j++){" + "\n\t\t\t Output_BatchNorm[" + str(
                            in_shape[1] * in_shape[2]) + " * i + j] = ((Input_BatchNorm[" + str(in_shape[1] * in_shape[
                            2]) + " * i + j] - MovMean[i]) / (" + type_sqrt + "(MovVar[i] + eps))) * gamma[i] + beta[i];\n\t\t}\n\t}\n}\n"
                    else:
                        source_Conv_cc += self.fxp_inc + " void BatchNorm2D_" + str(self.indexBatch) + "(" + str(self.type) + " Input_BatchNorm[" + str(in_shape[0] * in_shape[1] * in_shape[2]) + "], " + str(self.type) + " Output_BatchNorm[" + str(out_shape[0] * out_shape[1] * out_shape[2]) + "], " + str(self.type) + " gamma[" + str(len(gamma)) + "], " + str(self.type) + " beta[" + str(len(beta)) + "], " + str(self.type) + " MovMean[" + str(len(moving_mean)) + "], " + str(self.type) + " MovVar[" + str(len(moving_variance)) + "]) {" + "\n\t" + self.type + " eps = " + str(self.model.layers[i].epsilon) + ";\n\t for(int i = 0; i < " + str(in_shape[0]) + "; i++){\n\t\tfor(int j = 0; j < " + str(in_shape[1]) + "; j++){\n\t\t\tfor(int k = 0; k < " + str(in_shape[2]) + "; k++){" + "\n\t\t\t\t Output_BatchNorm[" + str(in_shape[1] * in_shape[2]) + " * i + " + str(in_shape[1]) + " * j + k] = ((Input_BatchNorm[" + str(in_shape[1] * in_shape[2]) + " * i + " + str(in_shape[1]) + " * j + k] - MovMean[k]) / (" + type_sqrt + "(MovVar[k] + eps))) * gamma[k] + beta[k];\n\t\t\t}\n\t\t}\n\t}\n}\n"

                    source_Conv_hh = "void BatchNorm2D_" + str(self.indexBatch) + "(" + str(
                        self.type) + " Input_BatchNorm[" + str(
                        in_shape[0] * in_shape[1] * in_shape[2]) + "], " + str(
                        self.type) + " Output_BatchNorm[" + str(
                        in_shape[0] * in_shape[1] * in_shape[2]) + "], " + str(self.type) + " gamma[" + str(
                        len(gamma)) + "], " + str(self.type) + " beta[" + str(len(beta)) + "], " + str(
                        self.type) + " MovMean[" + str(len(moving_mean)) + "], " + str(
                        self.type) + " MovVar[" + str(len(moving_variance)) + "]);\n"
                    self.full_source_CNN_cc.append(
                        ["\tBatchNorm2D_" + str(self.indexBatch) + "(", self.config["layers"][i]['config']['name'],
                         "&Weights[" + str(self.cnt_param) + "]",
                         "&Weights[" + str(self.cnt_param + len(gamma)) + "]",
                         "&Weights[" + str(self.cnt_param + len(gamma) + len(beta)) + "]",
                         "&Weights[" + str(self.cnt_param + len(gamma) + len(beta) + len(moving_mean)) + "]"])
                    self.full_source_Conv_cc.append(source_Conv_cc)
                    self.full_source_Conv_hh.append(source_Conv_hh)
                    self.indexBatch += 1
                    self.cnt_param += len(gamma) + len(beta) + len(moving_mean) + len(moving_variance)

                # 3D and 2D, in FC
                if (len(self.model.layers[i].input.shape) == 2):
                    in_shape = self.model.layers[i].input.shape[1]
                    out_shape = self.model.layers[i].output.shape[1]
                    gamma = self.model.layers[i].get_weights()[0]
                    beta = self.model.layers[i].get_weights()[1]
                    moving_mean = self.model.layers[i].get_weights()[2]
                    moving_variance = self.model.layers[i].get_weights()[3]
                    for k in gamma:
                        self.Weights.append(k)
                    for k in beta:
                        self.Weights.append(k)
                    for k in moving_mean:
                        self.Weights.append(k)
                    for k in moving_variance:
                        self.Weights.append(k)
                    self.call_function += "\t" + str(self.type) + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(in_shape) + "];\n"

                    if self.type == "fxp":
                        source_Conv_cc = "#include <hls_math.h>\n"
                        type_sqrt = "hls::sqrt"
                    else:
                        source_Conv_cc = "#include <cmath>\n"
                        type_sqrt = "sqrt"

                    source_Conv_cc += self.fxp_inc + " void BatchNorm2D_" + str(self.indexBatch) + "(" + str(
                        self.type) + " Input_BatchNorm[" + str(in_shape) + "], " + str(
                        self.type) + " Output_BatchNorm[" + str(out_shape) + "], " + str(self.type) + " gamma[" + str(
                        len(gamma)) + "]," + str(self.type) + " beta[" + str(len(beta)) + "], " + str(
                        self.type) + " MovMean[" + str(len(moving_mean)) + "], " + str(self.type) + " MovVar[" + str(
                        len(moving_variance)) + "]) {" + "\n\t" + self.type + " eps = " + str(
                        self.model.layers[i].epsilon) + ";\n\t for(int i = 0; i < " + str(
                        in_shape) + "; i++){\n\t\tOutput_BatchNorm[i] = ((Input_BatchNorm[i] - MovMean[i]) / (" + type_sqrt + "(MovVar[i] + eps)))* gamma[i] + beta[i];\n\t}\n}\n"
                    source_Conv_hh = "void BatchNorm2D_" + str(self.indexBatch) + "(" + str(
                        self.type) + " Input_BatchNorm[" + str(in_shape) + "], " + str(
                        self.type) + " Output_BatchNorm[" + str(out_shape) + "], " + str(self.type) + " gamma[" + str(
                        len(gamma)) + "], " + str(self.type) + " beta[" + str(len(beta)) + "], " + str(
                        self.type) + " MovMean[" + str(len(moving_mean)) + "], " + str(self.type) + " MovVar[" + str(
                        len(moving_variance)) + "]);\n"
                    self.full_source_CNN_cc.append(
                        ["\tBatchNorm2D_" + str(self.indexBatch) + "(", self.config["layers"][i]['config']['name'],
                         "&Weights[" + str(self.cnt_param) + "]",
                         "&Weights[" + str(self.cnt_param + len(gamma)) + "]",
                         "&Weights[" + str(self.cnt_param + len(gamma) + len(beta)) + "]",
                         "&Weights[" + str(self.cnt_param + len(gamma) + len(beta) + len(moving_mean)) + "]"])
                    self.full_source_Conv_cc.append(source_Conv_cc)
                    self.full_source_Conv_hh.append(source_Conv_hh)
                    self.indexBatch += 1
                    self.cnt_param += len(gamma) + len(beta) + len(moving_mean) + len(moving_variance)

                # 2D, In Convolutional layer
                if (len(self.model.layers[i].input.shape) == 3):
                    in_shape = (self.model.layers[i].input.shape[height_index], self.model.layers[i].input.shape[width_index])
                    out_shape = (self.model.layers[i].output.shape[height_index], self.model.layers[i].output.shape[width_index])
                    gamma = self.model.layers[i].get_weights()[0]
                    beta = self.model.layers[i].get_weights()[1]
                    moving_mean = self.model.layers[i].get_weights()[2]
                    moving_variance = self.model.layers[i].get_weights()[3]
                    for k in gamma:
                        self.Weights.append(k)
                    for k in beta:
                        self.Weights.append(k)
                    for k in moving_mean:
                        self.Weights.append(k)
                    for k in moving_variance:
                        self.Weights.append(k)
                    self.call_function += "\t" + str(self.type) + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(in_shape[0] * in_shape[1]) + "];\n"

                    type_sqrt = ""
                    if self.type == "fxp":
                        source_Conv_cc = "#include <hls_math.h>\n"
                        type_sqrt = "hls::sqrt"
                    else:
                        source_Conv_cc = "#include <cmath>\n"
                        type_sqrt = "sqrt"

                    source_Conv_cc += self.fxp_inc + " void BatchNorm1D_" + str(self.indexBatch) + "(" + str(
                        self.type) + " Input_BatchNorm[" + str(in_shape[0] * in_shape[1]) + "], " + str(
                        self.type) + " Output_BatchNorm[" + str(out_shape[0] * out_shape[1]) + "], " + str(
                        self.type) + " gamma[" + str(len(gamma)) + "]," + str(self.type) + " beta[" + str(
                        len(beta)) + "], " + str(self.type) + " MovMean[" + str(len(moving_mean)) + "], " + str(
                        self.type) + " MovVar[" + str(
                        len(moving_variance)) + "]) {" + "\n\t" + self.type + " eps = " + str(
                        self.model.layers[i].epsilon) + ";\n\t for(int i = 0; i < " + str(
                        in_shape[1]) + "; i++){\n\t\tfor(int j = 0; j < " + str(
                        in_shape[0]) + "; j++){\n\t\t\tOutput_BatchNorm[" + str(
                        in_shape[0]) + " * i + j] = ((Input_BatchNorm[" + str(in_shape[
                                                                                  0]) + " * i + j] - MovMean[i]) / (" + type_sqrt + "(MovVar[i] + eps)))* gamma[i] + beta[i];\n\t\t}\n\t}\n}\n"
                    source_Conv_hh = "void BatchNorm1D_" + str(self.indexBatch) + "(" + str(
                        self.type) + " Input_BatchNorm[" + str(in_shape[0] * in_shape[1]) + "], " + str(
                        self.type) + " Output_BatchNorm[" + str(out_shape[0] * out_shape[1]) + "], " + str(
                        self.type) + " gamma[" + str(len(gamma)) + "], " + str(self.type) + " beta[" + str(
                        len(beta)) + "], " + str(self.type) + " MovMean[" + str(len(moving_mean)) + "], " + str(
                        self.type) + " MovVar[" + str(len(moving_variance)) + "]);\n"
                    self.full_source_CNN_cc.append(
                        ["\tBatchNorm1D_" + str(self.indexBatch) + "(", self.config["layers"][i]['config']['name'],
                         "&Weights[" + str(self.cnt_param) + "]", "&Weights[" + str(self.cnt_param + len(gamma)) + "]",
                         "&Weights[" + str(self.cnt_param + len(gamma) + len(beta)) + "]",
                         "&Weights[" + str(self.cnt_param + len(gamma) + len(beta) + len(moving_mean)) + "]"])
                    self.full_source_Conv_cc.append(source_Conv_cc)
                    self.full_source_Conv_hh.append(source_Conv_hh)
                    self.indexBatch += 1
                    self.cnt_param += len(gamma) + len(beta) + len(moving_mean) + len(moving_variance)

            if layer.find("Activation") >= 0 or layer.find("ReLU") >= 0:
                found = 1
                # 3D and 1D, FC layer
                if len(self.model.layers[i].input.shape) == 2:
                    in_shape = self.model.layers[i].input.shape[1]
                    out_shape = self.model.layers[i].output.shape[1]
                    self.call_function += "\t" + str(self.type) + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(in_shape) + "];\n"
                    if layer.find("ReLU") >= 0:
                        activation = "relu"
                    else:
                        activation = self.config["layers"][i]['config']['activation']


                    if self.type == "fxp" and self.index == 0:
                        self.fxp_inc = self.fxp_include
                    else:
                        self.fxp_inc = ""
                    if self.type == "fxp":
                        if activation == "sigmoid" or activation == "tanh":
                            self.fxp_inc += "#include <hls_math.h>\n"
                        else:
                            self.fxp_inc += ""
                        type_sqrt = "hls::sqrt"
                    else:
                        if activation == "sigmoid" or activation == "tanh":
                            self.fxp_inc += "#include <cmath>\n"
                        else:
                            self.fxp_inc += ""
                        type_sqrt = "sqrt"
                    source = ""
                    if activation == "sigmoid":
                        source = "Output_Activation[i] = 1/(1 + " + type_sqrt + "(-Input_Activation[i]));"
                    elif activation == "relu":
                        source = "if(Input_Activation[i] > 0){\n\t\t\tOutput_Activation[i] = Input_Activation[i];\n\t\t}else\n\t\t{\n\t\t\tOutput_Activation[i] = 0;\n\t\t}"
                    elif activation == "tanh":
                        source = "Output_Activation[i]=(2/(1 + " + type_sqrt + "(-2*Input_Activation[i])))-1"
                    else:
                        source = "Output_Activation[i]=Input_Activation[i]"


                    source_Conv_cc = self.fxp_inc + " void Activation" + str(self.indexActi) + "(" + str(
                        self.type) + " Input_Activation[" + str(in_shape) + "], " + str(
                        self.type) + " Output_Activation[" + str(out_shape) + "]){\n\tfor (int i = 0; i < " + str(
                        out_shape) + "; i++){\n\t\t" + source + "\n\t}\n}\n"
                    source_Conv_hh = self.fxp_inc + "void Activation" + str(self.indexActi) + "(" + str(
                        self.type) + " Input_Activation[" + str(in_shape) + "], " + str(
                        self.type) + " Output_Activation[" + str(out_shape) + "]);\n"
                    self.full_source_CNN_cc.append(
                        ["\tActivation" + str(self.indexActi) + "(", self.config["layers"][i]['config']['name']])
                    self.full_source_Conv_cc.append(source_Conv_cc)
                    self.full_source_Conv_hh.append(source_Conv_hh)
                    self.indexActi += 1

                # 3D , Convolutional Layer
                if len(self.model.layers[i].input.shape) == 4:
                    in_shape = (
                    self.model.layers[i].input.shape[depth_index], self.model.layers[i].input.shape[height_index],
                    self.model.layers[i].input.shape[width_index])
                    out_shape = (
                    self.model.layers[i].output.shape[depth_index], self.model.layers[i].output.shape[height_index],
                    self.model.layers[i].output.shape[width_index])
                    self.call_function += "\t" + str(self.type) + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(out_shape[0] * out_shape[1] * out_shape[2]) + "];\n"
                    if layer.find("ReLU") >= 0:
                        activation = "relu"
                    else:
                        activation = self.config["layers"][i]['config']['activation']
                    if self.type == "fxp" and self.index == 0:
                        self.fxp_inc = self.fxp_include
                    else:
                        self.fxp_inc = ""
                    if self.type == "fxp":
                        if activation == "sigmoid" or activation == "tanh":
                            self.fxp_inc += "#include <hls_math.h>\n"
                        else:
                            self.fxp_inc += ""
                        type_sqrt = "hls::sqrt"
                    else:
                        if activation == "sigmoid" or activation == "tanh":
                            self.fxp_inc += "#include <cmath>\n"
                        else:
                            self.fxp_inc += ""
                        type_sqrt = "sqrt"
                    source = ""
                    if activation == "sigmoid":
                        source = "Output_Activation[i] = 1/(1 + " + type_sqrt + "(-Input_Activation[i]));"
                    elif activation == "relu":
                        source = "if(Input_Activation[i] > 0){\n\t\t\tOutput_Activation[i] = Input_Activation[i];\n\t\t}else\n\t\t{\n\t\t\tOutput_Activation[i] = 0;\n\t\t}"
                    elif activation == "tanh":
                        source = "Output_Activation[i]=(2/(1 + " + type_sqrt + "(-2*Input_Activation[i])))-1"
                    else:
                        source = "Output_Activation[i]=Input_Activation[i]"
                    source_Conv_cc = self.fxp_inc + " void Activation" + str(self.indexActi) + "(" + str(
                        self.type) + " Input_Activation[" + str(in_shape[0] * in_shape[1] * in_shape[2]) + "], " + str(
                        self.type) + " Output_Activation[" + str(
                        out_shape[0] * out_shape[1] * out_shape[2]) + "]){\n\tfor (int i = 0; i < " + str(
                        out_shape[0] * out_shape[1] * out_shape[
                            2]) + "; i++){\n\t\t" + source + "\n\t}\n}\n"
                    source_Conv_hh = self.fxp_inc + "void Activation" + str(self.indexActi) + "(" + str(
                        self.type) + " Input_Activation[" + str(in_shape[0] * in_shape[1] * in_shape[2]) + "], " + str(
                        self.type) + " Output_Activation[" + str(out_shape[0] * out_shape[1] * out_shape[2]) + "]);\n"
                    self.full_source_CNN_cc.append(
                        ["\tActivation" + str(self.indexActi) + "(", self.config["layers"][i]['config']['name']])
                    self.full_source_Conv_cc.append(source_Conv_cc)
                    self.full_source_Conv_hh.append(source_Conv_hh)
                    self.indexActi += 1

                # 1D, Convolutional Layer
                if len(self.model.layers[i].input.shape) == 3:
                    in_shape = (self.model.layers[i].input.shape[height_index], self.model.layers[i].input.shape[width_index])
                    out_shape = (self.model.layers[i].output.shape[height_index], self.model.layers[i].output.shape[width_index])
                    if layer.find("ReLU") >= 0:
                        activation = "relu"
                    else:
                        activation = self.config["layers"][i]['config']['activation']

                    if self.type == "fxp" and self.index == 0:
                        self.fxp_inc = self.fxp_include
                    else:
                        self.fxp_inc = ""
                    if self.type == "fxp":
                        if activation == "sigmoid" or activation == "tanh":
                            self.fxp_inc += "#include <hls_math.h>\n"
                        else:
                            self.fxp_inc += ""
                        type_sqrt = "hls::sqrt"
                    else:
                        if activation == "sigmoid" or activation == "tanh":
                            self.fxp_inc += "#include <cmath>\n"
                        else:
                            self.fxp_inc += ""
                        type_sqrt = "sqrt"
                    source = ""
                    if activation == "sigmoid":
                        source = "Output_Activation[i] = 1/(1 + " + type_sqrt + "(-Input_Activation[i]));"
                    elif activation == "relu":
                        source = "if(Input_Activation[i] > 0){\n\t\t\tOutput_Activation[i] = Input_Activation[i];\n\t\t}else\n\t\t{\n\t\t\tOutput_Activation[i] = 0;\n\t\t}"
                    elif activation == "tanh":
                        source = "Output_Activation[i]=(2/(1 + " + type_sqrt + "(-2*Input_Activation[i])))-1;"
                    else:
                        source = "Output_Activation[i]=Input_Activation[i]"
                    self.call_function += "\t" + str(self.type) + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(out_shape[0] * out_shape[1]) + "];\n"

                    source_Conv_cc = self.fxp_inc + " void Activation" + str(self.indexActi) + "(" + str(
                        self.type) + " Input_Activation[" + str(in_shape[0] * in_shape[1]) + "], " + str(
                        self.type) + " Output_Activation[" + str(
                        out_shape[0] * out_shape[1]) + "]){\n\tfor (int i = 0; i < " + str(out_shape[0] * out_shape[
                        1]) + "; i++){\n\t\t" + source + "\n\t}\n}\n"
                    source_Conv_hh = self.fxp_inc + "void Activation" + str(self.indexActi) + "(" + str(
                        self.type) + " Input_Activation[" + str(in_shape[0] * in_shape[1]) + "], " + str(
                        self.type) + " Output_Activation[" + str(out_shape[0] * out_shape[1]) + "]);\n"
                    self.full_source_CNN_cc.append(
                        ["\tActivation" + str(self.indexActi) + "(", self.config["layers"][i]['config']['name']])
                    self.full_source_Conv_cc.append(source_Conv_cc)
                    self.full_source_Conv_hh.append(source_Conv_hh)
                    self.indexActi += 1

            if layer.find("AveragePooling2D") >= 0:
                if layer.find("global_average_pooling2d") < 0 and layer.find("GlobalAveragePooling2D") < 0:
                    found = 1
                    in_shape = (
                    self.model.layers[i].input.shape[depth_index], self.model.layers[i].input.shape[height_index],
                    self.model.layers[i].input.shape[width_index])
                    out_shape = (
                    self.model.layers[i].output.shape[depth_index], self.model.layers[i].output.shape[height_index],
                    self.model.layers[i].output.shape[width_index])

                    strides = self.model.layers[i].strides[0]
                    poolSize = self.model.layers[i].pool_size[0]
                    if (in_shape[1] == 1):
                        strides = 1
                        poolSize = 1
                    if self.type == "fxp" and self.index_P2D == 0:
                        self.fxp_inc = self.fxp_include
                    else:
                        self.fxp_inc = ""


                    if self.config["layers"][i]['config']['padding'] == 'valid':
                        self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config']['name'] + "[" + str(out_shape[0] * out_shape[1] * out_shape[2]) + "];\n"
                        source_Pool_cc = self.fxp_inc + "void average_Pool2D_" + str(
                            self.index_P2D) + "(" + self.type + " input_AveragePooling[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "], " + self.type + " output_AveragePooling[" + str(
                            out_shape[0] * out_shape[1] * out_shape[2]) + "]){\n\tint PoolSize = " + str(
                            poolSize) + ";\n\tint stride = " + str(
                            strides) + ";\n\tint index = 0;\n\tfor (int i = 0; i < " + str(
                            out_shape[0]) + "; i++){\n\t\tindex = 0;\n\t\tfor (int z = 0; z < " + str(
                            out_shape[1]) + "; z++){\n\t\t\tfor (int y = 0; y < " + str(
                            out_shape[2]) + "; y++){\n\t\t\t\t" + str(self.type) + " Average = 0;\n\t\t\t\t" + str(
                            self.type) + " Pool_value = 0;\n\t\t\t\tfor (int h = 0; h < PoolSize; h++){\n\t\t\t\t\tfor (int w = 0; w < PoolSize; w++){\n\t\t\t\t\t\tint Pool_index = " + str(
                            in_shape[1]) + " * " + str(in_shape[2]) + " * i + " + str(in_shape[1]) + " * h + " + str(
                            in_shape[2]) + " * stride * z + w + y * stride;\n\t\t\t\t\t\tPool_value += input_AveragePooling[Pool_index];" + "\n\t\t\t\t\t\tAverage = Pool_value / " + str(
                            poolSize * poolSize) + ";\n\t\t\t\t\t}\n\t\t\t\t}" + "\n\t\t\t\tint outIndex = " + str(
                            out_shape[1]) + " * " + str(out_shape[2]) + " * i + index;\n\t\t\t\toutput_AveragePooling[outIndex] = Average;\n\t\t\t\tindex++;" + "\n\t\t\t}\n\t\t}\n\t}\n}\n"
                        source_Pool_hh = self.fxp_inc + "void average_Pool2D_" + str(
                            self.index_P2D) + "(" + self.type + " input_AveragePooling[" + str(
                            in_shape[0] * (in_shape[1] + 2)) + "], " + self.type + " output_AveragePooling[" + str(
                            out_shape[0] * out_shape[1]) + "]);\n"
                        self.full_source_CNN_cc.append(["\taverage_Pool2D_" + str(self.index_P2D) + "(",
                                                        self.config["layers"][i]['config']['name'], "", ""])
                        self.full_source_Pool_cc.append(source_Pool_cc)
                        self.full_source_Pool_hh.append(source_Pool_hh)
                    else:
                        if self.config["layers"][i]['config']['padding'] == 'same':
                            pad = math.floor((out_shape[1]*strides-(in_shape[1]-poolSize))/2)

                            # pad = 0.5
                            self.call_function += "\t" + self.type + " " + "OutPadPool" + str(
                                self.index_P2D) + "[" + str(
                                int(in_shape[0] * (in_shape[1] + 2 * pad) * (in_shape[1] + 2 * pad))) + "];\n"
                            self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                                'name'] + "[" + str(out_shape[0] * out_shape[1] * out_shape[2]) + "];\n"
                            source_pad_pool_cc = "void Padding_Pool2D_" + str(
                                self.index_P2D) + "(" + self.type + " input_Pad_Pool[" + str(
                                in_shape[0] * in_shape[1] * in_shape[
                                    2]) + "], " + self.type + " output_Pad_Pool[" + str(int(
                                in_shape[0] * (in_shape[1] + 2 * pad) * (
                                            in_shape[2] + 2 * pad))) + "]){\n\tloop_for_3_channel_pad_" + str(
                                self.index_P2D) + ":\n\tfor (int c = 0; c < " + str(
                                in_shape[0]) + "; c++){" + "\n\t\tloop_for_channel_pad_" + str(
                                self.index_P2D) + ":\n\t\tfor (int n = 0; n < " + str(
                                int(in_shape[1] + 2 * pad)) + "; n++){\n\t\t\tloop_for_weight_pad_" + str(
                                self.index_P2D) + ":\n\t\t\tfor (int i = 0; i < " + str(
                                int(in_shape[2] + 2 * pad)) + "; i++){\n\t\t\t\t"

                            if pad < 1:
                                if pad == 0:
                                    source_pad_pool_cc += "output_Pad_Pool[" + str(
                                        int(in_shape[2] + 2 * pad)) + " * " + str(
                                        int(in_shape[1] + 2 * pad)) + " * c + " + str(
                                        int(in_shape[1] + 2 * pad)) + " * n + i] = input_Pad_Pool[" + str(
                                        int(in_shape[2])) + " * " + str(int(in_shape[1])) + " * c + " + str(
                                        int(in_shape[1])) + " * n + i];\n\t\t\t}\n\t\t}\n\t}\n}\n"
                                else:
                                    source_pad_pool_cc += "if (n >= " + str(
                                        int(in_shape[1])) + ") output_Pad_Pool[" + str(
                                        int(in_shape[2] + 2 * pad)) + " * " + str(
                                        int(in_shape[1] + 2 * pad)) + " * c + " + str(
                                        int(in_shape[1] + 2 * pad)) + " * n + i] = 0; else if (i >= " + str(
                                        int(in_shape[2])) + ") output_Pad_Pool[" + str(
                                        int(in_shape[2] + 2 * pad)) + " * " + str(
                                        int(in_shape[1] + 2 * pad)) + " * c + " + str(
                                        int(in_shape[1] + 2 * pad)) + " * n + i] = 0; else output_Pad_Pool[" + str(
                                        int(in_shape[2] + 2 * pad)) + " * " + str(
                                        int(in_shape[1] + 2 * pad)) + " * c + " + str(
                                        int(in_shape[1] + 2 * pad)) + " * n + i] = input_Pad_Pool[" + str(
                                        int(in_shape[2])) + " * " + str(int(in_shape[1])) + " * c + " + str(
                                        int(in_shape[1])) + " * n + i];\n\t\t\t}\n\t\t}\n\t}\n}\n"
                            else:
                                source_pad_pool_cc += "if (n < " + str(pad) + " || n >= " + str(
                                    in_shape[1] + pad) + ") output_Pad_Pool[" + str(
                                    in_shape[2] + 2 * pad) + " * " + str(in_shape[1] + 2 * pad) + " * c + " + str(
                                    in_shape[1] + 2 * pad) + " * n + i] = 0; else if (i < " + str(
                                    pad) + " || i >= " + str(in_shape[2] + pad) + ") output_Pad_Pool[" + str(
                                    in_shape[2] + 2 * pad) + " * " + str(in_shape[1] + 2 * pad) + " * c + " + str(
                                    in_shape[1] + 2 * pad) + " * n + i] = 0; else output_Pad_Pool[" + str(
                                    in_shape[2] + 2 * pad) + " * " + str(in_shape[1] + 2 * pad) + " * c + " + str(
                                    in_shape[1] + 2 * pad) + " * n + i] = input_Pad_Pool[" + str(
                                    in_shape[2]) + " * " + str(in_shape[1]) + " * c + " + str(
                                    in_shape[1]) + " * n + i - " + str(2 * pad) + "];\n\t\t\t}\n\t\t}\n\t}\n}\n"

                            source_pad_pool_hh = "void Padding_Pool2D_" + str(
                                self.index_P2D) + "(" + self.type + " input_Pad_Pool[" + str(
                                in_shape[0] * in_shape[1] * in_shape[
                                    2]) + "], " + self.type + " output_Pad_Pool[" + str(
                                int(in_shape[0] * (in_shape[1] + 2 * pad) * (in_shape[2] + 2 * pad))) + "]);\n"
                            source_Pool_cc = self.fxp_inc + "void average_Pool2D_" + str(
                                self.index_P2D) + "(" + self.type + " input_AveragePooling[" + str(int(
                                in_shape[0] * (in_shape[1] + 2 * pad) * (in_shape[
                                                                             2] + 2 * pad))) + "], " + self.type + " output_AveragePooling[" + str(
                                out_shape[0] * out_shape[1] * out_shape[2]) + "]){\n\tint PoolSize = " + str(
                                poolSize) + ";\n\tint stride = " + str(
                                strides) + ";\n\tint index = 0;\n\tfor (int i = 0; i < " + str(
                                out_shape[0]) + "; i++){\n\t\tindex = 0;\n\t\tfor (int z = 0; z < " + str(
                                out_shape[1]) + "; z++){\n\t\t\tfor (int y = 0; y < " + str(
                                out_shape[2]) + "; y++){\n\t\t\t\t" + str(self.type) + " Average = 0;\n\t\t\t\t" + str(
                                self.type) + " Pool_value = 0;\n\t\t\t\tfor (int h = 0; h < PoolSize; h++){\n\t\t\t\t\tfor (int w = 0; w < PoolSize; w++){\n\t\t\t\t\t\tint Pool_index = " + str(
                                int(in_shape[1] + 2 * pad)) + " * " + str(int(in_shape[2] + 2 * pad)) + " * i + " + str(
                                int(in_shape[1] + 2 * pad)) + " * h + " + str(int(in_shape[
                                                                                      2] + 2 * pad)) + " * stride * z + w + y * stride;\n\t\t\t\t\t\tPool_value += input_AveragePooling[Pool_index];" + "\n\t\t\t\t\t\tAverage = Pool_value / " + str(
                                poolSize * poolSize) + ";\n\t\t\t\t\t}\n\t\t\t\t}" + "\n\t\t\t\tint outIndex = " + str(
                                out_shape[1]) + " * " + str(out_shape[
                                                                2]) + " * i + index;\n\t\t\t\toutput_AveragePooling[outIndex] = Average;\n\t\t\t\tindex++;" + "\n\t\t\t}\n\t\t}\n\t}\n}\n"
                            source_Pool_hh = self.fxp_inc + "void average_Pool2D_" + str(
                                self.index_P2D) + "(" + self.type + " input_AveragePooling[" + str(int(
                                in_shape[0] * (in_shape[1] + 2 * pad) * (in_shape[
                                                                             2] + 2 * pad))) + "], " + self.type + " output_AveragePooling[" + str(
                                out_shape[0] * out_shape[1] * out_shape[2]) + "]);\n"

                            self.full_source_CNN_cc.append(
                                ["\tPadding_Pool2D_" + str(self.index_P2D) + "(", "OutPadPool" + str(self.index_P2D),
                                 "", ""])
                            self.full_source_CNN_cc.append(["\taverage_Pool2D_" + str(self.index_P2D) + "(",
                                                            self.config["layers"][i]['config']['name'], "", ""])
                            self.full_source_Pool_cc.append(source_Pool_cc)
                            self.full_source_Pool_hh.append(source_Pool_hh)
                            self.full_source_Pool_cc.append(source_pad_pool_cc)
                            self.full_source_Pool_hh.append(source_pad_pool_hh)
                    self.index_P2D += 1

            if layer.find("MaxPooling2D") >= 0:
                if layer.find("global_max_pooling2d") < 0 and layer.find("GlobalMaxPooling2D") < 0:
                    found = 1
                    in_shape = (
                    self.model.layers[i].input.shape[depth_index], self.model.layers[i].input.shape[height_index],
                    self.model.layers[i].input.shape[width_index])
                    out_shape = (
                    self.model.layers[i].output.shape[depth_index], self.model.layers[i].output.shape[height_index],
                    self.model.layers[i].output.shape[width_index])
                    strides = self.model.layers[i].strides[0]
                    poolSize = self.model.layers[i].pool_size[0]
                    if self.type == "fxp" and self.index_P2D == 0:
                        self.fxp_inc = self.fxp_include
                    else:
                        self.fxp_inc = ""



                    if self.config["layers"][i]['config']['padding'] == 'valid':
                        self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                            'name'] + "[" + str(out_shape[0] * out_shape[1] * out_shape[2]) + "];\n"
                        source_Pool_cc = self.fxp_inc + "void Max_Pool2D_" + str(
                            self.index_P2D) + "(" + self.type + " input_MaxPooling[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "], " + self.type + " output_MaxPooling[" + str(
                            out_shape[0] * out_shape[1] * out_shape[2]) + "]){\n\tint PoolSize = " + str(
                            poolSize) + ";\n\tint stride = " + str(
                            strides) + ";\n\tint index = 0;\n\tfor (int i = 0; i < " + str(
                            out_shape[0]) + "; i++){\n\t\tindex = 0;\n\t\tfor (int z = 0; z < " + str(
                            out_shape[1]) + "; z++){\n\t\t\tfor (int y = 0; y < " + str(out_shape[
                                                                                            2]) + "; y++){\n\t\t\t\t" + self.type + " max_c = -10;\n\t\t\t\tfor (int h = 0; h < PoolSize; h++){\n\t\t\t\t\tfor (int w = 0; w < PoolSize; w++){\n\t\t\t\t\t\tint Pool_index = " + str(
                            in_shape[1]) + " * " + str(in_shape[2]) + " * i + " + str(in_shape[1]) + " * h + " + str(
                            in_shape[
                                2]) + " * stride * z + w + y * stride;\n\t\t\t\t\t\t" + self.type + " Pool_value = input_MaxPooling[Pool_index];" + "\n\t\t\t\t\t\tif (Pool_value >= max_c) max_c = Pool_value;" + "\n\t\t\t\t\t}\n\t\t\t\t}" + "\n\t\t\t\tint outIndex = " + str(
                            out_shape[1]) + " * " + str(out_shape[
                                                            2]) + " * i + index;\n\t\t\t\toutput_MaxPooling[outIndex] = max_c;\n\t\t\t\tindex++;" + "\n\t\t\t}\n\t\t}\n\t}\n}\n"
                        source_Pool_hh = self.fxp_inc + "void Max_Pool2D_" + str(
                            self.index_P2D) + "(" + self.type + " input_MaxPooling[" + str(
                            in_shape[0] * (in_shape[1] + 2)) + "], " + self.type + " output_MaxPooling[" + str(
                            out_shape[0] * out_shape[1]) + "]);\n"
                        self.full_source_CNN_cc.append(
                            ["\tMax_Pool2D_" + str(self.index_P2D) + "(", self.config["layers"][i]['config']['name'],
                             "", ""])
                        self.full_source_Pool_cc.append(source_Pool_cc)
                        self.full_source_Pool_hh.append(source_Pool_hh)
                    else:
                        if self.config["layers"][i]['config']['padding'] == 'same':
                            pad = math.floor((out_shape[1]*strides-(in_shape[1]-poolSize))/2)
                            # pad = 0.5
                            self.call_function += "\t" + self.type + " " + "OutPadPool" + str(
                                self.index_P2D) + "[" + str(
                                int(in_shape[0] * (in_shape[1] + 2 * pad) * (in_shape[1] + 2 * pad))) + "];\n"
                            self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                                'name'] + "[" + str(out_shape[0] * out_shape[1] * out_shape[2]) + "];\n"
                            source_pad_pool_cc = "void Padding_Pool2D_" + str(
                                self.index_P2D) + "(" + self.type + " input_Pad_Pool[" + str(
                                in_shape[0] * in_shape[1] * in_shape[
                                    2]) + "], " + self.type + " output_Pad_Pool[" + str(int(
                                in_shape[0] * (in_shape[1] + 2 * pad) * (
                                            in_shape[2] + 2 * pad))) + "]){\n\tloop_for_3_channel_pad_" + str(
                                self.index_P2D) + ":\n\tfor (int c = 0; c < " + str(
                                in_shape[0]) + "; c++){" + "\n\t\tloop_for_channel_pad_" + str(
                                self.index_P2D) + ":\n\t\tfor (int n = 0; n < " + str(
                                int(in_shape[1] + 2 * pad)) + "; n++){\n\t\t\tloop_for_weight_pad_" + str(
                                self.index_P2D) + ":\n\t\t\tfor (int i = 0; i < " + str(
                                int(in_shape[2] + 2 * pad)) + "; i++){\n\t\t\t\t"

                            if pad < 1:
                                source_pad_pool_cc += "if (n >= " + str(int(in_shape[1])) + ") output_Pad_Pool[" + str(
                                    int(in_shape[2] + 2 * pad)) + " * " + str(
                                    int(in_shape[1] + 2 * pad)) + " * c + " + str(
                                    int(in_shape[1] + 2 * pad)) + " * n + i] = 0; else if (i >= " + str(
                                    int(in_shape[2])) + ") output_Pad_Pool[" + str(
                                    int(in_shape[2] + 2 * pad)) + " * " + str(
                                    int(in_shape[1] + 2 * pad)) + " * c + " + str(
                                    int(in_shape[1] + 2 * pad)) + " * n + i] = 0; else output_Pad_Pool[" + str(
                                    int(in_shape[2] + 2 * pad)) + " * " + str(
                                    int(in_shape[1] + 2 * pad)) + " * c + " + str(
                                    int(in_shape[1] + 2 * pad)) + " * n + i] = input_Pad_Pool[" + str(
                                    int(in_shape[2])) + " * " + str(int(in_shape[1])) + " * c + " + str(
                                    int(in_shape[1])) + " * n + i];\n\t\t\t}\n\t\t}\n\t}\n}\n"
                            else:
                                source_pad_pool_cc += "if (n < " + str(int(pad)) + " || n >= " + str(int(in_shape[1] + pad)) + ") output_Pad_Pool[" + str(int(in_shape[2] + 2 * pad)) + " * " + str(int(in_shape[1] + 2 * pad)) + " * c + " + str(int(in_shape[1] + 2 * pad)) + " * n + i] = 0; else if (i < " + str(int(pad)) + " || i >= " + str(int(in_shape[2] + pad)) + ") output_Pad_Pool[" + str(int(in_shape[2] + 2 * pad)) + " * " + str(int(in_shape[1] + 2 * pad)) + " * c + " + str(int(in_shape[1] + 2 * pad)) + " * n + i] = 0; else output_Pad_Pool[" + str(
                                    int(in_shape[2] + 2 * pad)) + " * " + str(int(in_shape[1] + 2 * pad)) + " * c + " + str(
                                    int(in_shape[1] + 2 * pad)) + " * n + i] = input_Pad_Pool[" + str(
                                    int(in_shape[2])) + " * " + str(int(in_shape[1])) + " * c + " + str(
                                    int(in_shape[1])) + " * (n - " + str(int(pad)) + ") + i - " + str(int(pad)) + "];\n\t\t\t}\n\t\t}\n\t}\n}\n"

                            source_pad_pool_hh = "void Padding_Pool2D_" + str(
                                self.index_P2D) + "(" + self.type + " input_Pad_Pool[" + str(
                                in_shape[0] * in_shape[1] * in_shape[
                                    2]) + "], " + self.type + " output_Pad_Pool[" + str(
                                int(in_shape[0] * (in_shape[1] + 2 * pad) * (in_shape[2] + 2 * pad))) + "]);\n"
                            source_Pool_cc = self.fxp_inc + "void Max_Pool2D_" + str(
                                self.index_P2D) + "(" + self.type + " input_MaxPooling[" + str(int(
                                in_shape[0] * (in_shape[1] + 2 * pad) * (
                                            in_shape[2] + 2 * pad))) + "], " + self.type + " output_MaxPooling[" + str(
                                out_shape[0] * out_shape[1] * out_shape[2]) + "]){\n\tint PoolSize = " + str(
                                poolSize) + ";\n\tint stride = " + str(
                                strides) + ";\n\tint index = 0;\n\tfor (int i = 0; i < " + str(
                                out_shape[0]) + "; i++){\n\t\tindex = 0;\n\t\tfor (int z = 0; z < " + str(
                                out_shape[1]) + "; z++){\n\t\t\tfor (int y = 0; y < " + str(out_shape[
                                                                                                2]) + "; y++){\n\t\t\t\t" + self.type + " max_c = -10;\n\t\t\t\tfor (int h = 0; h < PoolSize; h++){\n\t\t\t\t\tfor (int w = 0; w < PoolSize; w++){\n\t\t\t\t\t\tint Pool_index = " + str(
                                int(in_shape[1] + 2 * pad)) + " * " + str(int(in_shape[2] + 2 * pad)) + " * i + " + str(
                                int(in_shape[1] + 2 * pad)) + " * h + " + str(int(in_shape[
                                                                                      2] + 2 * pad)) + " * stride * z + w + y * stride;\n\t\t\t\t\t\t" + self.type + " Pool_value = input_MaxPooling[Pool_index];" + "\n\t\t\t\t\t\tif (Pool_value >= max_c) max_c = Pool_value;" + "\n\t\t\t\t\t}\n\t\t\t\t}" + "\n\t\t\t\tint outIndex = " + str(
                                out_shape[1]) + " * " + str(out_shape[
                                                                2]) + " * i + index;\n\t\t\t\toutput_MaxPooling[outIndex] = max_c;\n\t\t\t\tindex++;" + "\n\t\t\t}\n\t\t}\n\t}\n}\n"
                            source_Pool_hh = self.fxp_inc + "void Max_Pool2D_" + str(
                                self.index_P2D) + "(" + self.type + " input_MaxPooling[" + str(int(
                                in_shape[0] * (in_shape[1] + 2 * pad) * (
                                            in_shape[2] + 2 * pad))) + "], " + self.type + " output_MaxPooling[" + str(
                                out_shape[0] * out_shape[1] * out_shape[2]) + "]);\n"

                            self.full_source_CNN_cc.append(
                                ["\tPadding_Pool2D_" + str(self.index_P2D) + "(", "OutPadPool" + str(self.index_P2D),
                                 "", ""])
                            self.full_source_CNN_cc.append(["\tMax_Pool2D_" + str(self.index_P2D) + "(",
                                                            self.config["layers"][i]['config']['name'], "", ""])
                            self.full_source_Pool_cc.append(source_Pool_cc)
                            self.full_source_Pool_hh.append(source_Pool_hh)
                            self.full_source_Pool_cc.append(source_pad_pool_cc)
                            self.full_source_Pool_hh.append(source_pad_pool_hh)
                        # else:
                        #     self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                        #         'name'] + "[" + str(out_shape[0] * out_shape[1] * out_shape[2]) + "];\n"
                        #     source_Pool_cc = self.fxp_inc + "void Max_Pool2D_" + str(
                        #         self.index_P2D) + "(" + self.type + " input_MaxPooling[" + str(
                        #         in_shape[0] * in_shape[1] * in_shape[
                        #             2]) + "], " + self.type + " output_MaxPooling[" + str(
                        #         out_shape[0] * out_shape[1] * out_shape[2]) + "]){\n\tint PoolSize = " + str(
                        #         poolSize) + ";\n\tint stride = " + str(
                        #         strides) + ";\n\tint index = 0;\n\tfor (int i = 0; i < " + str(
                        #         out_shape[0]) + "; i++){\n\t\tindex = 0;\n\t\tfor (int z = 0; z < " + str(
                        #         out_shape[1]) + "; z++){\n\t\t\tfor (int y = 0; y < " + str(out_shape[
                        #                                                                         2]) + "; y++){\n\t\t\t\t" + self.type + " max_c = -10;\n\t\t\t\tfor (int h = 0; h < PoolSize; h++){\n\t\t\t\t\tfor (int w = 0; w < PoolSize; w++){\n\t\t\t\t\t\tint Pool_index = " + str(
                        #         in_shape[1]) + " * " + str(in_shape[2]) + " * i + " + str(
                        #         in_shape[1]) + " * h + " + str(in_shape[
                        #                                            2]) + " * stride * z + w + y * stride;\n\t\t\t\t\t\t" + self.type + " Pool_value = input_MaxPooling[Pool_index];" + "\n\t\t\t\t\t\tif (Pool_value >= max_c) max_c = Pool_value;" + "\n\t\t\t\t\t}\n\t\t\t\t}" + "\n\t\t\t\tint outIndex = " + str(
                        #         out_shape[1]) + " * " + str(out_shape[
                        #                                         2]) + " * i + index;\n\t\t\t\toutput_MaxPooling[outIndex] = max_c;\n\t\t\t\tindex++;" + "\n\t\t\t}\n\t\t}\n\t}\n}\n"
                        #     source_Pool_hh = self.fxp_inc + "void Max_Pool2D_" + str(
                        #         self.index_P2D) + "(" + self.type + " input_MaxPooling[" + str(
                        #         in_shape[0] * (in_shape[1] + 2)) + "], " + self.type + " output_MaxPooling[" + str(
                        #         out_shape[0] * out_shape[1]) + "]);\n"
                        #     self.full_source_CNN_cc.append(["\tMax_Pool2D_" + str(self.index_P2D) + "(",
                        #                                     self.config["layers"][i]['config']['name'], "", ""])
                        #     self.full_source_Pool_cc.append(source_Pool_cc)
                        #     self.full_source_Pool_hh.append(source_Pool_hh)

                    self.index_P2D += 1

            # convert max_pooling1d layer into c array that act like an max_pooling1d layer
            if layer.find("MaxPooling1D") >= 0:
                if layer.find("GlobalMaxPooling1D") < 0:
                    found = 1
                    in_shape = (self.model.layers[i].input.shape[width_index], self.model.layers[i].input.shape[height_index])
                    out_shape = (self.model.layers[i].output.shape[width_index], self.model.layers[i].output.shape[height_index])
                    if self.type == "fxp" and self.index_P == 0:
                        self.fxp_inc = self.fxp_include
                    else:
                        self.fxp_inc = ""
                    strides = self.model.layers[i].strides[0]
                    poolSize = self.model.layers[i].pool_size[0]

                    if self.config["layers"][i]['config']['padding'] == 'same':
                        padding = ((out_shape[1] - 1) * strides - (in_shape[1] - poolSize)) / 2
                        in_shape_if_padding = (in_shape[0], (in_shape[1] + 2 * padding))

                        if padding - math.floor(padding) != 0.5:
                            source_pad_pool_cc = self.fxp_inc + "void Padding_Pool_" + str(
                                self.index_P) + "(" + self.type + " input_Pad_Pool[" + str(
                                in_shape[0] * in_shape[1]) + "], " + self.type + " output_Pad_Pool[" + str(int(
                                in_shape_if_padding[0] * in_shape_if_padding[
                                    1])) + "]){\n\tloop_for_3_channel_pad_" + str(
                                self.index_P) + ":\n\tfor (int c = 0; c < " + str(
                                int(in_shape_if_padding[0])) + "; c++){" + "\n\t\tloop_for_channel_pad_" + str(
                                self.index_P) + ":\n\t\tfor (int n = 0; n < " + str(
                                int(in_shape_if_padding[1])) + "; n++){\n\t\t\tif (n < " + str(
                                int(0 + padding)) + " || n >= " + str(
                                int(in_shape[1] + padding)) + ") output_Pad_Pool[" + str(
                                int(in_shape_if_padding[1])) + " * c + n]=0; else output_Pad_Pool[" + str(
                                int(in_shape_if_padding[1])) + " * c + n] = input_Pad_Pool[" + str(
                                in_shape[1]) + " * c + n - " + str(int(padding)) + "];\n\t\t}\n\t}\n}\n"
                            source_pad_pool_hh = self.fxp_inc + "void Padding_Pool_" + str(
                                self.index_P) + "(" + self.type + " input_Pad_Pool[" + str(
                                in_shape[0] * in_shape[1]) + "], " + self.type + " output_Pad_Pool[" + str(
                                int(in_shape_if_padding[0] * in_shape_if_padding[1])) + "]);\n"
                            self.full_source_Pool_cc.append(source_pad_pool_cc)
                            self.full_source_Pool_hh.append(source_pad_pool_hh)
                            self.call_function += "\t" + self.type + " OutPadPool" + str(self.index_P) + "[" + str(
                                int(in_shape_if_padding[0] * in_shape_if_padding[1])) + "];\n"
                            self.full_source_CNN_cc.append(
                                ["\tPadding_Pool_" + str(self.index_P) + "(", "OutPadPool" + str(self.index_P), "",
                                 ""])
                        else:
                            source_pad_pool_cc = self.fxp_inc + "void Padding_Pool_" + str(
                                self.index_P) + "(" + self.type + " input_Pad_Pool[" + str(
                                in_shape[0] * in_shape[1]) + "], " + self.type + " output_Pad_Pool[" + str(int(
                                in_shape_if_padding[0] * in_shape_if_padding[
                                    1])) + "]){\n\tloop_for_3_channel_pad_" + str(
                                self.index_P) + ":\n\tfor (int c = 0; c < " + str(
                                int(in_shape_if_padding[0])) + "; c++){" + "\n\t\tloop_for_channel_pad_" + str(
                                self.index_P) + ":\n\t\tfor (int n = 0; n < " + str(
                                int(in_shape_if_padding[1])) + "; n++){\n\t\t\tif (n >= " + str(
                                int(in_shape[1] + 2 * padding)) + ") output_Pad_Pool[" + str(
                                int(in_shape_if_padding[1])) + " * c + n]=0; else output_Pad_Pool[" + str(
                                int(in_shape_if_padding[1])) + " * c + n] = input_Pad_Pool[" + str(
                                in_shape[1]) + " * c + n];\n\t\t}\n\t}\n}\n"
                            source_pad_pool_hh = self.fxp_inc + "void Padding_Pool_" + str(
                                self.index_P) + "(" + self.type + " input_Pad_Pool[" + str(
                                in_shape[0] * in_shape[1]) + "], " + self.type + " output_Pad_Pool[" + str(
                                int(in_shape_if_padding[0] * in_shape_if_padding[1])) + "]);\n"
                            self.full_source_Pool_cc.append(source_pad_pool_cc)
                            self.full_source_Pool_hh.append(source_pad_pool_hh)
                            self.call_function += "\t" + self.type + " OutPadPool" + str(self.index_P) + "[" + str(
                                int(in_shape_if_padding[0] * in_shape_if_padding[1])) + "];\n"
                            self.full_source_CNN_cc.append(
                                ["\tPadding_Pool_" + str(self.index_P) + "(", "OutPadPool" + str(self.index_P), "",
                                 ""])

                        source_Pool_cc = self.fxp_inc + "void Max_Pool1D_" + str(self.index_P) + "(" + self.type + " input_MaxPooling[" + str(int(in_shape_if_padding[0] * in_shape_if_padding[1])) + "], " + self.type + " output_MaxPooling[" + str(out_shape[0] * out_shape[1]) + "]){\n\tint PoolSize = " + str(poolSize) + ";\n\tint stride= " + str(strides) + ";\n\tint index = 0;\n\tloop_for_channel_pool_" + str(self.index_P) + ":\n\tfor (int z = 0; z < " + str(out_shape[0]) + "; z++){\n\t\tindex = 0;\n\t\tloop_for_weight_pool_" + str(self.index_P) + ":\n\t\tfor (int y = 0; y < " + str(out_shape[
                                                                                    1]) + "; y++){\n\t\t\t" + self.type + " max = -10;\n\t\t\tfor (int j = 0; j < PoolSize; j++)\n\t\t\t{\n\t\t\t\tint pool_index = " + str(
                            int(in_shape_if_padding[
                                    1])) + " * z + j + y * stride;\n\t\t\t\t" + self.type + " pool_value = input_MaxPooling[pool_index];\n\t\t\t\tif (pool_value > max) max=pool_value;\n\t\t\t}\n\t\t\tint out_index = " + str(
                            out_shape[
                                1]) + " * z + index;\n\t\t\toutput_MaxPooling[out_index]=max;\n\t\t\tindex++;\n\t\t}\n\t}\n}\n"
                        source_Pool_hh = self.fxp_inc + "void Max_Pool1D_" + str(
                            self.index_P) + "(" + self.type + " input_MaxPooling[" + str(
                            int(in_shape_if_padding[0] * in_shape_if_padding[
                                1])) + "], " + self.type + " output_MaxPooling[" + str(
                            out_shape[0] * out_shape[1]) + "]);\n"
                        self.full_source_Pool_cc.append(source_Pool_cc)
                        self.full_source_Pool_hh.append(source_Pool_hh)

                        self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                            'name'] + "[" + str(
                            out_shape[0] *
                            out_shape[1]) + "];\n"
                        self.full_source_CNN_cc.append(
                            ["\tMax_Pool1D_" + str(self.index_P) + "(", self.config["layers"][i]['config']['name'], "",
                             ""])

                    else:
                        source_Pool_cc = self.fxp_inc + "void Max_Pool1D_" + str(
                            self.index_P) + "(" + self.type + " input_MaxPooling[" + str(
                            in_shape[0] * in_shape[1]) + "], " + self.type + " output_MaxPooling[" + str(
                            out_shape[0] * out_shape[1]) + "]){\n\tint PoolSize = " + str(
                            poolSize) + ";\n\tint stride= " + str(
                            strides) + ";\n\tint index = 0;\n\tloop_for_channel_pool_" + str(
                            self.index_P) + ":\n\tfor (int z = 0; z < " + str(
                            out_shape[0]) + "; z++){\n\t\tindex = 0;\n\t\tloop_for_weight_pool_" + str(
                            self.index_P) + ":\n\t\tfor (int y = 0; y < " + str(out_shape[
                                                                                    1]) + "; y++){\n\t\t\t" + self.type + " max = -10;\n\t\t\tfor (int j = 0; j < PoolSize; j++)\n\t\t\t{\n\t\t\t\tint pool_index = " + str(
                            in_shape[
                                1]) + " * z + j + y * stride;\n\t\t\t\t" + self.type + " pool_value = input_MaxPooling[pool_index];\n\t\t\t\tif (pool_value > max) max=pool_value;\n\t\t\t}\n\t\t\tint out_index = " + str(
                            out_shape[
                                1]) + " * z + index;\n\t\t\toutput_MaxPooling[out_index]=max;\n\t\t\tindex++;\n\t\t}\n\t}\n}\n"
                        source_Pool_hh = self.fxp_inc + "void Max_Pool1D_" + str(
                            self.index_P) + "(" + self.type + " input_MaxPooling[" + str(
                            in_shape[0] * (in_shape[1])) + "], " + self.type + " output_MaxPooling[" + str(
                            out_shape[0] * out_shape[1]) + "]);\n"
                        self.full_source_Pool_cc.append(source_Pool_cc)
                        self.full_source_Pool_hh.append(source_Pool_hh)

                        self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                            'name'] + "[" + str(
                            out_shape[0] *
                            out_shape[1]) + "];\n"
                        self.full_source_CNN_cc.append(
                            ["\tMax_Pool1D_" + str(self.index_P) + "(", self.config["layers"][i]['config']['name'], "",
                             ""])

                    self.index_P += 1

            if layer.find("AveragePooling1D") >= 0:
                if layer.find("GlobalAveragePooling1D") < 0:
                    found = 1
                    in_shape = (self.model.layers[i].input.shape[width_index], self.model.layers[i].input.shape[height_index])
                    out_shape = (self.model.layers[i].output.shape[width_index], self.model.layers[i].output.shape[height_index])
                    if self.type == "fxp" and self.index_P == 0:
                        self.fxp_inc = self.fxp_include
                    else:
                        self.fxp_inc = ""
                    strides = self.model.layers[i].strides[0]
                    poolSize = self.model.layers[i].pool_size[0]
                    if self.config["layers"][i]['config']['padding'] == 'same':
                        padding = ((out_shape[1] - 1) * strides - (in_shape[1] - poolSize)) / 2
                        in_shape_if_padding = (in_shape[0], (in_shape[1] + 2 * padding))

                        if padding - math.floor(padding) != 0.5:
                            source_pad_pool_cc = self.fxp_inc + "void Padding_Pool_" + str(
                                self.index_P) + "(" + self.type + " input_Pad_Pool[" + str(
                                in_shape[0] * in_shape[1]) + "], " + self.type + " output_Pad_Pool[" + str(int(
                                in_shape_if_padding[0] * in_shape_if_padding[
                                    1])) + "]){\n\tloop_for_3_channel_pad_" + str(
                                self.index_P) + ":\n\tfor (int c = 0; c < " + str(
                                int(in_shape_if_padding[0])) + "; c++){" + "\n\t\tloop_for_channel_pad_" + str(
                                self.index_P) + ":\n\t\tfor (int n = 0; n < " + str(
                                int(in_shape_if_padding[1])) + "; n++){\n\t\t\tif (n < " + str(
                                int(0 + padding)) + " || n >= " + str(
                                int(in_shape[1] + padding)) + ") output_Pad_Pool[" + str(
                                int(in_shape_if_padding[1])) + " * c + n]=0; else output_Pad_Pool[" + str(
                                int(in_shape_if_padding[1])) + " * c + n] = input_Pad_Pool[" + str(
                                in_shape[1]) + " * c + n - " + str(int(padding)) + "];\n\t\t}\n\t}\n}\n"
                            source_pad_pool_hh = self.fxp_inc + "void Padding_Pool_" + str(
                                self.index_P) + "(" + self.type + " input_Pad_Pool[" + str(
                                in_shape[0] * in_shape[1]) + "], " + self.type + " output_Pad_Pool[" + str(
                                int(in_shape_if_padding[0] * in_shape_if_padding[1])) + "]);\n"
                            self.full_source_Pool_cc.append(source_pad_pool_cc)
                            self.full_source_Pool_hh.append(source_pad_pool_hh)
                            self.call_function += "\t" + self.type + " OutPadPool" + str(self.index_P) + "[" + str(
                                int(in_shape_if_padding[0] * in_shape_if_padding[1])) + "];\n"
                            self.full_source_CNN_cc.append(
                                ["\tPadding_Pool_" + str(self.index_P) + "(", "OutPadPool" + str(self.index_P), "",
                                 ""])
                        else:
                            source_pad_pool_cc = self.fxp_inc + "void Padding_Pool_" + str(
                                self.index_P) + "(" + self.type + " input_Pad_Pool[" + str(
                                in_shape[0] * in_shape[1]) + "], " + self.type + " output_Pad_Pool[" + str(int(
                                in_shape_if_padding[0] * in_shape_if_padding[
                                    1])) + "]){\n\tloop_for_3_channel_pad_" + str(
                                self.index_P) + ":\n\tfor (int c = 0; c < " + str(
                                int(in_shape_if_padding[0])) + "; c++){" + "\n\t\tloop_for_channel_pad_" + str(
                                self.index_P) + ":\n\t\tfor (int n = 0; n < " + str(
                                int(in_shape_if_padding[1])) + "; n++){\n\t\t\tif (n >= " + str(
                                int(in_shape[1] + 2 * padding)) + ") output_Pad_Pool[" + str(
                                int(in_shape_if_padding[1])) + " * c + n]=0; else output_Pad_Pool[" + str(
                                int(in_shape_if_padding[1])) + " * c + n] = input_Pad_Pool[" + str(
                                in_shape[1]) + " * c + n];\n\t\t}\n\t}\n}\n"
                            source_pad_pool_hh = self.fxp_inc + "void Padding_Pool_" + str(
                                self.index_P) + "(" + self.type + " input_Pad_Pool[" + str(
                                in_shape[0] * in_shape[1]) + "], " + self.type + " output_Pad_Pool[" + str(
                                int(in_shape_if_padding[0] * in_shape_if_padding[1])) + "]);\n"
                            self.full_source_Pool_cc.append(source_pad_pool_cc)
                            self.full_source_Pool_hh.append(source_pad_pool_hh)
                            self.call_function += "\t" + self.type + " OutPadPool" + str(self.index_P) + "[" + str(
                                int(in_shape_if_padding[0] * in_shape_if_padding[1])) + "];\n"
                            self.full_source_CNN_cc.append(
                                ["\tPadding_Pool_" + str(self.index_P) + "(", "OutPadPool" + str(self.index_P), "",
                                 ""])

                        source_Pool_cc = self.fxp_inc + "void Average_Pool1D_" + str(
                            self.index_P) + "(" + self.type + " input_AveragePooling[" + str(
                            int(in_shape_if_padding[0] * in_shape_if_padding[1])) + "], " + self.type + " output_AveragePooling[" + str(
                            out_shape[0] * out_shape[1]) + "]){\n\tint PoolSize = " + str(
                            poolSize) + ";\n\tint stride= " + str(
                            strides) + ";\n\tint index = 0;\n\tloop_for_channel_pool_" + str(
                            self.index_P) + ":\n\tfor (int z = 0; z < " + str(
                            out_shape[0]) + "; z++){\n\t\tindex = 0;\n\t\tloop_for_weight_pool_" + str(
                            self.index_P) + ":\n\t\tfor (int y = 0; y < " + str(out_shape[1]) + "; y++){\n\t\t\t" + self.type + " Average = 0;\n\t\t\t" + self.type + " pool_value = 0;\n\t\t\tfor (int j = 0; j < PoolSize; j++)\n\t\t\t{\n\t\t\t\tint pool_index = " + str(int(in_shape_if_padding[1])) + " * z + j + y * stride;\n\t\t\t\tpool_value += input_AveragePooling[pool_index];\n\t\t\t}\n\t\t\tAverage = pool_value / PoolSize;\n\t\t\tint out_index = " + str(
                            out_shape[
                                1]) + " * z + index;\n\t\t\toutput_AveragePooling[out_index]=Average;\n\t\t\tindex++;\n\t\t}\n\t}\n}\n"
                        source_Pool_hh = self.fxp_inc + "void Average_Pool1D_" + str(
                            self.index_P) + "(" + self.type + " input_AveragePooling[" + str(
                            int(in_shape_if_padding[0] * (in_shape_if_padding[1]))) + "], " + self.type + " output_AveragePooling[" + str(
                            out_shape[0] * out_shape[1]) + "]);\n"
                        self.full_source_Pool_cc.append(source_Pool_cc)
                        self.full_source_Pool_hh.append(source_Pool_hh)
                        self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                            'name'] + "[" + str(
                            out_shape[0] *
                            out_shape[1]) + "];\n"
                        self.full_source_CNN_cc.append(
                            ["\tAverage_Pool1D_" + str(self.index_P) + "(", self.config["layers"][i]['config']['name'], "",
                             ""])

                    else:
                        source_Pool_cc = self.fxp_inc + "void Average_Pool1D_" + str(
                            self.index_P) + "(" + self.type + " input_AveragePooling[" + str(
                            in_shape[0] * in_shape[1]) + "], " + self.type + " output_AveragePooling[" + str(
                            out_shape[0] * out_shape[1]) + "]){\n\tint PoolSize = " + str(
                            poolSize) + ";\n\tint stride= " + str(
                            strides) + ";\n\tint index = 0;\n\tloop_for_channel_pool_" + str(
                            self.index_P) + ":\n\tfor (int z = 0; z < " + str(
                            out_shape[0]) + "; z++){\n\t\tindex = 0;\n\t\tloop_for_weight_pool_" + str(
                            self.index_P) + ":\n\t\tfor (int y = 0; y < " + str(out_shape[
                                                                                    1]) + "; y++){\n\t\t\t" + self.type + " Average = 0;\n\t\t\t" + self.type + " pool_value = 0;\n\t\t\tfor (int j = 0; j < PoolSize; j++)\n\t\t\t{\n\t\t\t\tint pool_index = " + str(
                            in_shape[
                                1]) + " * z + j + y * stride;\n\t\t\t\tpool_value += input_AveragePooling[pool_index];\n\t\t\t}\n\t\t\tAverage = pool_value / PoolSize;\n\t\t\tint out_index = " + str(
                            out_shape[
                                1]) + " * z + index;\n\t\t\toutput_AveragePooling[out_index]=Average;\n\t\t\tindex++;\n\t\t}\n\t}\n}\n"
                        source_Pool_hh = self.fxp_inc + "void Average_Pool1D_" + str(
                            self.index_P) + "(" + self.type + " input_AveragePooling[" + str(
                            in_shape[0] * (in_shape[1])) + "], " + self.type + " output_AveragePooling[" + str(
                            out_shape[0] * out_shape[1]) + "]);\n"
                        self.full_source_Pool_cc.append(source_Pool_cc)
                        self.full_source_Pool_hh.append(source_Pool_hh)
                        self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                            'name'] + "[" + str(
                            out_shape[0] *
                            out_shape[1]) + "];\n"
                        self.full_source_CNN_cc.append(
                            ["\tAverage_Pool1D_" + str(self.index_P) + "(", self.config["layers"][i]['config']['name'], "",
                             ""])

                    self.index_P += 1

            if layer.find("GlobalMaxPooling2D") >= 0:
                if len(self.model.layers[i].input.shape) == 4:
                    found = 1
                    in_shape = (
                    self.model.layers[i].input.shape[depth_index], self.model.layers[i].input.shape[height_index],
                    self.model.layers[i].input.shape[width_index])
                    out_shape = (self.model.layers[i].output.shape[1])

                    source_Flatten_hh = "void GlobalMaxPool2D_" + str(self.index_GlbMaxPool) + "(" + self.type + " input_GlobalMaxPool2D[" + str(
                        in_shape[0] * in_shape[1] * in_shape[2]) + "]," + self.type + " output_GlobalMaxPool2D[" + str(
                        out_shape) + "]);\n"
                    source_Flatten_cc = "void GlobalMaxPool2D_" + str(self.index_GlbMaxPool) + "(" + self.type + " input_GlobalMaxPool2D[" + str(
                        in_shape[0] * in_shape[1] * in_shape[2]) + "]," + self.type + " output_GlobalMaxPool2D[" + str(
                        out_shape) + "]){\n\tint hs = 0;\n\tfor (int i = 0; i < " + str(
                        in_shape[0]) + "; i++){\n\t\t" + self.type + " max = -10;\n\t\tfor (int j = 0; j < " + str(
                        in_shape[1]) + "; j++){\n\t\t\tfor (int k = 0; k < " + str(
                        in_shape[2]) + "; k++){\n\t\t\t\tif (input_GlobalMaxPool2D[" + str(in_shape[1]) + " * " + str(
                        in_shape[2]) + " * i + " + str(
                        in_shape[2]) + " * j + k] >= max) max = input_GlobalMaxPool2D[" + str(
                        in_shape[1]) + " * " + str(in_shape[2]) + " * i + " + str(in_shape[
                                                                                      2]) + " * j + k];\n\t\t\t}\n\t\t}\n\t\toutput_GlobalMaxPool2D[hs] = max;\n\t\ths++;\n\t}\n}\n"
                    self.full_source_Pool_cc.append(source_Flatten_cc)
                    self.full_source_Pool_hh.append(source_Flatten_hh)
                    self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(out_shape) + "];\n"
                    self.full_source_CNN_cc.append(
                        ["\tGlobalMaxPool2D_" + str(self.index_GlbMaxPool) + "(", self.config["layers"][i]['config']['name'], "", ""])
                    self.index_GlbMaxPool += 1

            if layer.find("GlobalAveragePooling2D") >= 0:
                if len(self.model.layers[i].input.shape) == 4:
                    found = 1
                    in_shape = (
                    self.model.layers[i].input.shape[depth_index], self.model.layers[i].input.shape[height_index],
                    self.model.layers[i].input.shape[width_index])
                    out_shape = (self.model.layers[i].output.shape[1])
                    # source_Flatten_cc = "void"
                    source_Flatten_hh = "void GlobalAveragePool2D_" + str(self.index_GlbAvgPool) + "(" + self.type + " input_GlobalAveragePool2D[" + str(
                        in_shape[0] * in_shape[1] * in_shape[
                            2]) + "]," + self.type + " output_GlobalAveragePool2D[" + str(out_shape) + "]);\n"
                    source_Flatten_cc = "void GlobalAveragePool2D_" + str(self.index_GlbAvgPool) + "(" + self.type + " input_GlobalAveragePool2D[" + str(
                        in_shape[0] * in_shape[1] * in_shape[
                            2]) + "]," + self.type + " output_GlobalAveragePool2D[" + str(
                        out_shape) + "]){\n\tint hs = 0;\n\tfor (int i = 0; i < " + str(
                        in_shape[0]) + "; i++){\n\t\t" + self.type + " avg = 0;\n\t\tfor (int j = 0; j < " + str(
                        in_shape[1]) + "; j++){\n\t\t\tfor (int k = 0; k < " + str(
                        in_shape[2]) + "; k++){\n\t\t\t\tavg += input_GlobalAveragePool2D[" + str(
                        in_shape[1]) + " * " + str(in_shape[2]) + " * i + " + str(
                        in_shape[2]) + " * j + k];\n\t\t\t}\n\t\t}\n\t\toutput_GlobalAveragePool2D[hs] = avg / (" + str(
                        in_shape[1]) + " * " + str(in_shape[2]) + ") ;\n\t\ths++;\n\t}\n}\n"
                    self.full_source_Pool_cc.append(source_Flatten_cc)
                    self.full_source_Pool_hh.append(source_Flatten_hh)
                    self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(out_shape) + "];\n"
                    self.full_source_CNN_cc.append(
                        ["\tGlobalAveragePool2D_" + str(self.index_GlbAvgPool) + "(", self.config["layers"][i]['config']['name'], "", ""])
                    self.index_GlbAvgPool += 1

            if layer.find("GlobalMaxPooling1D") >= 0:
                if len(self.model.layers[i].input.shape) == 3:
                    found = 1
                    in_shape = (self.model.layers[i].input.shape[height_index], self.model.layers[i].input.shape[width_index])
                    out_shape = (self.model.layers[i].output.shape[1])
                    source_Flatten_hh = "void GlobalMaxPool1D_" + str(self.index_GlbMaxPool) + "(" + self.type + " input_GlobalMaxPool1D[" + str(
                        in_shape[0] * in_shape[1]) + "]," + self.type + " output_GlobalMaxPool1D[" + str(
                        out_shape) + "]);\n"
                    source_Flatten_cc = "void GlobalMaxPool1D_" + str(self.index_GlbMaxPool) + "(" + self.type + " input_GlobalMaxPool1D[" + str(
                        in_shape[0] * in_shape[1]) + "]," + self.type + " output_GlobalMaxPool1D[" + str(
                        out_shape) + "]){\n\tint hs = 0;\n\tfor (int i = 0; i < " + str(
                        in_shape[1]) + "; i++){\n\t\t" + self.type + " max = -10;\n\t\tfor (int j = 0; j < " + str(
                        in_shape[0]) + "; j++){\n\t\t\tif (input_GlobalMaxPool1D[" + str(
                        in_shape[0]) + " * i + j] >= max) max = input_GlobalMaxPool1D[" + str(
                        in_shape[0]) + " * i + j];\n\t\t}\n\t\toutput_GlobalMaxPool1D[hs] = max ;\n\t\ths++;\n\t}\n}\n"
                    self.full_source_Pool_cc.append(source_Flatten_cc)
                    self.full_source_Pool_hh.append(source_Flatten_hh)
                    self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(out_shape) + "];\n"
                    self.full_source_CNN_cc.append(
                        ["\tGlobalMaxPool1D_" + str(self.index_GlbMaxPool) + "(", self.config["layers"][i]['config']['name'], "", ""])
                    self.index_GlbMaxPool+=1

            if layer.find("GlobalAveragePooling1D") >= 0:
                if len(self.model.layers[i].input.shape) == 3:
                    found = 1
                    in_shape = (self.model.layers[i].input.shape[height_index], self.model.layers[i].input.shape[width_index])
                    out_shape = (self.model.layers[i].output.shape[1])
                    source_Flatten_hh = "void GlobalAveragePool1D_" + str(self.index_GlbAvgPool) + "(" + self.type + " input_GlobalAveragePool1D[" + str(
                        in_shape[0] * in_shape[1]) + "]," + self.type + " output_GlobalAveragePool1D[" + str(
                        out_shape) + "]);\n"
                    source_Flatten_cc = "void GlobalAveragePool1D_" + str(self.index_GlbAvgPool) + "(" + self.type + " input_GlobalAveragePool1D[" + str(
                        in_shape[0] * in_shape[1]) + "]," + self.type + " output_GlobalAveragePool1D[" + str(
                        out_shape) + "]){\n\tint hs = 0;\n\tfor (int i = 0; i < " + str(
                        in_shape[1]) + "; i++){\n\t\t" + self.type + " avg = 0;\n\t\tfor (int j = 0; j < " + str(
                        in_shape[0]) + "; j++){\n\t\t\tavg += input_GlobalAveragePool1D[" + str(
                        in_shape[0]) + " * i + j] / " + str(
                        in_shape[0]) + ";\n\t\t}\n\t\toutput_GlobalAveragePool1D[hs] = avg;\n\t\ths++;\n\t}\n}\n"
                    self.full_source_Pool_cc.append(source_Flatten_cc)
                    self.full_source_Pool_hh.append(source_Flatten_hh)
                    self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(out_shape) + "];\n"
                    self.full_source_CNN_cc.append(
                        ["\tGlobalAveragePool1D_" + str(self.index_GlbAvgPool) + "(", self.config["layers"][i]['config']['name'], "", ""])
                    self.index_GlbAvgPool+=1

            # convert flatten layer into c array that act like an flatten layer
            if layer.find("Flatten") >= 0:
                # flatten for 1d
                found = 1
                if len(self.model.layers[i].input.shape) == 3:
                    in_shape = (self.model.layers[i].input.shape[width_index], self.model.layers[i].input.shape[height_index])
                    out_shape = self.model.layers[i].output.shape[1]
                    source_Flatten_cc = "void flatten" + str(self.index_Flatten) + "(" + self.type + " input_Flatten[" + str(
                        in_shape[0] * in_shape[1]) + "]," + self.type + " output_Flatten[" + str(
                        out_shape) + "]){\n\tint hs = 0;\n\tloop_for_a_flatten:\n\tfor (int i = 0; i < " + str(
                        in_shape[0]) + "; i++){\n\t\tloop_for_c_flatten:\n\t\tfor (int j = 0; j < " + str(
                        in_shape[1]) + "; j++){\n\t\t\toutput_Flatten[hs] = input_Flatten[" + str(in_shape[1]) + "*i+j];\n\t\t\ths++;\n\t\t}\n\t}\n}\n"
                    source_Flatten_hh = "void flatten" + str(self.index_Flatten) + "(" + self.type + " input_Flatten[" + str(
                        in_shape[0] * in_shape[1]) + "]," + self.type + " output_Flatten[" + str(out_shape) + "]);\n"
                    self.full_source_Pool_cc.append(source_Flatten_cc)
                    self.full_source_Pool_hh.append(source_Flatten_hh)
                    self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(out_shape) + "];\n"
                    self.full_source_CNN_cc.append(["\tflatten" + str(self.index_Flatten) + "(", self.config["layers"][i]['config']['name'], "", ""])
                # Flatten for 3d
                if len(self.model.layers[i].input.shape) == 4:
                    in_shape = (
                    self.model.layers[i].input.shape[depth_index], self.model.layers[i].input.shape[height_index],
                    self.model.layers[i].input.shape[width_index])
                    out_shape = self.model.layers[i].output.shape[1]
                    source_Flatten_cc = "void flatten" + str(self.index_Flatten) + "(" + self.type + " input_Flatten[" + str(
                        in_shape[0] * in_shape[1] * in_shape[2]) + "]," + self.type + " output_Flatten[" + str(
                        out_shape) + "]){\n\tint hs = 0;\n\tfor (int i = 0; i < " + str(
                        in_shape[1]) + "; i++){\n\t\tfor (int j = 0; j < " + str(
                        in_shape[2]) + "; j++){\n\t\t\tfor (int k = 0; k < " + str(
                        in_shape[0]) + "; k++){\n\t\t\t\toutput_Flatten[hs] = input_Flatten[" + str(
                        in_shape[1]) + " * i + " + str(in_shape[2]) + " * " + str(
                        in_shape[1]) + " * k + j ];\n\t\t\t\ths++;\n\t\t\t}\n\t\t}\n\t}\n}\n"
                    source_Flatten_hh = "void flatten" + str(self.index_Flatten) + "(" + self.type + " input_Flatten[" + str(
                        in_shape[0] * in_shape[1] * in_shape[2]) + "]," + self.type + " output_Flatten[" + str(
                        out_shape) + "]);\n"
                    self.full_source_Pool_cc.append(source_Flatten_cc)
                    self.full_source_Pool_hh.append(source_Flatten_hh)
                    self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                        'name'] + "[" + str(out_shape) + "];\n"
                    self.full_source_CNN_cc.append(["\tflatten" + str(self.index_Flatten) + "(", self.config["layers"][i]['config']['name'], "", ""])

                self.index_Flatten+=1

            # convert dense layer into c array that act like an dense layer
            if layer.find("Dense") >= 0:
                found = 1
                weight_shape = self.model.layers[i].get_weights()[0].shape
                h = self.model.layers[i].get_weights()[0].reshape(weight_shape[0] * weight_shape[1])
                for k in h:
                    self.Weights.append(k)
                for k in self.model.layers[i].get_weights()[1]:
                    self.Weights.append(k)
                in_shape = self.model.layers[i].input.shape[1]

                out_shape = self.model.layers[i].output.shape[1]
                activation = self.config["layers"][i]['config']['activation']
                if self.type == "fxp" and self.index_D == 0:
                    self.fxp_inc = self.fxp_include
                else:
                    self.fxp_inc = ""

                # if (self.num_of_output > 1):
                #     if activation == "softmax":
                #         if self.choose_only_output:
                #             self.out[0] += " &OutModel" + str(self.index_output)
                #             if self.index_output != self.num_of_output - 1:
                #                 self.out[0] += "," + self.type
                #             self.out[1] = "1"
                #             # end = "\toutput_Dense = maxindex;\n"
                #         else:
                #             assert self.choose_only_output == False, "Py2C haven't supported the case when num_of_output > 1 and choose_only_output is False yet!!!"
                #             # self.out[0] = " OutModel" + str(self.index_output) + "[" + str(out_shape) + "]"
                #             # self.out[1] = str(out_shape)
                #
                #
                #         self.full_source_CNN_cc.append(
                #             ["\tDense_" + str(self.index_D) + "(", "OutModel" + str(self.index_output),
                #              "&Weights[" + str(self.cnt_param + in_shape * out_shape) + "]",
                #              "&Weights[" + str(self.cnt_param) + "]"])
                #         self.index_output += 1
                #     else:
                #
                #         self.full_source_CNN_cc.append(
                #             ["\tDense_" + str(self.index_D) + "(", self.config["layers"][i]['config']['name'],
                #              "&Weights[" + str(self.cnt_param + in_shape * out_shape) + "]",
                #              "&Weights[" + str(self.cnt_param) + "]"])
                #         out_dense = self.type + " output_Dense[" + str(out_shape) + "]"
                #         self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                #             'name'] + "[" + str(
                #             out_shape) + "];\n"
                # else:
                #     if self.choose_only_output:
                #         self.out[0] += " &OutModel" + str(self.index_output)
                #         # if self.index_output != self.num_of_output - 1:
                #         #     self.out[0] += "," + self.type
                #         self.out[1] = "1"
                #     else:
                #         self.out[0] = " OutModel" + "[" + str(out_shape) + "]"
                #         self.out[1] = str(out_shape)
                #     if i == len(self.config["layers"]) - 1:
                #         self.full_source_CNN_cc.append(["\tDense_" + str(self.index_D) + "(", "OutModel",
                #                                         "&Weights[" + str(self.cnt_param + in_shape * out_shape) + "]",
                #                                         "&Weights[" + str(self.cnt_param) + "]"])
                #     else:
                #         self.full_source_CNN_cc.append(
                #             ["\tDense_" + str(self.index_D) + "(", self.config["layers"][i]['config']['name'],
                #              "&Weights[" + str(self.cnt_param + in_shape * out_shape) + "]",
                #              "&Weights[" + str(self.cnt_param) + "]"])
                #         out_dense = self.type + " output_Dense[" + str(out_shape) + "]"
                #         self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                #             'name'] + "[" + str(
                #             out_shape) + "];\n"

                if (self.num_of_output > 1):
                    if activation == "softmax":

                        self.full_source_CNN_cc.append(
                            ["\tDense_" + str(self.index_D) + "(", "OutModel" + str(self.index_output),
                             "&Weights[" + str(self.cnt_param + in_shape * out_shape) + "]",
                             "&Weights[" + str(self.cnt_param) + "]"])
                        self.index_output += 1
                    else:

                        self.full_source_CNN_cc.append(
                            ["\tDense_" + str(self.index_D) + "(", self.config["layers"][i]['config']['name'],
                             "&Weights[" + str(self.cnt_param + in_shape * out_shape) + "]",
                             "&Weights[" + str(self.cnt_param) + "]"])
                        out_dense = self.type + " output_Dense[" + str(out_shape) + "]"
                        self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                            'name'] + "[" + str(
                            out_shape) + "];\n"
                else:
                    if i == len(self.config["layers"]) - 1:
                        self.full_source_CNN_cc.append(["\tDense_" + str(self.index_D) + "(", "OutModel" + str(self.index_output),
                                                        "&Weights[" + str(self.cnt_param + in_shape * out_shape) + "]",
                                                        "&Weights[" + str(self.cnt_param) + "]"])
                    else:
                        self.full_source_CNN_cc.append(
                            ["\tDense_" + str(self.index_D) + "(", self.config["layers"][i]['config']['name'],
                             "&Weights[" + str(self.cnt_param + in_shape * out_shape) + "]",
                             "&Weights[" + str(self.cnt_param) + "]"])
                        self.call_function += "\t" + self.type + " " + self.config["layers"][i]['config'][
                            'name'] + "[" + str(
                            out_shape) + "];\n"


                if activation == "sigmoid":
                    out_dense = self.type + " output_Dense[" + str(out_shape) + "]"
                    if self.type == "fxp":
                        include = "#include <hls_math.h>\n"
                        type_sqrt = "hls::exp"
                    else:
                        include = "#include <cmath>\n"
                        type_sqrt = "exp"
                    result_acc = "\tfor (int i = 0; i < " + str(out_shape) + "; i++){\n\t\toutput_Dense[i]=1/(1 + " + type_sqrt + "(-out_Dense[i]));\n\t}\n"

                elif activation == "relu":
                    out_dense = self.type + " output_Dense[" + str(out_shape) + "]"
                    result_acc = "\tfor (int i = 0; i < " + str(out_shape) + "; i++){\n\t\tif (out_Dense[i] < 0) output_Dense[i] = 0; else output_Dense[i] = out_Dense[i];\n\t}\n"
                    include = ""
                elif activation == "softmax":
                    if self.type == "fxp":
                        include = "#include <hls_math.h>\n"
                        type_sqrt = "hls::exp"
                        out_dense = self.type + " &output_Dense_"+ str(self.index_output)
                    else:
                        include = "#include <cmath>\n"
                        type_sqrt = "exp"
                        out_dense = self.type + " &output_Dense" + str(self.index_output)
                    result_acc = "\tint maxindex = 0;\n\t" + self.type + " max=out_Dense[0];\n\tloop_detect:\n\tfor (int i=0; i<" + str(out_shape) + "; i++){\n\t\tif (out_Dense[i]> max) {\n\t\t\tmax=out_Dense[i];\n\t\t\tmaxindex=i;\n\t\t}\n\t}\n\t" + self.type + " sum_exp_x = 0.0;\n\tfor(int i = 0; i <" + str(out_shape) + ";i++){\n\t\tsum_exp_x += " + type_sqrt + "(out_Dense[i]- out_Dense[maxindex]);\n\t}\n\t" + self.type + " max_value = out_Dense[maxindex];\n\tfor(int i = 0; i <" + str(out_shape) + ";i++){\n\t\tout_Dense[i] = " + type_sqrt + "(out_Dense[i] - max_value) / sum_exp_x;\n\t}\n\t" + self.type + " maxindex_2 = 0;\n\t" + self.type + " max_2 = out_Dense[0];\n\tfor(int i = 0; i <" + str(out_shape) + ";i++){\n\t\tif (out_Dense[i] > max_2) {\n\t\t\tmax_2 = out_Dense[i];\n\t\t\tmaxindex_2 = i;\n\t\t}\n\t}\n\toutput_Dense" + str(self.index_output) + " = maxindex_2;\n"

                else:
                    out_dense = self.type + " output_Dense[" + str(out_shape) + "]"
                    include = ""
                    result_acc = "\tfor (int i = 0; i < " + str(out_shape) + "; i++){\n\t\toutput_Dense[i] = out_Dense[i];\n\t}\n"


                source_Dense_cc = self.fxp_inc + include + "void Dense_" + str(
                    self.index_D) + "(" + self.type + " input_Dense[" + str(
                    in_shape) + "]," + out_dense + "," + self.type + " bias[" + str(
                    out_shape) + "]," + self.type + " weight[" + str(
                    in_shape * out_shape) + "]){\n\t" + self.type + " out_Dense[" + str(
                    out_shape) + "];\n" + "\tloop_for_a_Dense_" + str(self.index_D) + ":\n\tfor (int i = 0; i < " + str(
                    out_shape) + "; i++){\n\t\t" + self.type + " s=0;\n\t\tloop_for_b_Dense_" + str(
                    self.index_D) + ":\n\t\tfor (int j = 0; j < " + str(
                    in_shape) + "; j++){\n\t\t\ts+=input_Dense[j]*weight[j*" + str(
                    out_shape) + "+i];\n\t\t}\n\t\t" + "out_Dense[i]=s+bias[i];" + "\n\t}\n" + result_acc + "}\n"
                source_Dense_hh = self.fxp_inc + "void Dense_" + str(
                    self.index_D) + "(" + self.type + " input_Dense[" + str(
                    in_shape) + "]," + out_dense + "," + self.type + " bias[" + str(
                    out_shape) + "]," + self.type + " weight[" + str(in_shape * out_shape) + "]);\n"
                self.full_source_Dense_cc.append(source_Dense_cc)
                self.full_source_Dense_hh.append(source_Dense_hh)
                self.index_D += 1
                self.cnt_param += in_shape * out_shape + out_shape

            if layer.find("Input") >= 0:
                found = 1


            assert layer.find("conv2d_input") < 0, "Py2C is now only supporting Keras Functional API"

            assert layer.find("conv1d_input") < 0, "Py2C is now only supporting Keras Functional API"

            assert found == 1, "Py2C has not supporting " + str(self.config["layers"][i]['class_name']) + " yet"

        ######################################### SET UP INPUT PHARSE ##################################################
        def find_input(Layer_name, index):
            if "Pad" in Layer_name:
                subname = self.full_source_CNN_cc[index + 1][1]
            else:
                subname = Layer_name
            if 'OutModel' in Layer_name:
                if self.choose_only_output == True:
                    for i in range(len(self.config["layers"])):
                        if 'Dense' in self.config["layers"][i]['class_name']:
                            if self.config["layers"][i]['config']['activation'] == 'softmax' and self.count_dense_output == 0:
                                input_name = self.config["layers"][i]['inbound_nodes'][0][0][0]
                                self.count_dense_output += 1
                                return input_name
                            if self.config["layers"][i]['config']['activation'] == 'softmax' and self.count_dense_output == 1:
                                input_name = self.config["layers"][i+1]['inbound_nodes'][0][0][0]
                                self.count_dense_output += 1
                                return input_name
                            if self.config["layers"][i]['config']['activation'] == 'softmax' and self.count_dense_output == 2:
                                input_name = self.config["layers"][i+2]['inbound_nodes'][0][0][0]
                                self.count_dense_output += 1
                                return input_name
                else:
                    for i in range(len(self.config["layers"])):
                        if 'Dense' in self.config["layers"][i]['class_name']:
                            input_name = self.config["layers"][i]['inbound_nodes'][0][0][0]
                            return input_name

            for i in range(len(self.config["layers"])):
                if self.config["layers"][i]['config']['name'] == subname:
                    if ("conv" in subname and "Pad" in self.full_source_CNN_cc[index - 1][1]) or (
                            "pool" in subname and "Pad" in self.full_source_CNN_cc[index - 1][1]):
                        input_name = self.full_source_CNN_cc[index - 1][1]
                    else:
                        input_name = self.config["layers"][i]['inbound_nodes'][0][0][0]
            if 'input' in input_name:
                input_name = self.full_source_CNN_cc[0][1]
            return input_name

        def find_input_for_add(Layer_name, index):
            argu_add = ""
            for i in range(len(self.config["layers"])):
                if self.config["layers"][i]['config']['name'] == Layer_name:
                    for k in range(0, len(self.config["layers"][i]['inbound_nodes'][0])):
                        argu_add += self.config["layers"][i]['inbound_nodes'][0][k][0]
                        argu_add += ", "
            return argu_add

        def find_input_for_concatenate(Layer_name, index):
            argu = ""
            for i in range(len(self.config["layers"])):
                if self.config["layers"][i]['config']['name'] == Layer_name:
                    for k in range(0, len(self.config["layers"][i]['inbound_nodes'][0])):
                        argu += self.config["layers"][i]['inbound_nodes'][0][k][0]
                        argu += ", "
            return argu

        for i in range(len(self.full_source_CNN_cc)):
            if i == 0:
                continue
            else:
                if (len(self.full_source_CNN_cc[i]) > 2):
                    if self.full_source_CNN_cc[i][2] == "":
                        self.call_function += self.full_source_CNN_cc[i][0] + find_input(self.full_source_CNN_cc[i][1],
                                                                                         i) + "," + \
                                              self.full_source_CNN_cc[i][1] + ");\n"
                    else:
                        if (len(self.full_source_CNN_cc[i]) == 6):
                            self.call_function += self.full_source_CNN_cc[i][0] + find_input(
                                self.full_source_CNN_cc[i][1], i) + "," + self.full_source_CNN_cc[i][1] + "," + \
                                                  self.full_source_CNN_cc[i][2] + "," + self.full_source_CNN_cc[i][
                                                      3] + "," + self.full_source_CNN_cc[i][4] + "," + \
                                                  self.full_source_CNN_cc[i][5] + ");\n"
                        if (len(self.full_source_CNN_cc[i]) == 5):
                            self.call_function += self.full_source_CNN_cc[i][0] + find_input(
                                self.full_source_CNN_cc[i][1], i) + "," + self.full_source_CNN_cc[i][1] + "," + \
                                                  self.full_source_CNN_cc[i][2] + "," + self.full_source_CNN_cc[i][
                                                      3] + "," + self.full_source_CNN_cc[i][4] + ");\n"
                        if (len(self.full_source_CNN_cc[i]) == 4):
                            # test = find_input(self.full_source_CNN_cc[i][1], i)
                            self.call_function += self.full_source_CNN_cc[i][0] + find_input(self.full_source_CNN_cc[i][1], i) + "," + self.full_source_CNN_cc[i][1] + "," + self.full_source_CNN_cc[i][2] + "," + self.full_source_CNN_cc[i][3] + ");\n"

                        if (len(self.full_source_CNN_cc[i]) == 3):
                            self.call_function += self.full_source_CNN_cc[i][0] + find_input(self.full_source_CNN_cc[i][1], i) + "," + self.full_source_CNN_cc[i][1] + "," + self.full_source_CNN_cc[i][2] + ");\n"

                else:
                    if "add" in self.full_source_CNN_cc[i][1]:
                        self.call_function += self.full_source_CNN_cc[i][0] + find_input_for_add(
                            self.full_source_CNN_cc[i][1], i) + self.full_source_CNN_cc[i][1] + ");\n"
                    elif ("concatenate" in self.full_source_CNN_cc[i][0]) or ("Concatenate" in self.full_source_CNN_cc[i][0]):
                        self.call_function += self.full_source_CNN_cc[i][0] + find_input_for_concatenate(self.full_source_CNN_cc[i][1], i) + self.full_source_CNN_cc[i][1] + ");\n"
                    else:
                        self.call_function += self.full_source_CNN_cc[i][0] + find_input(self.full_source_CNN_cc[i][1],
                                                                                         i) + "," + \
                                              self.full_source_CNN_cc[i][1] + ");\n"



        if (self.num_of_output > 1):
            if activation == "softmax":
                if self.choose_only_output:
                    for self.index_output in range(0,self.num_of_output):
                        self.out[0] += " &OutModel" + str(self.index_output)
                        if self.index_output != self.num_of_output - 1:
                            self.out[0] += "," + self.type

                    self.out[1] = "1"
                else:
                    assert self.choose_only_output == False, "Py2C haven't supported the case when num_of_output > 1 and choose_only_output is False yet!!!"
        else:
            if self.choose_only_output:
                self.out[0] += " &OutModel" + str(self.index_output)
                self.out[1] = "1"
            else:
                self.out[0] = " OutModel" + str(self.index_output) + "[" + str(out_shape) + "]"
                self.out[1] = str(out_shape)

        if len(self.model.layers[1].input.shape) == 4:
            self.source_CNN += "void CNN(" + self.type + " InModel[" + str(self.model.layers[1].input.shape[depth_index] * self.model.layers[1].input.shape[height_index] *
                self.model.layers[1].input.shape[width_index]) + "]," + self.type + self.out[0] + "," + self.type + " Weights[" + str(
                self.cnt_param) + "]){\n" + self.call_function + "}\n"
            self.source_CNN_hh = "void CNN(" + self.type + " InModel[" + str(
                self.model.layers[1].input.shape[2] *
                self.model.layers[1].input.shape[1] * self.model.layers[1].input.shape[
                    3]) + "]," + self.type + self.out[0] + "," + self.type + " Weights[" + str(
                self.cnt_param) + "]);\n"
        if len(self.model.layers[1].input.shape) == 3:
            self.source_CNN += "void CNN(" + self.type + " InModel[" + str(
                self.model.layers[1].input.shape[2] * self.model.layers[1].input.shape[1]) + "]," + self.type + \
                               self.out[0] + "," + self.type + " Weights[" + str(
                self.cnt_param) + "]){\n" + self.call_function + "}\n"
            self.source_CNN_hh = "void CNN(" + self.type + " InModel[" + str(
                self.model.layers[1].input.shape[2] *
                self.model.layers[1].input.shape[1]) + "]," + self.type + self.out[
                                     0] + "," + self.type + " Weights[" + str(
                self.cnt_param) + "]);\n"
        if len(self.model.layers[1].input.shape) == 2:
            self.source_CNN += "void CNN(" + self.type + " InModel[" + str(
                self.model.layers[1].input.shape[1]) + "]," + self.type + self.out[
                                   0] + "," + self.type + " Weights[" + str(
                self.cnt_param) + "]){\n" + self.call_function + "}\n"
            self.source_CNN_hh = "void CNN(" + self.type + " InModel[" + str(
                self.model.layers[1].input.shape[1]) + "]," + self.type + self.out[
                                     0] + "," + self.type + " Weights[" + str(self.cnt_param) + "]);\n"

        ######################################### WRITING TESTBENCH PHARSE #############################################
        if self.choose_only_output == False:
            add_because_of_onlyoutput = [" * " + self.out[1],
                                         "for (int j = 0; j < " + self.out[1] + "; j++){\n\t\t\t*(OutArray + " +
                                         self.out[1] + " * i + j) = OutModel0[j];\n\t\t}"]
        else:
            add_because_of_onlyoutput = ["", "*(OutArray + i) = OutModel0;"]

        if self.ide == "vs":
            ignore_warning = "#define _CRT_SECURE_NO_WARNINGS\n"
        else:
            ignore_warning = ""


        if self.choose_only_output == False:
            self.call[0] = "CNN(Image, OutModel0, Weights);"
            self.call[1] = self.out[0] + ";"
        else:
            if self.num_of_output == 1:
                self.call[0] = "CNN(Image, OutModel0, Weights);"
                self.call[1] = "OutModel0;"
            elif self.num_of_output == 2:
                self.call[0] = "CNN(Image, OutModel0, OutModel1, Weights);"
                self.call[1] = "OutModel0, OutModel1;"
            elif self.num_of_output == 3:
                self.call[0] = "CNN(Image, OutModel0, OutModel1, OutModel2, Weights);"
                self.call[1] = "OutModel0, OutModel1, OutModel2;"
            else:
                self.call[0] = "CNN(Image, OutModel0, Weights);"
                self.call[1] = "OutModel0;"

        if (len(self.model.layers[1].input.shape) == 4):
            self.source_CNN_tb = ignore_warning + "#include <conio.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <math.h>\n#include <string>\n#include <fstream>\n#include <iostream>\n#include \"CNN.h\"\n#include \"Conv.h\"\n#include \"Pool.h\"\n#include \"Dense.h\"\n#define NumberOfPicture " + "...\n#define d " + "...\n" + self.fxp_inc + "int main(){\n\t" + self.type + " " + self.call[1] + "\n\t" + self.type + "* Weights = (" + self.type + "*)malloc(" + str(
                self.cnt_param) + " * sizeof(" + self.type + "));\n\tfloat tmp;\n\tFILE* Weight = fopen(\"Float_Weights.txt\", \"r\");\n\tfor (int i = 0; i < " + str(
                self.cnt_param) + "; i++){\n\t\tfscanf(Weight, \"%f\", &tmp);\n\t\t*(Weights + i)=tmp;\n\t}\n\tfclose(Weight);" + "\n\t////read Input" + "\n\t" + self.type + "* InModel = (" + self.type + "*)malloc((NumberOfPicture * d * " + str(
                self.model.layers[1].input.shape[height_index]) + " * " + str(self.model.layers[1].input.shape[
                                                                                  width_index]) + ") * sizeof(" + self.type + "));\n\tFILE* Input = fopen(\"X.txt\", \"r\");\n\tfor (int i = 0; i < " + "NumberOfPicture * d * " + str(
                self.model.layers[1].input.shape[height_index]) + " * " + str(self.model.layers[1].input.shape[
                                                                                  width_index]) + "; i++){\n\t\tfscanf(Input, \"%f\", &tmp);\n\t\t*(InModel + i)=tmp;\n\t}\n\tfclose(Input);" + "\n\t//Read Label" + "\n\t" + self.type + "*Label = (" + self.type + "*)malloc((NumberOfPicture" + \
                                 add_because_of_onlyoutput[
                                     0] + ") * sizeof(" + self.type + "));" + "\n\tFILE* Output = fopen(\"Y.txt\", \"r\");\n\tfor (int i = 0; i < NumberOfPicture" + \
                                 add_because_of_onlyoutput[
                                     0] + " ; i++)\n\t{\n\t\tfscanf(Output, \"%f\", &tmp);\n\t\t*(Label + i) = tmp;\n\t}\n\tfclose(Output);\n\t" + self.type + "*OutArray = (" + self.type + "*)malloc((NumberOfPicture" + \
                                 add_because_of_onlyoutput[
                                     0] + ") * sizeof(" + self.type + "));\n\t" + self.type + " Image[d * " + str(
                self.model.layers[1].input.shape[height_index]) + " * " + str(self.model.layers[1].input.shape[
                                                                                  width_index]) + "] = {};\n\tfor (int i = 0; i < NumberOfPicture ; i++)\n\t{\n\t\tint startIndex = i * d * " + str(
                self.model.layers[1].input.shape[height_index]) + " * " + str(
                self.model.layers[1].input.shape[width_index]) + ";\n\t\tfor (int k = 0; k < d * " + str(
                self.model.layers[1].input.shape[height_index]) + " * " + str(self.model.layers[1].input.shape[
                                                                                  width_index]) + "; k++)\n\t\t{\n\t\t\tImage[k] = *(InModel + startIndex + k);\n\t\t}\n\t\t" + self.call[0] + "\n\t\t*(OutArray + i) = OutModel0;\n\t}\n\tfloat countTrue = 0;\n\tfor (int i = 0; i < NumberOfPicture" + \
                                 add_because_of_onlyoutput[
                                     0] + "; i++)\n\t{\n\t\tint labelValue = *(Label + i);\n\t\tint PredictValue = *(OutArray + i);\n\t\tif (labelValue == PredictValue)\n\t\t{\n\t\t\tcountTrue = countTrue + 1;\n\t\t}\n\t}\n\tfloat accuracy = (float)((countTrue / (NumberOfPicture" + \
                                 add_because_of_onlyoutput[
                                     0] + ")) * 100);\n\tstd::cout << \"accuracy of Model: \" << accuracy << \"%\\n\";" + "\n\t//std::cout << \"Result: \" <<  OutModel <<  \"\\n\";" + "\n\treturn 0;\n}\n"
        if (len(self.model.layers[1].input.shape) == 3):
            self.source_CNN_tb = ignore_warning + "#include <conio.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <math.h>\n#include <string>\n#include <fstream>\n#include <iostream>\n#include \"CNN.h\"\n#include \"Conv.h\"\n#include \"Pool.h\"\n#include \"Dense.h\"\n#define NumberOfPicture " + "...\n#define d " + "...\n" + self.fxp_inc + "int main(){\n\t" + self.type + " " + self.call[1] + "\n\t" + self.type + "* Weights = (" + self.type + "*)malloc(" + str(
                self.cnt_param) + " * sizeof(" + self.type + "));\n\tfloat tmp;\n\tFILE* Weight = fopen(\"Float_Weights.txt\", \"r\");\n\tfor (int i = 0; i < " + str(
                self.cnt_param) + "; i++){\n\t\tfscanf(Weight, \"%f\", &tmp);\n\t\t*(Weights + i)=tmp;\n\t}\n\tfclose(Weight);" + "\n\t////read Input" + "\n\t" + self.type + "* InModel = (" + self.type + "*)malloc((NumberOfPicture * d * " + str(
                self.model.layers[1].input.shape[1]) + " * " + str(self.model.layers[1].input.shape[
                                                                       2]) + ") * sizeof(" + self.type + "));\n\tFILE* Input = fopen(\"X.txt\", \"r\");\n\tfor (int i = 0; i < " + "NumberOfPicture * d * " + str(
                self.model.layers[1].input.shape[1]) + " * " + str(self.model.layers[1].input.shape[
                                                                       2]) + "; i++){\n\t\tfscanf(Input, \"%f\", &tmp);\n\t\t*(InModel + i)=tmp;\n\t}\n\tfclose(Input);" + "\n\t//Read Label" + "\n\t" + self.type + "*Label = (" + self.type + "*)malloc((NumberOfPicture" + \
                                 add_because_of_onlyoutput[
                                     0] + ") * sizeof(" + self.type + "));" + "\n\tFILE* Output = fopen(\"Y.txt\", \"r\");\n\tfor (int i = 0; i < NumberOfPicture" + \
                                 add_because_of_onlyoutput[
                                     0] + " ; i++)\n\t{\n\t\tfscanf(Output, \"%f\", &tmp);\n\t\t*(Label + i) = tmp;\n\t}\n\tfclose(Output);\n\t" + self.type + "*OutArray = (" + self.type + "*)malloc((NumberOfPicture" + \
                                 add_because_of_onlyoutput[
                                     0] + ") * sizeof(" + self.type + "));\n\t" + self.type + " Image[d * " + str(
                self.model.layers[1].input.shape[1]) + " * " + str(self.model.layers[1].input.shape[
                                                                       2]) + "] = {};\n\tfor (int i = 0; i < NumberOfPicture ; i++)\n\t{\n\t\tint startIndex = i * d * " + str(
                self.model.layers[1].input.shape[1]) + " * " + str(
                self.model.layers[1].input.shape[2]) + ";\n\t\tfor (int k = 0; k < d * " + str(
                self.model.layers[1].input.shape[1]) + " * " + str(self.model.layers[1].input.shape[
                                                                       2]) + "; k++)\n\t\t{\n\t\t\tImage[k] = *(InModel + startIndex + k);\n\t\t}\n\t\t" + self.call[0] + "\n\t\t" + \
                                 add_because_of_onlyoutput[
                                     1] + "\n\t}\n\tfloat countTrue = 0;\n\tfor (int i = 0; i < NumberOfPicture" + \
                                 add_because_of_onlyoutput[
                                     0] + "; i++)\n\t{\n\t\tint labelValue = *(Label + i);\n\t\tint PredictValue = *(OutArray + i);\n\t\tif (labelValue == PredictValue)\n\t\t{\n\t\t\tcountTrue = countTrue + 1;\n\t\t}\n\t}\n\tfloat accuracy = (float)((countTrue / (NumberOfPicture" + \
                                 add_because_of_onlyoutput[
                                     0] + ")) * 100);\n\tstd::cout << \"accuracy of Model: \" << accuracy << \"%\\n\";" + "\n\t//std::cout << \"Result: \" <<  OutModel <<  \"\\n\";" + "\n\treturn 0;\n}\n"
        if (len(self.model.layers[1].input.shape) < 3):
            self.source_CNN_tb = ignore_warning + "#include <conio.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <math.h>\n#include <string>\n#include <fstream>\n#include <iostream>\n#include \"CNN.h\"\n#include \"Conv.h\"\n#include \"Pool.h\"\n#include \"Dense.h\"\n#define NumberOfPicture " + "...\n#define d " + "...\n" + self.fxp_inc + "int main(){\n\t" + self.type + " " + self.call[1] + "\n\t" + self.type + "* Weights = (" + self.type + "*)malloc(" + str(
                self.cnt_param) + " * sizeof(" + self.type + "));\n\tfloat tmp;\n\tFILE* Weight = fopen(\"Float_Weights.txt\", \"r\");\n\tfor (int i = 0; i < " + str(
                self.cnt_param) + "; i++){\n\t\tfscanf(Weight, \"%f\", &tmp);\n\t\t*(Weights + i)=tmp;\n\t}\n\tfclose(Weight);" + "\n\t////read Input" + "\n\t" + self.type + "* InModel = (" + self.type + "*)malloc((NumberOfPicture * d * " + str(
                self.model.layers[1].input.shape[
                    1]) + ") * sizeof(" + self.type + "));\n\tFILE* Input = fopen(\"X.txt\", \"r\");\n\tfor (int i = 0; i < " + "NumberOfPicture * d * " + str(
                self.model.layers[1].input.shape[
                    1]) + "; i++){\n\t\tfscanf(Input, \"%f\", &tmp);\n\t\t*(InModel + i)=tmp;\n\t}\n\tfclose(Input);" + "\n\t//Read Label" + "\n\t" + self.type + "*Label = (" + self.type + "*)malloc((NumberOfPicture) * sizeof(" + self.type + "));" + "\n\tFILE* Output = fopen(\"Y.txt\", \"r\");\n\tfor (int i = 0; i < NumberOfPicture ; i++)\n\t{\n\t\tfscanf(Output, \"%f\", &tmp);\n\t\t*(Label + i) = tmp;\n\t}\n\tfclose(Output);\n\t" + self.type + " OutArray[NumberOfPicture] = {};\n\t" + self.type + " Image[d * " + str(
                self.model.layers[1].input.shape[
                    1]) + "] = {};\n\tfor (int i = 0; i < NumberOfPicture ; i++)\n\t{\n\t\tint startIndex = i * d * " + str(
                self.model.layers[1].input.shape[1]) + ";\n\t\tfor (int k = 0; k < d * " + str(
                self.model.layers[1].input.shape[
                    1]) + "; k++)\n\t\t{\n\t\t\tImage[k] = *(InModel + startIndex + k);\n\t\t}\n\t\t" + self.call[0] + "\n\t\tOutArray[i] = OutModel;\n\t}\n\tfloat countTrue = 0;\n\tfor (int i = 0; i < NumberOfPicture; i++)\n\t{\n\t\tint labelValue = *(Label + i);\n\t\tif (labelValue == OutArray[i])\n\t\t{\n\t\t\tcountTrue = countTrue + 1;\n\t\t}\n\t}\n\tfloat accuracy = (float)((countTrue / NumberOfPicture) * 100);\n\tstd::cout << \"accuracy of Model: \" << accuracy << \"%\\n\";" + "\n\t//std::cout << \"Result: \" <<  OutModel <<  \"\\n\";" + "\n\treturn 0;\n}\n"
        if self.type == "fxp":
            self.fxp_inc = self.fxp_include
        else:
            self.fxp_inc = ""
        print("Successful Converting")

    def WriteCfile(self):
        path = []
        cnt = 0
        if len(self.full_source_Conv_cc):
            path.append(self.path_w[0])
            cnt += 1
            with open(self.path_w[0], mode='w') as f:
                for i in range(len(self.full_source_Conv_cc)):
                    for j in range(len(self.full_source_Conv_cc[i].split("\n")) - 1):
                        f.write(self.full_source_Conv_cc[i].split("\n")[j] + "\n")
        if len(self.full_source_Conv_hh):
            path.append(self.path_w[1])
            with open(self.path_w[1], mode='w') as f:
                for i in range(len(self.full_source_Conv_hh)):
                    test_j = len(self.full_source_Conv_hh[i].split("\n")) - 1
                    for j in range(len(self.full_source_Conv_hh[i].split("\n")) - 1):
                        test = self.full_source_Conv_hh[i].split("\n")[j] + "\n"
                        f.write(self.full_source_Conv_hh[i].split("\n")[j] + "\n")
        if len(self.full_source_Pool_cc):
            cnt += 1
            path.append(self.path_w[2])
            with open(self.path_w[2], mode='w') as f:
                for i in range(len(self.full_source_Pool_cc)):
                    for j in range(len(self.full_source_Pool_cc[i].split("\n")) - 1):
                        f.write(self.full_source_Pool_cc[i].split("\n")[j] + "\n")
        if len(self.full_source_Pool_hh):
            path.append(self.path_w[3])
            with open(self.path_w[3], mode='w') as f:
                for i in range(len(self.full_source_Pool_hh)):
                    for j in range(len(self.full_source_Pool_hh[i].split("\n")) - 1):
                        f.write(self.full_source_Pool_hh[i].split("\n")[j] + "\n")
        if len(self.full_source_Dense_cc):
            cnt += 1
            path.append(self.path_w[4])
            with open(self.path_w[4], mode='w') as f:
                for i in range(len(self.full_source_Dense_cc)):
                    for j in range(len(self.full_source_Dense_cc[i].split("\n")) - 1):
                        f.write(self.full_source_Dense_cc[i].split("\n")[j] + "\n")
        if len(self.full_source_Dense_hh):
            path.append(self.path_w[5])
            with open(self.path_w[5], mode='w') as f:
                for i in range(len(self.full_source_Dense_hh)):
                    for j in range(len(self.full_source_Dense_hh[i].split("\n")) - 1):
                        f.write(self.full_source_Dense_hh[i].split("\n")[j] + "\n")
        if len((self.CNN_include + self.fxp_inc + self.source_CNN).split("\n")) - 1 and cnt:
            path.append(self.path_w[6])
            with open(self.path_w[6], mode='w') as f:
                for j in range(len((self.CNN_include + self.fxp_inc + self.source_CNN).split("\n")) - 1):
                    f.write((self.CNN_include + self.fxp_inc + self.source_CNN).split("\n")[j] + "\n")
        if len((self.fxp_inc + self.source_CNN_hh).split("\n")) - 1 and cnt:
            path.append(self.path_w[7])
            with open(self.path_w[7], mode='w') as f:
                for j in range(len((self.fxp_inc + self.source_CNN_hh).split("\n")) - 1):
                    f.write((self.fxp_inc + self.source_CNN_hh).split("\n")[j] + "\n")
            self.Weights = np.array(self.Weights)
        if len((self.source_CNN_tb).split("\n")) - 1 and cnt:
            path.append(self.path_w[8])
            with open(self.path_w[8], mode='w') as f:
                for j in range(len((self.source_CNN_tb).split("\n")) - 1):
                    f.write((self.source_CNN_tb).split("\n")[j] + "\n")

        if len(path):
            print("Successful Writing file")
            print("There are ", str(len(path)), " file(s) such as:")
            for name in path:
                print("\t", name)
        else:
            print("Py2C do not support your model!!!")

    def del_one_file(self, name):
        if os.path.exists(name):
            try:
                # Delete the file
                os.remove(name)
                print(f"The file {name} has been deleted.")
            except Exception as e:
                print(f"Unable to delete the file {name}: {e}")
        else:
            print(f"The file {name} does not exist.")

    def del_any_file(self, name_arr):
        for name in name_arr:
            self.del_one_file(name)

    def del_all_file(self):
        for name in self.path_w:
            self.del_one_file(name)


    def Write_Float_Weights_File(self, path="Float_Weights.txt"):
        assert len(self.Weights) != 0, "Converting has not implemented yet!!! Please Run convert2C in Py2C"
        with open(path, mode='w') as f:
            for i in self.Weights:
                f.write(str(i) + " ")
        print("Successful Writing Float Weights file!!!")

    def Write_IEEE754_32bits_Weights_File(self, path="IEEE754_32bits_Weights.txt"):
        def float_to_binary32(f):
            # Chuyển đổi số float thành binary32
            packed = struct.pack('!f', f)

            # Chuyển đổi binary32 thành chuỗi nhị phân
            binary = ''.join(format(byte, '08b') for byte in packed)

            return binary

        assert len(self.Weights) != 0, "Converting has not implemented yet!!! Please Run convert2C in Py2C"
        with open(path, mode='w') as f:
            for i in range(len(self.Weights)):
                # print(i)
                binary_num = float_to_binary32(self.Weights[i])
                decimal_num = int(binary_num, 2)
                f.write(str(decimal_num) + " ")
        print("Successful Writing IEEE754 32bits Weights file!!!")

    def Write_FixedPoint_Weights_File(self, path="FixedPoint_Weights.txt"):
        def float_to_binary32(f):
            packed = struct.pack('!f', f)

            binary = ''.join(format(byte, '08b') for byte in packed)

            return binary

        def binary32_to_fixedpoint(f):
            f32 = int(float_to_binary32(f), 2)
            Nbitbot = self.fxp_para[0] - 2
            Nbittop = self.fxp_para[1] - 2
            Limitbot = 2 ** (23 - Nbitbot)
            M = ((f32 // Limitbot) & (2 ** Nbitbot - 1)) + 2 ** Nbitbot  # and add Nbittop bit(s) 0 state before

            E = int((f32 & 2139095040) / (2 ** 23))
            S = f32 & 2147483648
            Es = E - 127
            F = int(M * (2 ** Es))
            Nbitbotaf = Nbittop + Nbitbot + 1 - self.fxp_para[0] + 1

            G = F // (2 ** Nbitbotaf) & (2 ** (Nbittop + Nbitbot + 1) - 1)
            if S == 1:
                G = 2 ** self.fxp_para[0] - G
            return G

        assert len(self.Weights) != 0, "Converting has not implemented yet!!! Please Run convert2C in Py2C"
        assert self.fxp_para is not None, "Fxp parameter has not set yet!! Please run set_Fxp_Param in Py2C"
        with open(path, mode='w') as f:
            for i in range(len(self.Weights)):
                # test = float_to_binary32(self.Weights[i])
                binary_num = binary32_to_fixedpoint(self.Weights[i])
                f.write(str(binary_num) + " ")
        print("Successful Writing Fixed Point Weights file!!!")