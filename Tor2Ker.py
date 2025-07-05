import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import ast

class tor2ker:

    def __init__(self, tor_model, input_size):
        self.tor_model = torch.jit.load(tor_model)
        self.input_size = input_size

    def parse_forward_method_with_lines(self, text):
        # Parse the text into an Abstract Syntax Tree (AST)
        tree = ast.parse(text)

        # List to store (layer_name, line_number) tuples
        layer_info = []

        # Find the forward method
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'forward':
                # Iterate through statements in the forward method
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                        # Handle torch.reshape
                        if isinstance(stmt.value.func,
                                      ast.Attribute) and stmt.value.func.value.id == 'torch' and stmt.value.func.attr == 'reshape':
                            line_number = stmt.lineno
                            layer_info.append(('reshape', line_number))
                        # Handle .forward calls like x0 = (conv_1).forward(x, )
                        elif isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == 'forward':
                            # Check if the call is wrapped in a tuple or directly on a Name
                            target = stmt.value.func.value
                            if isinstance(target, ast.Name):
                                var_name = target.id
                                # Find the assignment of the variable (e.g., conv_1 = self.conv_1)
                                for prev_stmt in node.body[:node.body.index(stmt)]:
                                    if isinstance(prev_stmt, ast.Assign) and len(prev_stmt.targets) == 1 and \
                                            prev_stmt.targets[0].id == var_name:
                                        if isinstance(prev_stmt.value,
                                                      ast.Attribute) and prev_stmt.value.value.id == 'self':
                                            line_number = stmt.lineno
                                            layer_info.append((prev_stmt.value.attr, line_number))
                                            break
                            elif isinstance(target, ast.Tuple) and len(target.elts) == 1 and isinstance(target.elts[0],
                                                                                                        ast.Name):
                                var_name = target.elts[0].id
                                # Find the assignment of the variable
                                for prev_stmt in node.body[:node.body.index(stmt)]:
                                    if isinstance(prev_stmt, ast.Assign) and len(prev_stmt.targets) == 1 and \
                                            prev_stmt.targets[0].id == var_name:
                                        if isinstance(prev_stmt.value,
                                                      ast.Attribute) and prev_stmt.value.value.id == 'self':
                                            line_number = stmt.lineno
                                            layer_info.append((prev_stmt.value.attr, line_number))
                                            break

                    elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                        # Handle return (linear_2).forward(x9, )
                        if isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == 'forward':
                            target = stmt.value.func.value
                            if isinstance(target, ast.Name):
                                var_name = target.id
                                # Find the assignment of the variable (e.g., linear_2 = self.linear_2)
                                for prev_stmt in node.body:
                                    if isinstance(prev_stmt, ast.Assign) and len(prev_stmt.targets) == 1 and \
                                            prev_stmt.targets[0].id == var_name:
                                        if isinstance(prev_stmt.value,
                                                      ast.Attribute) and prev_stmt.value.value.id == 'self':
                                            line_number = stmt.lineno
                                            layer_info.append((prev_stmt.value.attr, line_number))
                                            break
                            elif isinstance(target, ast.Tuple) and len(target.elts) == 1 and isinstance(target.elts[0],
                                                                                                        ast.Name):
                                var_name = target.elts[0].id
                                # Find the assignment of the variable
                                for prev_stmt in node.body:
                                    if isinstance(prev_stmt, ast.Assign) and len(prev_stmt.targets) == 1 and \
                                            prev_stmt.targets[0].id == var_name:
                                        if isinstance(prev_stmt.value,
                                                      ast.Attribute) and prev_stmt.value.value.id == 'self':
                                            line_number = stmt.lineno
                                            layer_info.append((prev_stmt.value.attr, line_number))
                                            break

        layer_name = []
        for i, (name, line) in enumerate(layer_info, 1):
            layer_name.append(name)
        print("test")
        return layer_name

    def convert_conv(self, pmodule, kmodel, first_layer):
        print(pmodule.original_name)
        text = pmodule._c.code_with_constants[0]

        def extract_more_attributes(text):
            tree = ast.parse(text)
            attributes = {
                'padding': None,
                'kernel_size': None,
                'stride': None,
                'out_channels': None,
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            attr_name = item.target.id
                            if attr_name in attributes:
                                if isinstance(item.value, ast.Tuple):
                                    attributes[attr_name] = tuple(elt.value for elt in item.value.elts)
                                elif isinstance(item.value, ast.Constant):
                                    attributes[attr_name] = item.value.value

            return attributes

        pattribute = extract_more_attributes(text)
        pweights = pmodule.weight.data.cpu().numpy()
        pweights = np.transpose(pweights, (2, 1, 0))
        if first_layer == 0:
            klayer = tf.keras.layers.Conv1D(filters=pattribute['out_channels'], kernel_size=pattribute['kernel_size'],
                                            strides=pattribute['stride'], padding="same", use_bias=False,
                                            input_shape=(self.input_size))
            klayer.build(input_shape=(self.input_size))
        else:
            klayer = tf.keras.layers.Conv1D(filters=pattribute['out_channels'], kernel_size=pattribute['kernel_size'],
                                            strides=pattribute['stride'], padding="same", use_bias=False)
            klayer.build(kmodel.outputs[0].shape)
        klayer.set_weights([pweights])

        kmodel.add(klayer)

    def convert_bn(self, pmodule, kmodel):
        print("test")
        klayer = tf.keras.layers.BatchNormalization()
        klayer.build(kmodel.outputs[0].shape)
        weights = pmodule.weight.data.cpu().numpy()
        biases = pmodule.bias.data.cpu().numpy()
        mean = pmodule.running_mean.data.cpu().numpy()
        var = pmodule.running_var.data.cpu().numpy()
        klayer.set_weights([weights, biases, mean, var])
        kmodel.add(klayer)
        print("test")

    def convert_relu(self, pmodule, kmodel):
        klayer = tf.keras.layers.ReLU()
        kmodel.add(klayer)

    def convert_softmax(self, pmodule, kmodel):
        klayer = tf.keras.layers.Softmax()
        kmodel.add(klayer)

    def convert_flatten(self, pmodule, kmodel):
        klayer = tf.keras.layers.Flatten()
        kmodel.add(klayer)

    def convert_linear(self, pmodule, kmodel):
        text = pmodule._c.code

        def extract_linear_attributes(text):
            # Phân tích đoạn văn bản thành cây cú pháp trừu tượng (AST)
            tree = ast.parse(text)

            # Dictionary để lưu giá trị
            attributes = {
                'out_features': None
            }

            # Duyệt qua các node trong cây cú pháp
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Tìm các gán giá trị trong định nghĩa lớp
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            # Lấy tên thuộc tính
                            attr_name = item.target.id
                            if attr_name in attributes:
                                # Trích xuất giá trị
                                if isinstance(item.value, ast.Constant):
                                    attributes[attr_name] = item.value.value

            return attributes

        pattribute = extract_linear_attributes(text)
        pweights = pmodule.weight.data.cpu().numpy()
        pbiases = pmodule.bias.data.cpu().numpy()
        klayer = tf.keras.layers.Dense(units=pattribute['out_features'])
        klayer.build(kmodel.outputs[0].shape)
        klayer.set_weights([pweights.T, pbiases])
        kmodel.add(klayer)

    def convert_maxpool(self, pmodule, kmodel):
        text = pmodule._c.code

        def extract_maxpool2d_attributes(text):
            # Phân tích đoạn văn bản thành cây cú pháp trừu tượng (AST)
            tree = ast.parse(text)

            # Dictionary để lưu các giá trị
            attributes = {
                'kernel_size': None,
                'padding': None,
                'stride': None
            }

            # Duyệt qua các node trong cây cú pháp
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Tìm các gán giá trị trong định nghĩa lớp
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            # Lấy tên thuộc tính
                            attr_name = item.target.id
                            if attr_name in attributes:
                                # Trích xuất giá trị
                                if isinstance(item.value, ast.Constant):
                                    attributes[attr_name] = item.value.value

            return attributes

        pattribute = extract_maxpool2d_attributes(text)
        if pattribute["padding"] == 0:
            pattribute["padding"] = "valid"
        else:
            pattribute["padding"] = "same"

        klayer = tf.keras.layers.MaxPool1D(pool_size=pattribute["kernel_size"], padding=pattribute["padding"],
                                           strides=pattribute['stride'])
        kmodel.add(klayer)

    def convert(self):
        pytorch_layers = nn.ModuleList()
        forwar_text = self.tor_model.forward.code
        layer_names = self.parse_forward_method_with_lines(forwar_text)
        for module_name in layer_names:
            layer_none = nn.Identity()
            if module_name == "reshape":
                pytorch_layers.append(layer_none)
                continue
            for attr_name in dir(self.tor_model):
                if not attr_name.startswith('__'):
                    if (module_name == attr_name):
                        attr_value = getattr(self.tor_model, attr_name)
                        pytorch_layers.append(attr_value)
        kmodel = tf.keras.Sequential()
        first_layer = 0
        for module in pytorch_layers:
            if first_layer == 0:
                if module.original_name == "Conv1d":
                    self.convert_conv(module, kmodel, first_layer)
                    first_layer = 1
            else:
                test2 = module.named_modules().gi_frame.f_locals['self']
                test3 = str(test2)
                if test3 == "Identity()":
                    self.convert_flatten(module, kmodel)
                    continue
                if module.original_name == "Conv1d":
                    self.convert_conv(module, kmodel, first_layer)
                if module.original_name == "ReLU":
                    self.convert_relu(module, kmodel)
                if module.original_name == "Softmax":
                    self.convert_softmax(module, kmodel)
                if module.original_name == "MaxPool1d":
                    self.convert_maxpool(module, kmodel)
                if module.original_name == "Linear":
                    self.convert_linear(module, kmodel)
                if module.original_name == "BatchNorm1d":
                    self.convert_bn(module,kmodel)


        return kmodel
