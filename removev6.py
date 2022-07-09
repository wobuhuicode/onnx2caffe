from hashlib import new
from operator import mod
import sys
import os
import onnx

# 修改 yolov5 网络，去除transpose及后处理部分，修改输出形状


# 需要给定的参数
model_name = 'model/yolov6n.onnx'
class_num = 1

node_remove_start = ['Transpose_108', 'Transpose_149', 'Transpose_190']
node_remove_end = ['Conv_133', 'Conv_174']
reshape_param_list = ['onnx::Reshape_443', 'onnx::Reshape_453',  'onnx::Reshape_463']
reshape_node = ['Reshape_107', 'Reshape_148', 'Reshape_189']

output_name = 'model/yolov6n_removed.onnx'

# model_name = 'new_weights/mbv3l5.onnx'
# class_num = 1

# node_remove_start = ['Transpose_312', 'Transpose_330', 'Transpose_348']
# node_remove_end = ['Conv_328', 'Conv_346']
# reshape_param_list = ['onnx::Reshape_1131', 'onnx::Reshape_1142',  'onnx::Reshape_1153']
# reshape_node = ['Reshape_311', 'Reshape_329', 'Reshape_347']

# output_name = 'new_weights/mbv3l5_removed.onnx'

# 自动化
model = onnx.load(model_name)
graph = model.graph
nodes = graph.node



# for node in nodes:
#     if node.name == 'Div_29':
#         print(node.input_tensors)


# for value_info in graph.initializer:
#     if value_info.name == 'onnx::Div_397':
#         print(value_info)

# print("ini")

# for value_info in graph.value_info:
#     if value_info.name == 'onnx::Div_397':
#         print(value_info)

# input()

print(graph.input)
print(graph.output)

len_object = class_num + 5

outputs = []

# 添加新的输出节点
output1 = onnx.helper.make_tensor_value_info('output1', onnx.TensorProto.FLOAT, [1, 3, len_object, 6400])
output2 = onnx.helper.make_tensor_value_info('output2', onnx.TensorProto.FLOAT, [1, 3, len_object, 1600])
output3 = onnx.helper.make_tensor_value_info('output3', onnx.TensorProto.FLOAT, [1, 3, len_object, 400])

outputs.extend([output1, output2, output3])


graph.output.extend(outputs)
graph.output.remove(graph.output[0])

print(graph.output)


# 删除没用的节点
remove_list = []
flag = False

for i in range(len(nodes)):
    if nodes[i].name in node_remove_start :
        flag = True
    if nodes[i].name in node_remove_end :
        flag = False
    if flag :
        remove_list.append(nodes[i])

for i in range(len(remove_list)):
    graph.node.remove(remove_list[i])

# 需要修改的 reshape param list

remove_list = []
for i in range(len(graph.initializer)):
    if graph.initializer[i].name in reshape_param_list:
        remove_list.append(graph.initializer[i])

for i in range(len(remove_list)):
    graph.initializer.remove(remove_list[i])


# 创建3个新的 reshape tensor
new_reshape1 = onnx.helper.make_tensor(reshape_param_list[0], onnx.TensorProto.INT32, [4], [1, 1, len_object, 6400])
new_reshape2 = onnx.helper.make_tensor(reshape_param_list[1], onnx.TensorProto.INT32, [4], [1, 1, len_object, 1600])
new_reshape3 = onnx.helper.make_tensor(reshape_param_list[2], onnx.TensorProto.INT32, [4], [1, 1, len_object, 400])
graph.initializer.extend([new_reshape1, new_reshape2, new_reshape3])

# 修改 reshape node

# reshape 的输出接到output
for i in range(len(graph.node)):
    if graph.node[i].name == reshape_node[0]:
        graph.node[i].output.remove(graph.node[i].output[0])
        graph.node[i].output.append('output1')
    if graph.node[i].name == reshape_node[1]:
        graph.node[i].output.remove(graph.node[i].output[0])
        graph.node[i].output.append('output2')
    if graph.node[i].name == reshape_node[2]:
        graph.node[i].output.remove(graph.node[i].output[0])
        graph.node[i].output.append('output3')


onnx.checker.check_model(model)
onnx.save_model(model, output_name)

    


