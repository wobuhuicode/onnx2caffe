
file_name = 'new_weights/mbv3l5.prototxt'
new_file_context = ''

with open(file=file_name) as file:
    for line in file:
        # 包含 name
        if 'name' in line:
            line = line[:-2]
            line = line + '_cpu"\n'
        new_file_context += line

# 写入新文件
with open(file=file_name + 'cpu', mode='x') as file:
    file.write(new_file_context)