# 此处是将提取出的数据单独存为文件
# 原训练集500万条，为了快速训练，提取前23万条，其中2万条作为验证集，valid作为测试集
# 接下来提取数据：
input_filename = "机器翻译/data/translation2019zh_train.json"
output_filename = "机器翻译/data/trainslation2019zh_cutted.json"

lines_to_extract = 230000
with open(input_filename,'rt',encoding='utf-8') as in_file,\
    open(output_filename,'rt',encoding='utf-8') as out_file:
    line_count = 0
    for line in in_file:
        if line_count >= lines_to_extract:
            break
        out_file.write(line)
        line_count += 1
print(f'Extracted {line_count} lines and saved to {output_filename}')