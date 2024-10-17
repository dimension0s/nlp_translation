# 1.处理数据集
# 1.1）加载数据集
import numpy as np
import os,json
from torch.utils.data import Dataset,DataLoader,random_split
train_set_size = 210000
valid_set_size = 20000

class TRANS(Dataset):
    def __init__(self,data_file):
        self.data = self.load_data(data_file)

    def load_data(self,data_file):
        Data = {}
        with open(data_file,'rt',encoding='utf-8') as f:
            for idx,line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


data = TRANS("E:\\NLPProject\\机器翻译\data\\trainslation2019zh_cutted.json")
train_data,valid_data = random_split(data,[train_set_size,valid_set_size])
test_data = TRANS("E:\\NLPProject\\机器翻译\\data\\translation2019zh_valid.json")

# 输出数据集大小并打印一个训练样本
print(f'train set size:{len(train_data)}')
print(f'valid set size:{len(valid_data)}')
print(f'test set size:{len(test_data)}')
print(next(iter(train_data)))