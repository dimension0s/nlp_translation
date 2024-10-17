# 1.2) 处理批次函数，分词，编码
# 注意：由于任务性质，需要对原文和译文都要做分词编码，以及对pad字符做特殊处理：-100，为了计算交叉熵损失
import torch
from torch.utils.data import DataLoader
from transformers import MarianTokenizer,MarianMTModel
from data import train_data,valid_data,test_data

max_input_length = 128
max_target_length = 128

model_path = "Helsinki-NLP/opus-mt-zh-en"  # 中译英
tokenizer = MarianTokenizer.from_pretrained(model_path)  # ,local_files_only=True
model = MarianMTModel.from_pretrained(model_path)  # ,local_files_only=True

def collote_fn(batch_samples):
    batch_inputs,batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['chinese'])
        batch_targets.append(sample['english'])
    batch_data = tokenizer(
        batch_inputs,
        padding=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets,
            padding=True,
            max_length=max_target_length,
            truncation=True,
            return_tensors='pt'
        )['input_ids']

        # 重点步骤：
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)

        # eos_token_id:end of sequence:序列的结束
        # torch.where(labels==tokenizer.eos_token_id)：返回的是（行索引，列索引）
        # [1]:取列索引
        end_token_index = torch.where(labels==tokenizer.eos_token_id)[1]
        for idx,end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data

train_dataloader = DataLoader(train_data,batch_size=32,shuffle=True,collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data,batch_size=32,shuffle=False,collate_fn=collote_fn)
test_dataloader=DataLoader(test_data,batch_size=32,shuffle=False,collate_fn=collote_fn)

# 尝试打印一个batch的数据，以验证是否处理正确：
batch = next(iter(train_dataloader))
print(batch.keys())
print('batch shape:',{k:v.shape for k,v in batch.items()})
print(batch)