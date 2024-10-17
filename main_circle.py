# 3.主训练循环
import random,os
import torch
from transformers import AdamW,get_scheduler
import numpy as np
from collate_fn import model,train_dataloader,valid_dataloader,test_dataloader
from train import train_loop
from test import test_loop

def seed_everything(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    os.environ['PYTHONHASHSEED']=str(seed)
seed_everything(42)

learning_rate = 1e-5
epoch_num=5

optimizer = AdamW(model.parameters(),lr=learning_rate)
lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num*len(train_dataloader))

best_bleu = 0.
total_loss=0.

for epoch in range(epoch_num):
    print(f'Epoch {epoch+1}/{epoch_num}\n---------------------------')
    total_loss = train_loop(train_dataloader,model,optimizer,lr_scheduler,epoch+1)
    valid_bleu = test_loop(valid_dataloader,model,'Valid')
    if valid_bleu>best_bleu:
        best_bleu=valid_bleu
        print('saving new weights...\n')
        torch.save(model.state_dict(),
                   f'epoch_{epoch+1}_valid_bleu_{valid_bleu:0.2f}_model_weights.bin')
        # 打印验证集评价指标
        print(f'bleu_score:{valid_bleu}')