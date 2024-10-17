# 该示例直接使用了 Transformers 库自带的 MarianMTModel类来构建模型，
# 并且在批处理函数中还调用了模型自带的 prepare_decoder_input_ids_from_labels 函数
# 因此接下来只需完成训练，验证和测试环节，不用再对模型架构进一步加工

# 2.模型训练
# 2.1）训练函数
from tqdm.auto import tqdm
import os
from device import device

def train_loop(dataloader,model,optimizer,lr_scheduler,epoch):
    total_loss = 0.
    model = model.to(device)
    model.train()

    progress_bar = tqdm(enumerate(dataloader),total=len(dataloader))
    for step,batch_data in progress_bar:
        batch_data = batch_data.to(device)
        outputs = model(**batch_data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        progress_bar.set_description(f'Epoch:{epoch},Loss:{avg_loss:.4f}')
    return total_loss