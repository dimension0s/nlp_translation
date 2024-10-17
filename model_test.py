# 4.模型测试
from collate_fn import test_dataloader,model,max_target_length,max_input_length,tokenizer
import json,torch
from sacrebleu.metrics import BLEU
import numpy as np
from device import device
from tqdm.auto import tqdm

model.load_state_dict(torch.load('epoch_10_valid_accuracy_0.9028_weights.pth'))
model.eval()

# 加一项别的source,为了匹配该项，其余的指标也重新写了一遍，稍微啰嗦一点
with torch.no_grad():
    print('evaluating on test set......')
    sources, preds, labels = [], [], []
    for batch_data in tqdm(test_dataloader):
        batch_data = batch_data.to(device)
        # 1.生成预测
        generated_tokens = model.generate(
            batch_data['input_ids'],
            attention_mask=batch_data['attention_mask'],
            max_length=max_target_length, ).cpu().numpy()
        # 对预测解码
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # 2.对答案解码
        label_tokens = batch_data['labels'].cpu().numpy()
        label_tokens = np.where(labels != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        # 3.对原文本也做解码
        decoded_sources = tokenizer.batch_deocde(
            batch_data['input_ids'].cpu().numpy(),
            skip_special_tokens=True,
            use_source_tokenizer=True, )
        sources += [source.strip() for source in decoded_sources]

        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]

    bleu = BLEU()
    bleu_score = bleu.corpus_score(preds, labels).score
    print(f'Test BLEU:{bleu_score:>0.2f}\n')
    results = []
    print('saving predicted results...')
    for source, pred, label in zip(sources, preds, labels):
        results.append({
            'sentence': source,
            'prediction': pred,
            'translation': label[0],  # label是列表格式
        })
    with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
        for example_result in results:
            f.write(json.dumps(example_result, ensure_ascii=False) + '\n')
