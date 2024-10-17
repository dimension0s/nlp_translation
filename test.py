# 2.2) 验证/测试函数
from sacrebleu.metrics import BLEU # 专用于翻译任务的验证指标
import numpy,torch
from device import device
from collate_fn import tokenizer
import numpy as np

bleu = BLEU()

def test_loop(dataloader,model,mode='Valid'):
    assert mode in ['Valid','Test']
    preds,labels = [],[]

    model.eval()
    for batch_data in dataloader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            # 1.生成预测
            generated_tokens = model.generate(
                batch_data['input_ids'],
                attention_mask=batch_data['attention_mask'],
                max_length=128
            ).cpu().numpy()

        # 对预测解码(重点)：tokenizer的用法：.batch_decode
        decoded_tokens = tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)

        # 2.导出标签并解码
        label_tokens = batch_data['labels'].cpu().numpy()
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens,skip_special_tokens=True)

        preds += [pred.strip() for pred in decoded_tokens]
        labels += [[label.strip()] for label in decoded_labels]

    bleu_score = bleu.corpus_score(preds,labels).score
    print(f"BLEU:{bleu_score:>0.2f}\n")
    return bleu_score 





