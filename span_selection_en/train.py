import sys
sys.path.append('.')
sys.path.append('..')
from tqdm import tqdm
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch.utils.data.dataloader import DataLoader
from span_selection_model_en import Span_tagger_en, Span_selection_model_en
from ss_dataset import Ss_dataset
from loss import selection_masked_loss, classify_masked_loss, loss_print
from decoder import classify_decode, selection_decode
from calc import calc_scores, get_re_cpg, get_ent_cpg, score_print


###
epochs = 100
batch_size = 2
lr = 0.00001
max_len = 128
max_win = 6
mode = 'webnlg'
selection_loss_weight = 0.75
classify_loss_weight = 0.25

div_path = 'nyt'
rel_num = 25

# div_path = 'webnlg'
# rel_num = 172

mapping = 'start'
tag_num = 2
###

rel_extractor = Span_selection_model_en(rel_num=rel_num, window_size=max_win, tag_num=tag_num)
#rel_extractor = nn.DataParallel(rel_extractor)
rel_extractor.to(device)
#print(rel_extractor.combine_fc.weight.size())

span_tagger = Span_tagger_en(max_len=max_len, window_size=max_win, div_path=div_path, mapping=mapping)
id2rel = {v: k for k, v in span_tagger.rel2id.items()}
id2entity = {v: k for k, v in span_tagger.entity2id.items()}
dataset = Ss_dataset(div_path=div_path, mode=mode, span_tagger=span_tagger, max_len=max_len)
train_data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=dataset.train_fn)
optimizer = torch.optim.AdamW(rel_extractor.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_data_loader) * 2, 1)

# num_params = 0
# for param in rel_extractor.parameters():
#     print(param.,param.numel())
#     num_params += param.numel()
# print(num_params )
# a = len([param for  param in rel_extractor.parameters()])
# b = len([name for name in rel_extractor.state_dict()])
# print (a,b)
# for name,param in rel_extractor.state_dict().items():
#     num_params+=param.numel()
#     print(name,param.numel(),sep=' ')
# print(num_params )

assert div_path in ['nyt', 'webnlg'], 'div_path error'
assert mapping in ['start', 'end', 'length'], 'mapping error'
if os.path.exists(f'span_selection_en_{div_path}_{mapping}.pth'):
    print('参数文件存在！')
    rel_extractor.load_state_dict(torch.load(f'span_selection_en_{div_path}_{mapping}.pth'))

for epoch in range(epochs):
    avg_loss = 0
    avg_classify_loss = 0
    avg_selection_loss = 0

    cpg = {'rel_cpg': [0, 0, 0],
           'ent_cpg': [0, 0, 0],}

    for idx, data in tqdm(enumerate(train_data_loader)):
        sample_list = data['samples']
        text_ids = data['text_ids'].to(device)
        len_tokens = []
        for token in text_ids:
            x = 0
            for i in token:
                if i > 0:
                    x += 1
            len_tokens.append(x)
        selection_ids = data['selection_ids'].to(device)
        selection_masks = data['selection_masks'].to(device)
        classify_ids = data['classify_ids'].to(device)
        offsets = data['offsets']

        selection_logits, classify_logics = rel_extractor.forward(text_ids=text_ids, span_mask=selection_masks)   #在此进入span_selection_model_en.py
        selection_loss = selection_masked_loss(selection_mask=selection_masks.bool(), selection_logits=selection_logits, selection_gold=selection_ids)
        classify_loss = classify_masked_loss(mask=selection_masks.bool(), classify_logits=classify_logics.float(), classify_gold=classify_ids.float())
        loss = selection_loss_weight * selection_loss + classify_loss_weight * classify_loss

        avg_loss += loss.item()
        avg_classify_loss += classify_loss.item()
        avg_selection_loss += selection_loss.item()
        loss.backward()
        # 梯度积累

        rel = selection_decode(mask=selection_masks, texts=[sample['text'] for sample in sample_list],
                               len_tokens=len_tokens,
                               selection_logits=selection_logits, id2rel=id2rel,
                               span_idx2matrix_idx=span_tagger.span_idx2matrix_idx, offset=offsets)
        ent = classify_decode(mask=selection_masks, texts=[sample['text'] for sample in sample_list],
                              len_tokens=len_tokens,
                              classify_logits=classify_logics, id2entity=id2entity,
                              span_idx2matrix_idx=span_tagger.span_idx2matrix_idx, offset=offsets)

        rel_cpg = get_re_cpg([sample['relation_list'] for sample in sample_list], rel)
        ent_cpg = get_ent_cpg([sample['entity_list'] for sample in sample_list], ent)
        for i in range(3):
            cpg['rel_cpg'][i] += rel_cpg[i]
            cpg['ent_cpg'][i] += ent_cpg[i]

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if idx % 100 == 0 and idx > 0:
            print('epoch:',epoch)
            score_print(cpg)
            print('\n\n')
        if idx % 200 == 0 and idx > 0:
            assert div_path in ['nyt', 'webnlg'], 'div_path error'
            assert mapping in ['start', 'end', 'length'], 'mapping error'
            torch.save(rel_extractor.state_dict(), f'span_selection_en_{div_path}_{mapping}.pth')
            print("模型已保存")
    torch.save(rel_extractor.state_dict(), f'span_selection_en_{div_path}_{mapping}.pth')
    print('epoch:', epoch)
    score_print(cpg)
    print("模型已保存")


