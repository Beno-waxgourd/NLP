import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertAttention
from config import nyt_star_rel2id, webnlg_star_rel2id
# from pprint import pprint

class Span_selection_model_en(nn.Module):
    def __init__(self, window_size, rel_num, tag_num=2, mapping='start'):
        super(Span_selection_model_en, self).__init__()
        self.encoder = BertModel.from_pretrained('../bert-base-cased')
        hidden_size = self.encoder.config.hidden_size
        self.window_size = window_size
        self.mapping = mapping
        self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        self.attention = BertAttention(self.encoder.config)
        self.classify_fc = nn.Linear(hidden_size, tag_num)
        self.rel_embedding = nn.Embedding(num_embeddings=rel_num, embedding_dim=hidden_size)
        self.selection_u = nn.Linear(hidden_size, hidden_size)
        self.selection_v = nn.Linear(hidden_size, hidden_size)
        self.selection_uv = nn.Linear(2 * hidden_size, hidden_size)
        self.parameters().__init__()

    def forward(self, text_ids, span_mask):
        mask = text_ids > 0
        bert_out = self.encoder(text_ids, mask)
        seq_hiddens, pool_hiddens = bert_out['last_hidden_state'], bert_out['pooler_output']
        pool_hiddens = pool_hiddens.unsqueeze(1)
        seq_len = seq_hiddens.size()[-2]
        span_hiddens_list = []

        assert self.mapping in ['start', 'end', 'length'], 'mapping error!'
        if self.mapping == 'start':
            for idx in range(seq_len):
                end_idx = min(idx + self.window_size, seq_len)
                for idxx in range(idx + 1, end_idx + 1):
                    span_hiddens = torch.max(seq_hiddens[:, idx:idxx, :], dim=1, keepdim=True)[0]
                    span_hiddens = torch.cat((span_hiddens, pool_hiddens), dim=-1)
                    span_hiddens_list.append(span_hiddens)
        elif self.mapping == 'end':
            for idx in range(1,seq_len+1):
                start_idx = max(idx - self.window_size, 0)
                for idxx in range(start_idx, idx):
                    span_hiddens = torch.max(seq_hiddens[:, idxx:idx, :], dim=1, keepdim=True)[0]
                    span_hiddens = torch.cat((span_hiddens, pool_hiddens), dim=-1)
                    span_hiddens_list.append(span_hiddens)
        elif self.mapping == 'length':
            for win in range(self.window_size):
                for idx in range(seq_len-win):
                    span_hiddens = torch.max(seq_hiddens[:, idx:idx+win+1, :], dim=1, keepdim=True)[0]
                    span_hiddens = torch.cat((span_hiddens, pool_hiddens), dim=-1)
                    span_hiddens_list.append(span_hiddens)
        spans_hiddens = torch.cat(span_hiddens_list, dim=1)
        spans_hiddens = torch.tanh(self.combine_fc(spans_hiddens))

        self.lstm.flatten_parameters()
        spans_hiddens = self.lstm(spans_hiddens)[0]
        attention_mask = span_mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * -10000.0

        spans_hiddens = self.attention(hidden_states=spans_hiddens, attention_mask=attention_mask)[0]

        classify_logics = self.classify_fc(spans_hiddens)
        B, L, H = spans_hiddens.size()
        u = torch.tanh(self.selection_u(spans_hiddens)).unsqueeze(1).expand(B, L, L, -1)
        v = torch.tanh(self.selection_v(spans_hiddens)).unsqueeze(2).expand(B, L, L, -1)
        uv = torch.tanh(self.selection_uv(torch.cat((u, v), dim=-1)))
        selection_logics = torch.einsum('bijh,rh->birj', uv,
                                        self.rel_embedding.weight)

        return selection_logics, classify_logics     #多头选择 +片段分类

class Span_tagger_en:
    def __init__(self, max_len, window_size, div_path='nyt', mapping='start'):
        self.matrix_size = max_len
        self.window_size = window_size
        self.mapping = mapping

        assert div_path in ['nyt', 'webnlg'], 'div_path error'
        if div_path == 'nyt':
            self.rel2id = nyt_star_rel2id
        elif div_path == 'webnlg':
            self.rel2id = webnlg_star_rel2id

        self.entity2id = {
            "DEFAULT": 0, "N": 1
        }

        self.rel_num = len(self.rel2id)
        self.ent_num = len(self.entity2id)

        assert self.mapping in ['start', 'end', 'length'], 'mapping error!'
        if self.mapping == 'start':
            self.index = [(idx, end_idx) for idx in range(self.matrix_size)
                          for end_idx in range(idx, min(idx + self.window_size, self.matrix_size))]
        elif self.mapping == 'end':
            self.index = [(start_idx, idx) for idx in range(self.matrix_size)
                          for start_idx in range(max(idx - self.window_size + 1, 0), idx + 1)]
        elif self.mapping == 'length':
            self.index = [(idx, idx + wins) for wins in range(self.window_size)
                          for idx in range(self.matrix_size - wins)]

        self.matrix_idx2span_idx = {(matrix_idx[0], matrix_idx[1]): span_idx for span_idx, matrix_idx in
                                    enumerate(self.index)}

        self.span_idx2matrix_idx = {v: k for k, v in self.matrix_idx2span_idx.items()}

    def get_span_selection_info(self, sample, len_tokens):
        length = len(self.index)
        selection_id = np.zeros((length, self.rel_num, length))
        mask_len = min(len_tokens, self.matrix_size - 2) #!!!中文可以用len(sample['text'])英文用len_token
        selection_mask = np.zeros(length)
        classify_id = np.zeros((length, self.ent_num))

        assert self.mapping in ['start', 'end', 'length'], 'mapping error!'
        index = []
        if self.mapping == 'start':
            index = [(idx, end_idx) for idx in range(1, mask_len+1)
                     for end_idx in range(idx, min(idx + self.window_size, mask_len+1))]
        elif self.mapping == 'end':
            index = [(start_idx, idx) for idx in range(1, mask_len+1)
                     for start_idx in range(max(idx - self.window_size + 1, 0), idx + 1)]
        elif self.mapping == 'length':
            index = [(idx, idx + wins) for wins in range(self.window_size)
                     for idx in range(mask_len - wins+1)]

        for idx in index:
            selection_mask[self.matrix_idx2span_idx[idx]] = 1

        selection_id[:, self.rel2id['N'], :] = 1
        rel_list = sample['relation_list']
        for rel in rel_list:
            # print(rel)
            relation_pos = self.rel2id[rel['predicate']]
            subject_pos = (rel['subj_tok_span'][0] + 1, rel['subj_tok_span'][1])
            object_pos = (rel['obj_tok_span'][0] + 1, rel['obj_tok_span'][1])

            if subject_pos[1] >= self.matrix_size or object_pos[1] >= self.matrix_size:
                continue
            if subject_pos[1] + 1 - subject_pos[0] > self.window_size or \
                    object_pos[1] + 1 - object_pos[0] > self.window_size:
                continue
            subject_index = self.matrix_idx2span_idx[subject_pos]
            object_index = self.matrix_idx2span_idx[object_pos]
            selection_id[subject_index, relation_pos, object_index] = 1
            selection_id[subject_index, self.rel2id['N'], object_index] = 0

        classify_id[:, self.entity2id['N']] = 1
        ent_list = sample['entity_list']
        for ent in ent_list:
            ent_type_pos = self.entity2id[ent['type']]
            ent_pos = (ent['tok_span'][0] + 1, ent['tok_span'][1])
            if ent_pos[1] >= self.matrix_size:
                continue
            if ent_pos[1] + 1 - ent_pos[0] > self.window_size:
                continue
            ent_index = self.matrix_idx2span_idx[ent_pos]
            classify_id[ent_index, ent_type_pos] = 1
            classify_id[ent_index, self.entity2id['N']] = 0

        return torch.tensor(selection_id, dtype=torch.long), torch.tensor(selection_mask, dtype=torch.long), \
               torch.tensor(classify_id, dtype=torch.long)

