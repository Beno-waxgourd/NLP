import json
import torch
from tqdm import tqdm
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from torch.utils.data.dataset import Dataset, T_co
from pprint import pprint
from span_selection_model_en import Span_tagger_en
class Ss_dataset(Dataset):
    pre_model_path = '../bert-base-cased'

    def __init__(self, span_tagger, mode='train', div_path = 'nyt', max_len=64):
        self.text_ids = []
        self.samples = []
        self.offsets = []
        self.tokenizer = BertTokenizerFast.from_pretrained(self.pre_model_path)
        self.tagger = span_tagger
        self.max_len = max_len

        path = '../data4bert/nyt_star/train_data_new.json'

        assert div_path in ['nyt', 'webnlg'], 'div_path error'
        assert mode in ['train', 'test'], 'mode error'
        path = f'../data4bert/{div_path}_star/{mode}_data_new.json'

        with open(path, 'r', encoding='utf-8-sig') as f:
            x = json.load(f)
            for line in tqdm(x):
                sample = line
                text = sample['text']
                info = self.tokenizer.encode_plus(text=text, padding='max_length', truncation=True,
                                                  max_length=self.max_len, return_offsets_mapping=True)
                offset = info['offset_mapping']
                text_id = info['input_ids']
                self.samples.append(sample)
                self.text_ids.append(torch.tensor(text_id, dtype=torch.long))
                self.offsets.append(offset)

    def __getitem__(self, index) -> T_co:
        return self.samples[index], self.text_ids[index], self.offsets[index],

    def __len__(self):
        return len(self.samples)

    def train_fn(self, data):
        selection_ids = []
        selection_masks = []
        classify_ids = []
        for sample in [i[0] for i in data]:
            len_tokens = len(self.tokenizer.tokenize(sample['text']))
            selection_id, selection_mask, classify_id = self.tagger.get_span_selection_info(sample=sample, len_tokens=len_tokens)
            selection_ids.append(selection_id)
            selection_masks.append(selection_mask)
            classify_ids.append(classify_id)

        return {
            'samples': [i[0] for i in data],
            'text_ids': torch.stack([i[1] for i in data], dim=0),
            'selection_ids': torch.stack(selection_ids, dim=0),
            'selection_masks': torch.stack(selection_masks, dim=0),
            'classify_ids': torch.stack(classify_ids, dim=0),
            'offsets': [i[2] for i in data]
        }
