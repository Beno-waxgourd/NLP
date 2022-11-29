import torch
import sys
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def find_entity(pos, text, len_tokens, offset):
    start_index = pos[0]  #
    end_index = pos[1]
    start_offset = offset[start_index][0]
    end_offset = offset[end_index][1]
    if start_index == 0:
        return '[cls]' + text[start_offset:end_offset]
    elif start_index == len_tokens-1:
        return '[sep]' + '[pad]'*(end_index-start_index)
    elif start_index > len_tokens-1:
        return '[pad]' * (end_index - start_index + 1)
    elif end_index >= len_tokens-1:
        return text[start_offset:offset[len_tokens-2][1]]+'[sep]' + '[pad]' * (end_index - len_tokens + 1)
    return text[start_offset:end_offset]

def classify_decode(mask, texts, len_tokens, classify_logits, id2entity, span_idx2matrix_idx, offset):
    classify_mask = mask.unsqueeze(2).expand(-1, -1, classify_logits.size(2))
    classify_tags = (torch.sigmoid(classify_logits) * classify_mask.float()) > 0.5
    idx = torch.nonzero(input=classify_tags.to(device), as_tuple=False)
    batch_num = len(texts)
    result = [[] for _ in range(batch_num)]
    for i in range(idx.size()[0]):
        b, e, t = idx[i].tolist()  # batch, entity, type
        predicate = id2entity[t]
        # print(b,e,t)
        if predicate == 'N':
            continue

        entity_span = span_idx2matrix_idx[e]
        entity = find_entity(pos=entity_span, text=texts[b], len_tokens=len_tokens[b], offset=offset[b])

        ent = {
            'text': entity,
            'type': predicate
        }
        result[b].append(ent)
    return result

def selection_decode(mask, texts, len_tokens, selection_logits, id2rel, span_idx2matrix_idx, offset):
    selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2).expand(-1, -1, len(id2rel), -1)
    selection_tags = (torch.sigmoid(selection_logits) * selection_mask.float()) > 0.5
    idx = torch.nonzero(input=selection_tags.to(device), as_tuple=False)
    batch_num = len(texts)
    result = [[] for _ in range(batch_num)]
    # print('tokens:',tokens)
    for i in range(idx.size()[0]):

        b, s, p, o = idx[i].tolist()  # batch subj predict obj
        predicate = id2rel[p]
        # print('b,s,p,o:',b,s,p,o)
        if predicate == 'N':
            continue

        subj_span = span_idx2matrix_idx[s]
        obj_span = span_idx2matrix_idx[o]

        subj = find_entity(pos=subj_span, text=texts[b], len_tokens=len_tokens[b], offset=offset[b])
        obj = find_entity(pos=obj_span, text=texts[b],  len_tokens=len_tokens[b], offset=offset[b])

        rel = {
            'subject': subj,
            'object': obj,
            'predicate': predicate,
        }
        result[b].append(rel)
    return result
