import torch.nn.functional as F


def selection_masked_loss(selection_mask, selection_logits, selection_gold):
    selection_masked = (selection_mask.unsqueeze(2) * selection_mask.unsqueeze(1)).unsqueeze(2).expand(
        -1, -1, selection_logits.size(2), -1)  # batch x seq x rel x seq
    selection_loss = F.binary_cross_entropy_with_logits(selection_logits.float(),
                                                        selection_gold.float(),
                                                        reduction='none')
    selection_loss = selection_loss.masked_select(selection_masked).sum()
    selection_loss /= selection_mask.sum()
    return selection_loss


def classify_masked_loss(mask, classify_logits, classify_gold):
    classify_masked = mask.unsqueeze(2).expand(-1, -1, classify_logits.size(2))  # batch  x seq x type

    classify_loss = F.binary_cross_entropy_with_logits(classify_logits.float(),
                                                       classify_gold.float(),
                                                       reduction='none')

    classify_loss = classify_loss.masked_select(classify_masked).sum()
    classify_loss /= mask.sum()
    return classify_loss

def loss_print(avg_loss, avg_classify_loss, avg_selection_loss, step):
    print(f'avg_classify_loss:{avg_classify_loss / step:.6f}')
    print(f'avg_selection_loss:{avg_selection_loss / step:.6f}')
    print(f'avg_loss:{avg_loss / step:.6f}')