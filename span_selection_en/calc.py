def get_re_cpg(true_rel, pred_rel):
    rel_cpg = [0, 0, 0]
    res_set = set(
        '_'.join((j['object'], j['predicate'], j['subject'])) for i in pred_rel
        if i for j in i
    )
    sample_set = set(
        '_'.join((j['object'], j['predicate'], j['subject'])) for i in true_rel
        if i for j in i)

    rel_cpg[0] += len(res_set & sample_set)
    rel_cpg[1] += len(res_set)
    rel_cpg[2] += len(sample_set)
    return rel_cpg

def get_ent_cpg(true_ent, pred_ent):
    ent_cpg = [0, 0, 0]
    res_set = set(
        '_'.join((j['text'], j['type'])) for i in pred_ent
        if i for j in i
    )
    sample_set = set(
        '_'.join((j['text'], j['type'])) for i in true_ent
        if i for j in i)

    ent_cpg[0] += len(res_set & sample_set)
    ent_cpg[1] += len(res_set)
    ent_cpg[2] += len(sample_set)
    return ent_cpg

def calc_scores(c, p, g):
    mini = 1e-12
    precision = c / (p + mini)
    recall = c / (g + mini)
    f1 = 2 * precision * recall / (precision + recall + mini)
    return precision, recall, f1

def score_print(cpg) :
    ent_cpg = cpg['ent_cpg']
    rel_cpg = cpg['rel_cpg']

    ent_score = calc_scores(ent_cpg[0], ent_cpg[1], ent_cpg[2])
    rel_score = calc_scores(rel_cpg[0], rel_cpg[1], rel_cpg[2])
    print(cpg)
    print(f'ent:  {ent_score}')
    print(f'rel:  {rel_score}')
