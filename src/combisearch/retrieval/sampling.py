import random


def random_by_turn(embs, ratio=0.1):
    n_selected = int(ratio * len(embs))
    print(f"randomly select {ratio} of turns, i.e. {n_selected} turns")
    selected_keys = random.sample(list(embs), n_selected)
    return {k: v for k, v in embs.items() if k in selected_keys}


def random_by_dialog(embs, ratio=0.1):
    dial_ids = {turn_label.split('_')[0] for turn_label in embs}
    n_selected = int(len(dial_ids) * ratio)
    print(f"randomly select {ratio} of dialogs, i.e. {n_selected} dialogs")
    selected_dial_ids = random.sample(list(dial_ids), n_selected)
    return {k: v for k, v in embs.items() if k.split('_')[0] in selected_dial_ids}


def pre_assigned(embs, examples, *, empty_if_none: bool = False):
    embs = {} if empty_if_none and embs is None else embs
    selected_dial_ids = {dial['ID'] for dial in examples}
    return {k: v for k, v in embs.items() if k.split('_')[0] in selected_dial_ids}


def sgd_pre_assigned(embs, examples):
    embs = {} if embs is None else embs
    selected_dial_ids = {dial['ID'] for dial in examples}
    return {k: v for k, v in embs.items() if '_'.join(k.split('_')[:2]) in selected_dial_ids}
