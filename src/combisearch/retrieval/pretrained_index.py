"""Pretrained SBERT index helpers."""
import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from combisearch.runtime.io import read_json_from_data_dir
from combisearch.representation.turn_text import data_item_to_string, DOMAINS


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed_single_sentence(sentence, tokenizer: AutoTokenizer, model: AutoModel, cls=False):
    device = model.device
    sentences = [sentence]

    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    with torch.no_grad():
        model_output = model(input_ids, attention_mask)

    if cls:
        sentence_embeddings = model_output[0][:, 0, :]
    else:
        sentence_embeddings = mean_pooling(model_output, attention_mask)

    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def read_MW_with_string_transformation(mw_json_fn, filter_domains: bool = True, **input_kwargs):
    # filter_domains uses the MultiWOZ-only DOMAINS list; SGD (services not in DOMAINS)
    # passes filter_domains=False so its turns aren't dropped from the pretrained index.
    data = read_json_from_data_dir(mw_json_fn)
    dial_dict = {}

    for turn in data:
        if filter_domains and not set(turn["domains"]).issubset(set(DOMAINS)):
            continue
        history = data_item_to_string(turn, **input_kwargs)
        name = f"{turn['ID']}_turn_{turn['turn_id']}"
        dial_dict[name] = history

    return dial_dict


def store_embed(input_dataset, output_filename, forward_fn):
    outputs = {}
    txt_keys = []
    with torch.no_grad():
        for k, v in tqdm(input_dataset.items()):
            outputs[k] = forward_fn(v).detach().cpu().numpy()
            txt_keys.append({k: v})
    np.save(output_filename, outputs)
    with open(output_filename.replace('.npy', '_keys.txt'), 'w') as f:
        json.dump(txt_keys, f, indent=2)
    return
