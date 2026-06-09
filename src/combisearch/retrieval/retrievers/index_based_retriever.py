"""Dense KDTree retriever over a saved embedding index.
"""
import random

import numpy as np
from scipy.spatial import KDTree


class Retriever:
    """General cosine-similarity dense retriever."""

    def normalize(self, emb):
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)

    def __init__(self, emb_dict):

        self.emb_keys = list(emb_dict.keys())
        emb_dim = emb_dict[self.emb_keys[0]].shape[-1]

        self.emb_values = np.zeros((len(self.emb_keys), emb_dim))
        for i, k in enumerate(self.emb_keys):
            self.emb_values[i] = emb_dict[k]

        # normalize for cosine distance (kdtree only support euclidean when p=2)
        self.emb_values = self.normalize(self.emb_values)
        self.kdtree = KDTree(self.emb_values)

    def topk_nearest_dialogs(self, query_emb, k=5):
        query_emb = self.normalize(query_emb)
        if k == 1:
            return [self.emb_keys[i] for i in self.kdtree.query(query_emb, k=k, p=2)[1]]
        return [self.emb_keys[i] for i in self.kdtree.query(query_emb, k=k, p=2)[1][0]]

    def topk_nearest_distinct_dialogs(self, query_emb, k=5):
        return self.topk_nearest_dialogs(query_emb, k=k)


class IndexRetriever:
    """Cosine-similarity dense retriever that filters the search index to a subset of dialogues."""

    @staticmethod
    def random_sample_selection_by_turn(embs, ratio=0.1):
        n_selected = int(ratio * len(embs))
        print(f"randomly select {ratio} of turns, i.e. {n_selected} turns")
        selected_keys = random.sample(list(embs), n_selected)
        return {k: v for k, v in embs.items() if k in selected_keys}

    @staticmethod
    def random_sample_selection_by_dialog(embs, ratio=0.1):
        dial_ids = set([turn_label.split('_')[0] for turn_label in embs.keys()])
        n_selected = int(len(dial_ids) * ratio)
        print(f"randomly select {ratio} of dialogs, i.e. {n_selected} dialogs")
        selected_dial_ids = random.sample(dial_ids, n_selected)
        return {k: v for k, v in embs.items() if k.split('_')[0] in selected_dial_ids}

    @staticmethod
    def pre_assigned_sample_selection(embs, examples):
        selected_dial_ids = set([dial['ID'] for dial in examples])
        return {k: v for k, v in embs.items() if k.split('_')[0] in selected_dial_ids}

    @staticmethod
    def sgd_pre_assigned_sample_selection(embs, examples):
        # SGD dialogue ids are the first two '_'-separated fields (e.g. "1_00000"),
        # unlike MultiWOZ ids which have no underscore.
        selected_dial_ids = set([dial['ID'] for dial in examples])
        return {k: v for k, v in embs.items() if '_'.join(k.split('_')[:2]) in selected_dial_ids}

    def __init__(self, datasets, embedding_filenames, search_index_filename, sampling_method="none", ratio=1.0):

        self.data_items = []
        for dataset in datasets:
            self.data_items += dataset

        self.all_embeddings = {}
        for fn in embedding_filenames:
            this_embs = np.load(fn, allow_pickle=True).item()
            for k, v in this_embs.items():
                self.all_embeddings[k] = v

        self.search_embs = np.load(search_index_filename, allow_pickle=True).item()

        if sampling_method == "none":
            self.retriever = Retriever(self.search_embs)
        elif sampling_method == 'random_by_dialog':
            self.retriever = Retriever(self.random_sample_selection_by_dialog(self.search_embs, ratio=ratio))
        elif sampling_method == 'random_by_turn':
            self.retriever = Retriever(self.random_sample_selection_by_turn(self.search_embs, ratio=ratio))
        elif sampling_method == 'pre_assigned':
            self.retriever = Retriever(self.pre_assigned_sample_selection(self.search_embs, self.data_items))
        elif sampling_method == 'sgd_pre_assigned':
            self.retriever = Retriever(self.sgd_pre_assigned_sample_selection(self.search_embs, self.data_items))
        else:
            raise ValueError("selection method not supported")

    def data_item_to_embedding(self, data_item):
        ID = data_item['ID']
        turn = data_item['turn_id']
        label = f"{ID}_turn_{turn}"

        return self.all_embeddings[label]

    def label_to_data_item(self, label):
        # split on the '_turn_' marker only, so SGD ids (which contain '_') parse correctly
        ID, turn_id = label.split('_turn_')
        turn_id = int(turn_id)

        for d in self.data_items:
            if d['ID'] == ID and d['turn_id'] == turn_id:
                return d
        raise ValueError(f"label {label} not found. check data items input")

    def label_to_nearest_labels(self, label, k=5):
        data_item = self.label_to_data_item(label)
        return [l for l in self.retriever.topk_nearest_distinct_dialogs(
            self.data_item_to_embedding(data_item), k=k)
                ][::-1]
