import json
import os

import nltk
from datasets import Dataset
from seqeval.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer

from new_named_entity import ner_config


def create_dataset_from_dataframe(df):
    df=preprocess_dataframe(df, ner_config.label_mapping)
    dataset=Dataset.from_pandas(df)
    return dataset

def preprocess_dataframe(df, label_mapping):
    df['tokens'], df['labels'] = zip(*df['text'].map(create_tokens_and_labels))
    df['ner_tags'] = df['labels'].apply(lambda x: [label_mapping[i] for i in x])
    return df

def extract_tokens_and_create_labels(tokens, special_elements):
    new_tokens = [token for token in tokens if token not in special_elements]
    labels = ['O'] * len(new_tokens)
    entity_starts = [i for i, token in enumerate(tokens) if "StartEntity" in token]
    entity_ends = [i for i, token in enumerate(tokens) if "StopEntity" in token]
    for start, end in zip(entity_starts, entity_ends):
        entity_label = tokens[start].split("Start")[1]
        try:
            start = new_tokens.index(tokens[start + 1])
            end = new_tokens.index(tokens[end - 1]) + 1
        except Exception as e:
            print(str(e))
            print(f"Tokens: {tokens}")
        labels[start] = "B-" + entity_label
        for i in range(start + 1, end):
            labels[i] = "I-" + entity_label
    return new_tokens, labels


def create_tokens_and_labels(text):
    replacement_map = {"<e1>": " StartEntity1 ", "</e1>": " StopEntity1 ", "<e2>": " StartEntity2 ",
                       "</e2>": " StopEntity2 "}
    for word, replacement in replacement_map.items():
        text = text.replace(word, replacement)
    text_tokens = nltk.word_tokenize(text)
    # TODO unhardcode this
    special_elements = ['StartEntity1','StopEntity1','StartEntity2','StopEntity2']
    tokens, labels = extract_tokens_and_create_labels(text_tokens, special_elements)
    return tokens, labels


def split_dataset(ds, split=0.2):
    split_ds = ds.train_test_split(split)
    train = split_ds["train"]
    val = split_ds["test"]
    return train, val

    def tokenize_adjust_labels(self, all_samples_per_split):
        tokenized_samples = self.tokenizer.batch_encode_plus(all_samples_per_split["tokens"], is_split_into_words=True)
        total_adjusted_labels = []
        for k in range(0, len(tokenized_samples["input_ids"])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = all_samples_per_split["ner_tags"][k]
            i = -1
            adjusted_label_ids = []
            for wid in word_ids_list:
                if wid is None:
                    adjusted_label_ids.append(-100)
                elif wid != prev_wid:
                    i = i + 1
                    adjusted_label_ids.append(existing_label_ids[i])
                    prev_wid = wid
                else:
                    adjusted_label_ids.append(existing_label_ids[i])

            total_adjusted_labels.append(adjusted_label_ids)
        tokenized_samples["labels"] = total_adjusted_labels
        return tokenized_samples
