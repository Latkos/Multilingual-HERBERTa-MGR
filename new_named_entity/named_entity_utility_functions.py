import json
import os

import nltk
from seqeval.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def extract_entities(tokens, special_elements):
    new_tokens = [token for token in tokens if token not in special_elements]
    labels = ['O'] * len(new_tokens)
    entity_starts = [i for i, token in enumerate(tokens) if "StartEntity" in token]
    entity_ends = [i for i, token in enumerate(tokens) if "StopEntity" in token]
    for start, end in zip(entity_starts, entity_ends):
        entity_label = tokens[start].split("Start")[1]
        start = new_tokens.index(tokens[start + 1])
        end = new_tokens.index(tokens[end - 1]) + 1
        labels[start] = "B-" + entity_label
        for i in range(start + 1, end):
            labels[i] = "I-" + entity_label
    return new_tokens, labels



def create_tokens_and_labels(text):
    replacement_map = {"<e1>": " StartEntity1 ", "</e1>": " StopEntity1 ", "<e2>": " StartEntity2 ", "</e2>": " StopEntity2 "}
    for word, replacement in replacement_map.items():
        text = text.replace(word, replacement)
    text_tokens = nltk.word_tokenize(text)
    special_elements=list(replacement_map.values())
    tokens,labels=extract_entities(text_tokens,special_elements)
    return tokens,labels



def get_labels(df):
    df['tokens'],df['labels']=zip(*df['text'].map(create_tokens_and_labels))
    return df



# # Get the values for input_ids, token_type_ids, attention_mask
# def tokenize_adjust_labels(tokenizer, all_samples_per_split):
#     dataset = dataset.map(lambda e: tokenizer(e['sentence1'], truncation=True, padding='max_length'), batched=True
#     tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], is_split_into_words=True)
#     # tokenized_samples is not a datasets object so this alone won't work with Trainer API, hence map is used
#     # so the new keys [input_ids, labels (after adjustment)]
#     # can be added to the datasets dict for each train test validation split
#     total_adjusted_labels = []
#     print(len(tokenized_samples["input_ids"]))
#     for k in range(0, len(tokenized_samples["input_ids"])):
#         prev_wid = -1
#         word_ids_list = tokenized_samples.word_ids(batch_index=k)
#         existing_label_ids = all_samples_per_split["ner_tags"][k]
#         i = -1
#         adjusted_label_ids = []
#         '''print(word_ids_list)
#         print(existing_label_ids)
#         print(all_samples_per_split["tokens"][k])
#         print(tokenized_samples["input_ids"][k])'''
#         for wid in word_ids_list:
#             if (wid is None):
#                 adjusted_label_ids.append(-100)
#             elif (wid != prev_wid):
#                 i = i + 1
#                 adjusted_label_ids.append(existing_label_ids[i])
#                 prev_wid = wid
#             else:
#                 label_name = label_names[existing_label_ids[i]]
#                 '''if(label_name == "O"):
#                   adjusted_label_ids.append(existing_label_ids[i])
#                 elif(label_name[0:2]=="B-"):
#                   adjusted_label_ids.append(label_names.index(label_name.replace("B-","I-")))
#                 else:
#                   adjusted_label_ids.append(existing_label_ids[i])'''
#                 adjusted_label_ids.append(existing_label_ids[i])
#
#         total_adjusted_labels.append(adjusted_label_ids)
#     tokenized_samples["labels"] = total_adjusted_labels
#     return tokenized_samples
