import re
import nltk
from datasets import Dataset, load_metric
import numpy as np
from config import general_config


def create_dataset_from_dataframe(df):
    df = preprocess_dataframe(df, general_config.label_mapping)
    dataset = Dataset.from_pandas(df)
    return dataset


def preprocess_dataframe(df, label_mapping):
    df["tokens"], df["labels"] = zip(*df["text"].map(create_tokens_and_labels_for_two_entities))
    df["ner_tags"] = df["labels"].apply(lambda x: [label_mapping[i] for i in x])
    return df


def extract_tokens_and_create_labels(tokens, special_elements):
    new_tokens = [token for token in tokens if token not in special_elements]
    labels = ["O"] * len(new_tokens)
    entity_starts = [i for i, token in enumerate(tokens) if "StartEntity" in token]
    entity_ends = [i for i, token in enumerate(tokens) if "StopEntity" in token]
    for start, end in zip(entity_starts, entity_ends):
        entity_label = tokens[start].split("Start")[1]
        try:
            start = new_tokens.index(tokens[start + 1])
            end = new_tokens.index(tokens[end - 1]) + 1
        except Exception as e:
            print(f"Exception: {tokens}")
            print(f"Tokens: {tokens}")
        labels[start] = "B-" + entity_label
        for i in range(start + 1, end):
            labels[i] = "I-" + entity_label
    return new_tokens, labels


def create_tokens_and_labels_for_two_entities(text):
    replacement_map = {
        "<e1>": " StartEntity1 ",
        "</e1>": " StopEntity1 ",
        "<e2>": " StartEntity2 ",
        "</e2>": " StopEntity2 ",
    }
    for word, replacement in replacement_map.items():
        text = text.replace(word, replacement)
    text_tokens = nltk.word_tokenize(text)
    special_elements = ["StartEntity1", "StopEntity1", "StartEntity2", "StopEntity2"]
    tokens, labels = extract_tokens_and_create_labels(text_tokens, special_elements)
    return tokens, labels


def split_dataset(ds, split=0.2):
    split_ds = ds.train_test_split(split)
    train = split_ds["train"]
    val = split_ds["test"]
    return train, val

def remove_tags(sentence):
    sentence=re.sub(r'<[^>]+>', '', sentence)
    return sentence

def compute_metrics(p):
    metric = load_metric("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [general_config.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [general_config.label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }
    for k in results.keys():
        if k not in flattened_results.keys():
            flattened_results[k + "_f1"] = results[k]["f1"]
    return flattened_results

def compute_objective(p):
    result = compute_metrics(p)
    return result["overall_f1"]

def get_unlabeled_text(sentence):
    return {"entity_1": "", "entity_2": "", "text": sentence}

def get_model_output_as_sentence(ner_output, sentence):
    # Initialize variables for entity tracking
    entity_1_start, entity_1_end, entity_2_start, entity_2_end = None, None, None, None
    entity_1, entity_2 = "", ""
    offset = 0

    for item in ner_output:
        # Check for LABEL_1 and LABEL_2 (Entity 1)
        if item["entity_group"] in ["LABEL_1", "LABEL_2"]:
            if entity_1_start is None:
                entity_1_start = item["start"]
            entity_1_end = item["end"]
            entity_1 += item["word"] + " "
        # Check for LABEL_3 and LABEL_4 (Entity 2)
        elif item["entity_group"] in ["LABEL_3", "LABEL_4"]:
            if entity_2_start is None:
                entity_2_start = item["start"]
            entity_2_end = item["end"]
            entity_2 += item["word"] + " "

    # Add tags for Entity 1
    if entity_1_start is not None:
        sentence = sentence[:entity_1_start + offset] + "<e1>" + sentence[entity_1_start + offset:entity_1_end + offset] + "</e1>" + sentence[entity_1_end + offset:]
        offset += 9  # Length of <e1></e1>

    # Add tags for Entity 2
    if entity_2_start is not None:
        sentence = sentence[:entity_2_start + offset] + "<e2>" + sentence[entity_2_start + offset:entity_2_end + offset] + "</e2>" + sentence[entity_2_end + offset:]
        offset += 9  # Length of <e2></e2>

    return {"entity_1": entity_1.strip(), "entity_2": entity_2.strip(), "text": sentence}

# Example usage:
# result = get_model_output_as_sentence(ner_output, sentence)

