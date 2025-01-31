import json
import os

import numpy as np
from datasets import load_metric
from seqeval.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def get_texts_and_labels(df, model_path, read=False):
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    if read:
        with open(f"{model_path}/map.json") as map_file:
            map = json.load(map_file)
    else:
        map = dict(
            [(y, x) for x, y in enumerate(sorted(set(labels)))]
        )  # get a dict of distinct labels and their numbers
        map_path = f"{model_path}/map.json"
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        with open(map_path, "w+") as f:
            json.dump(map, f)
    labels = [map[x] for x in labels]
    return texts, labels


def prune_prefixes_from_labels(predictions):
    predicted_labels = []
    for prediction in predictions:
        label = prediction["label"]
        label = label.replace("LABEL_", "")
        label = int(label)
        predicted_labels.append(label)
    return predicted_labels


def map_result_to_text(result, model_path):
    map_path = f"{model_path}/map.json"
    with open(map_path, "r") as f:
        map = json.loads(f.read())
    reverse_label_map = {value: key for key, value in map.items()}
    result = [reverse_label_map[label] for label in result]
    return result


def calculate_metrics(labels, predictions, average_type="micro"):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=average_type)
    accuracy = accuracy_score(labels, predictions)
    print(f"PRECISION: {round(precision,6)}")
    print(f"RECALL: {round(recall, 6)}")
    print(f"F1 SCORE: {round(f1, 6)}")
    print(f"ACCURACY: {round(accuracy, 6)}")
    result = {"precision": precision, "accuracy": accuracy, "recall": recall, "f1": f1}
    return result


def compute_metrics(p):
    metric = load_metric("f1")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    results = metric.compute(predictions=predictions, references=labels, average="micro")
    print(f"RESULTS: {results}")
    return results


def remove_tags_from_dataframe(df):
    tags = ['<e1>', '</e1>', '<e2>', '</e2>']
    for tag in tags:
        df['text'] = df['text'].str.replace(tag, '', regex=False)
    return df
