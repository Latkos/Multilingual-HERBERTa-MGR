import json
import os

import numpy as np
from datasets import load_metric
from seqeval.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def get_texts_and_labels(df, model_path,read=False):
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    if read:
        with open(f"{model_path}/map.json") as map_file:
            map=json.load(map_file)
    else:
        map = dict([(y, x) for x, y in enumerate(sorted(set(labels)))])  # get a dict of distinct labels and their numbers
        map_path= f"{model_path}/map.json"
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        with open(map_path, "w+") as f:
            json.dump(map, f)
    labels = [map[x] for x in labels]
    # map_path =
    # inverted_map = {v: k for k, v in map.items()}
    # print(f"INVERTED MAP: {inverted_map}")
    # os.makedirs(os.path.dirname(map_path), exist_ok=True)
    # with open(map_path, "w+") as f:
    #     json.dump(inverted_map, f)
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
    result = [map[str(label)] for label in result]
    return result


def calculate_metrics(labels, predictions, average_type="micro"):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=average_type)
    accuracy = accuracy_score(labels, predictions)
    print("PRECISION: %.2f" % precision)
    print("RECALL: %.2f" % recall)
    print("F1 SCORE: %.2f" % f1)
    print("ACCURACY: %.2f" % accuracy)
    result = {"precision": precision, "accuracy": accuracy, "recall": recall, "f1": f1}
    return result


def get_f1_from_metrics(metrics):
    print("IN FUNCTION get_f1_from_metrics, the variable metrics has value:")
    print(metrics)
    f1 = metrics["f1"]
    return f1
