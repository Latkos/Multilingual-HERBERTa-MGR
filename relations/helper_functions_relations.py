import json
import os
from seqeval.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd


def get_texts_and_labels(df, model_path):
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    map = dict([(y, x) for x, y in enumerate(sorted(set(labels)))])
    labels = [map[x] for x in labels]
    map_path = f"{model_path}/map.json"
    inverted_map = {v: k for k, v in map.items()}
    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    with open(map_path, "w+") as f:
        json.dump(inverted_map, f)
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


def calculate_metrics(test_labels, predicted_test_labels):
    precision, recall, f1, _= precision_recall_fscore_support(
        test_labels, predicted_test_labels, average=None
    )
    accuracy = accuracy_score(test_labels, predicted_test_labels)
    print("PRECISION: %.2f" % precision)
    print("RECALL: %.2f" % recall)
    print("F1 SCORE: %.2f" % f1)
    print("ACCURACY: %.2f" % accuracy)
    return precision, recall
