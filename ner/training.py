import pandas as pd
import numpy as np

import nltk
from datasets import Dataset
from transformers import (
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
)

from ner.ner_config import NERConfig

nltk.download("punkt")


def read_tsv_file(tsv_file):
    """
    Read tsv file, shuffle the rows.

    Args:
        tsv_file (String): Tsv file

    Returns:
        pandas.DataFrame: Shuffled Data frame
    """

    df = pd.read_csv(tsv_file, sep="\t", header=0)
    df_ranodmize_rows = df.sample(frac=1)
    return df_ranodmize_rows


def create_tokens(raw_text):
    """
    Clears a sentence from tags (<e1>, </e1>, <e2>, </e2>)
    and builds a token list.

    Args:
        raw_text (str): Raw text from data frame row

    Returns:
        list: list of tokens
    """

    new_text = (
        raw_text.replace("<e1>", " ")
        .replace("</e1>", " ")
        .replace("<e2>", " ")
        .replace("</e2>", " ")
        .replace(".", " . ")
    )
    tokens = nltk.word_tokenize(new_text)
    return tokens


def create_ner_tags(raw_text):
    """Create ner tags for raw text

    Args:
        raw_text (str): Raw text from data frame row

    Returns:
        list: list of ner tags
    """
    ner_tags_list = []
    new_text = (
        raw_text.replace(".", " . ")
        .replace("<e1>", " StartEnity1 ")
        .replace("</e1>", " StopEnity1 ")
        .replace("<e2>", " StartEnity2 ")
        .replace("</e2>", " StopEnity2 ")
    )

    text_tokens = nltk.word_tokenize(new_text)
    flag = 0
    for token in text_tokens:
        if token in ["StartEnity1", "StopEnity1", "StartEnity2", "StopEnity2"]:
            if token == "StartEnity1":
                flag = 1
            elif token == "StopEnity1":
                flag = 0
            elif token == "StartEnity2":
                flag = 3
            elif token == "StopEnity2":
                flag = 0
        elif flag == 0:
            ner_tags_list.append(NERConfig.DICT_LABELS.get("O"))
        elif flag == 1:
            ner_tags_list.append(NERConfig.DICT_LABELS.get("B-ENTITIY_1"))
            flag = 2
        elif flag == 2:
            ner_tags_list.append(NERConfig.DICT_LABELS.get("I-ENTITIY_1"))
        elif flag == 3:
            ner_tags_list.append(NERConfig.DICT_LABELS.get("B-ENTITIY_2"))
            flag = 4
        elif flag == 4:
            ner_tags_list.append(NERConfig.DICT_LABELS.get("I-ENTITIY_2"))

    return ner_tags_list


def add_tokens_ner_tags(data_frame):
    """Add column with list of tokens and list of ner tags.

    Args:
        data_frame (pandas.DataFrame): Raw Data frame

    Returns:
        pandas.DataFrame: Data frame with extra column 'tokens' and 'ner_tags'
    """
    data_frame["tokens"] = data_frame.apply(lambda row: create_tokens(row.text), axis=1)
    data_frame["ner_tags"] = data_frame.apply(
        lambda row: create_ner_tags(row.text), axis=1
    )
    return data_frame


def create_dataset(data_frame):
    """Convert tokens and ner_tags columns from data frame to Dataset.

    Args:
        data_frame (pandas.DataFrame):

    Returns:
        Dataset: Dataset object
    """

    new_data_frame = data_frame[["tokens", "ner_tags"]]
    return Dataset.from_pandas(new_data_frame)


def tokenize_adjust_labels(all_samples_per_split):
    """
    Adjust labes. For each sample, we need to get the values for input_ids,
    token_type_ids and attention_mask as well as adjust the labels.

    Args:
        all_samples_per_split (Dataset.row): Dataset row

    Returns:
        Dataset: Dataset with tokenized samples.
    """
    tokenized_samples = NERConfig.TOKENIZER.batch_encode_plus(
        all_samples_per_split["tokens"], is_split_into_words=True
    )
    # tokenized_samples is not a datasets object so
    # this alone won't work with Trainer API, hence map is used
    # so the new keys [input_ids, labels (after adjustment)]
    # can be added to the datasets dict for each train test validation split
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
                # label_name = NERConfig.LABEL_NAMES[existing_label_ids[i]]
                adjusted_label_ids.append(existing_label_ids[i])
        total_adjusted_labels.append(adjusted_label_ids)
    tokenized_samples["labels"] = total_adjusted_labels
    return tokenized_samples


def preprocess_dataset(tsv_file, split=None):
    """
    Preprocess the Dataset.
    1. Read all data frames,
    2. Add tokens and ner tags
    3. Convert to dataset
    4. Tokenize and adjust labels
    5. Split dataset for training and validation if split is not none.
    5. return dataset used for training or test model

    Args:
        tsv_file (string): tsv file
        split (float): split dataset (train and val)
    Returns:
        Dataset: Dataset with tokenized samples.
    """
    df1 = read_tsv_file(tsv_file=tsv_file)
    df2 = add_tokens_ner_tags(df1)
    d1 = create_dataset(df2)
    d2 = d1.map(tokenize_adjust_labels, batched=True)
    if split is None:
        return d2
    else:
        d3 = d2.train_test_split(split)
        train = d3["train"]
        val = d3["test"]
        return train, val


def compute_metrics(p):
    """Calculate metrics for val dataset

    Args:
        p (tuple): prediction

    Returns:
        dict: metrics result
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [NERConfig.LABEL_NAMES[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [NERConfig.LABEL_NAMES[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    results = NERConfig.METRIC.compute(
        predictions=true_predictions, references=true_labels
    )

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


def create_trainer(train_datset, val_dataset, training_arguments):
    model = BertForTokenClassification.from_pretrained(
        NERConfig.MODEL_NAME, num_labels=len(NERConfig.LABEL_NAMES)
    )
    data_collator = NERConfig.DATA_COLLATOR
    not_none_params = {k: v for k, v in training_arguments.items() if v is not None}
    args = TrainingArguments(**not_none_params)
    trainer = Trainer(
        model,
        args,
        train_dataset=train_datset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=NERConfig.TOKENIZER,
        compute_metrics=compute_metrics,
    )
    return trainer


def train_model(
    train_tsv_file, test_tsv_file, model_name, training_arguments, split=0.2
):
    """Training ner model

    Args:
        train_tsv_file (str): Training tsv file
        test_tsv_file (str, optional): Test tsv file
        model_name (str, optional): Model name
        training_arguments (dict): Training arguments
        split (float): Train/val split
    Returns:
        dic: Test result
    """
    train_dataset, val_dataset = preprocess_dataset(
        tsv_file=train_tsv_file, split=split
    )
    test_dataset = preprocess_dataset(tsv_file=test_tsv_file)
    trainer = create_trainer(
        train_datset=train_dataset,
        val_dataset=val_dataset,
        training_arguments=training_arguments,
    )
    trainer.train()
    result = trainer.evaluate(test_dataset)
    print("EVALUATE: ", result)
    trainer.save_model(model_name)
    return result
