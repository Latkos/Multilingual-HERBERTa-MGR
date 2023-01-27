import click
from transformers import Trainer, BertForTokenClassification, DataCollatorForTokenClassification, AutoTokenizer
from transformers.data import data_collator

from ner import training as ner_training
from ner import prediction as ner_prediction
from config_parser import get_training_args

import pandas as pd

from ner.ner_config import NERConfig
from ner.training import compute_metrics, preprocess_dataset
from relations.relations_model import re_train_model, re_evaluate_model, re_predict


@click.group()
def bert_cli():
    """
Command Line Interpreter for multilingual m-bert model\
for named entity recognition (NER) and relations extraction (RE)

Example:

- python main_in_works.py train ./data/en-small_corpora_train.tsv \
./data/en-small_corpora_test.tsv --config ./config/base_config.yaml \
--model_path_ner en-small_corpora

- python main_in_works.py predict "Kyrgyz International Airlines was an airline based in Kyrgyzstan." \
--model_path_ner en-small_corpora
"""
    pass


@bert_cli.command(
    help="""Train M-BERT model for named entity recognition (NER) and
relations extraction (RE) with train file test file and config file""",
    short_help="Train M-BERT model",
)
@click.option(
    "--config", default="./config/base_config.yaml", type=str, help="path to train config"
)
@click.option("--model_path_ner", default=None, type=str, help="model path for NER")
@click.option("--model_path_re", default=None, type=str, help="model path for RE")
@click.argument("train_file", nargs=1, required=True, type=str)
@click.argument("test_file", nargs=1, required=True, type=str)
@click.option("--split", nargs=1, default=0.2, type=float, help="Train/val split")
def train(train_file, test_file, config, model_path_ner, model_path_re, split):
    if model_path_ner:
        args = get_training_args(config, "ner")
        ner_training.train_model(
            train_tsv_file=train_file,
            test_tsv_file=test_file,
            model_name=model_path_ner,
            training_arguments=args,
            split=split
        )
    if model_path_re:
        args = get_training_args(config, "re")
        re_train_model(train_file=train_file, model_path=model_path_re, training_arguments=args, split=split)
        re_evaluate_model(test_file, model_path_re)


@bert_cli.command(
    help="""Predict entity_1, entity_2 and relation between entities with \
pretrained M-BERT model for inputted text""",
    short_help="Predict entity_1, entity_2, relation",
)
@click.argument("text", nargs=1, required=True, type=str)
@click.option(
    "--model_path_ner", nargs=1, default=None, type=str, help="model path NER"
)
@click.option("--model_path_re", nargs=1, default=None, type=str, help="model path RE")
def predict(text, model_path_ner, model_path_re):
    final_prediction = pd.DataFrame()
    if type(text) != list:
        text = [text]
    if model_path_ner:
        sentence = " ".join(text)
        prediction_result = ner_prediction.prediction(
            model_name=model_path_ner, sentences=sentence
        )
        print("NER result:", prediction_result)
        text = []
        for prediction in prediction_result:
            text.append(prediction["TEXT"])
    if model_path_re:
        prediction_result = re_predict(text=text, model_path=model_path_re)
        final_prediction["relation"] = prediction_result
        print("RE result:", prediction_result)


if __name__ == "__main__":
    model = BertForTokenClassification.from_pretrained(
        '3languages_ner', num_labels=5).to("cuda:0")
    MODEL_NAME = "bert-base-multilingual-cased"
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator=DataCollatorForTokenClassification(TOKENIZER)
    trainer = Trainer(
        model,
        data_collator=data_collator,
        tokenizer=NERConfig.TOKENIZER,
        compute_metrics=compute_metrics,
    )
    test_dataset = preprocess_dataset(tsv_file='test-3languages.tsv')
    trainer.evaluate(test_dataset)
    re_evaluate_model('test-3languages.tsv','3languages_re')
