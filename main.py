import click

from named_entity.named_entity_model import NamedEntityModel
from relations.relations_model import RelationsModel
from utils.config_parser import get_training_args
from utils import csv_merger
import pandas as pd
from transformers import DataCollatorForTokenClassification, AutoModelForSequenceClassification

# @click.group()
# def bert_cli():
#     """
# Command Line Interpreter for multilingual m-bert model\
# for named entity recognition (NER) and old_relations extraction (RE)
#
# Example:
#
# - python main.py train ./data/en-small_corpora_train.tsv \
# ./data/en-small_corpora_test.tsv --config ./config/base_config.yaml \
# --model_path_ner en-small_corpora
#
# - python main.py predict "Kyrgyz International Airlines was an airline based in Kyrgyzstan." \
# --model_path_ner en-small_corpora
# """
#     pass
#
#
# @bert_cli.command(
#     help="""Train M-BERT model for named entity recognition (NER) and
# old_relations extraction (RE) with train file test file and config file""",
#     short_help="Train M-BERT model",
# )
# @click.option(
#     "--config", default="./config/base_config.yaml", type=str, help="path to train config"
# )
# @click.option("--model_path_ner", default=None, type=str, help="model path for NER")
# @click.option("--model_path_re", default=None, type=str, help="model path for RE")
# @click.argument("train_file", nargs=1, required=True, type=str)
# @click.argument("test_file", nargs=1, required=True, type=str)
# @click.option("--split", nargs=1, default=0.2, type=float, help="Train/val split")
# def train(train_file, test_file, config, model_path_ner, model_path_re, split):
#     if model_path_ner:
#         args = get_training_args(config, "ner")
#         ner_training.train_model(
#             train_tsv_file=train_file,
#             test_tsv_file=test_file,
#             model_name=model_path_ner,
#             training_arguments=args,
#             split=split
#         )
#     if model_path_re:
#         args = get_training_args(config, "re")
#         re_train_model(train_file=train_file, model_path=model_path_re, training_arguments=args, split=split)
#         re_evaluate_model(test_file, model_path_re)
#
#
# @bert_cli.command(
#     help="""Predict entity_1, entity_2 and relation between entities with \
# pretrained M-BERT model for inputted text""",
#     short_help="Predict entity_1, entity_2, relation",
# )
# @click.argument("text", nargs=1, required=True, type=str)
# @click.option(
#     "--model_path_ner", nargs=1, default=None, type=str, help="model path NER"
# )
# @click.option("--model_path_re", nargs=1, default=None, type=str, help="model path RE")
# def predict(text, model_path_ner, model_path_re):
#     final_prediction = pd.DataFrame()
#     if type(text) != list:
#         text = [text]
#     if model_path_ner:
#         sentence = " ".join(text)
#         prediction_result = ner_prediction.prediction(
#             model_name=model_path_ner, sentences=sentence
#         )
#         print("NER result:", prediction_result)
#         text = []
#         for prediction in prediction_result:
#             text.append(prediction["TEXT"])
#     if model_path_re:
#         prediction_result = re_predict(text=text, model_path=model_path_re)
#         final_prediction["relation"] = prediction_result
#         print("RE result:", prediction_result)

INCORRECT_SENTENCES_IDS=[215077, ]

def filter_out_wrong_data(df):
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    df = df.dropna(subset=['id'])
    df['id'] = df['id'].astype('int')
    df=df[~df['text'].str.contains('<e2</e1>')]
    df=df[~df['text'].str.contains('<e2></e2>')]
    return df

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 12, 16, 20, 24, 28, 32, 36]),
    }


if __name__ == "__main__":
    train_df = pd.read_csv("data/merged_train.tsv", sep="\t")
    test_df = pd.read_csv("data/merged_test.tsv", sep="\t")
    train_df=train_df.sample(frac=0.1)
    test=test_df.sample(frac=0.1)

    # train_df = pd.read_csv("data/en-small_corpora_train.tsv", sep="\t")
    # test_df = pd.read_csv("data/en-small_corpora_test.tsv", sep="\t")

    train_df=filter_out_wrong_data(train_df)
    test_df=filter_out_wrong_data(test_df)
    ner_model=NamedEntityModel()
    ner_model.perform_hyperparameter_search(train_df=train_df,space=optuna_hp_space)

    # train_df=train_df.sample(frac=0.25)
    # # ner_model=NamedEntityModel(model_path="models/ner-en-small%")
    # ner_model.train(train_df)
    # ner_model.evaluate(test_df)
    # csv_merger.merge_csv_files(path='data',pattern="*train.tsv",result_file_name="merged_train.tsv")
    # csv_merger.merge_csv_files(path='data',pattern="*test.tsv",result_file_name="merged_test.tsv")
