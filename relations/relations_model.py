import json

import torch
from tokenizers.trainers import Trainer

from base_model.base_model import BaseModel
from relations.relations_dataset import RelationsDataset
from relations.relations_utility_functions import (
    prune_prefixes_from_labels,
    map_result_to_text,
    calculate_metrics,
    get_texts_and_labels,
    get_f1_from_metrics,
)
from sklearn.model_selection import train_test_split
from transformers import (
    Trainer,
    BertForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)

from utils.config_parser import get_training_args


class RelationsModel(BaseModel):
    def __init__(self, model_path, model_name="bert-base-multilingual-cased"):
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def create_trainer(
        self,
        train_df,
        model_path,
        training_arguments=None,
        split=0.2,
        config_path="./config/base_config.yaml",
        model_init=None,
    ):
        if training_arguments is None:
            training_arguments = get_training_args(config_path=config_path, model_type="re")
        train_texts, train_labels = get_texts_and_labels(train_df, model_path)

        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=split)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)
        train_dataset = RelationsDataset(train_encodings, train_labels)
        val_dataset = RelationsDataset(val_encodings, val_labels)
        with open(f"{model_path}/map.json") as map_file:
            map = json.load(map_file)
        labels_number = len(map)
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased", num_labels=labels_number
        ).to("cuda:0")
        not_none_params = {k: v for k, v in training_arguments.items() if v is not None}
        training_args = TrainingArguments(**not_none_params)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            model_init=model_init,
        )
        return trainer

    def train(
        self, train_df, model_path=None, training_arguments=None, split=0.2, config_path="./config/base_config.yaml"
    ):
        if model_path is None:
            model_path=self.model_path
        trainer = self.create_trainer(
            train_df=train_df,
            model_path=model_path,
            training_arguments=training_arguments,
            split=split,
            config_path=config_path,
        )
        trainer.train()
        trainer.save_model(model_path)

    def evaluate(self, test_df, model_path=None, average_type="weighted"):
        if model_path is None:
            model_path = self.model_path
        test_texts, test_labels = get_texts_and_labels(test_df, model_path,read=True)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        with open(f"{model_path}/map.json") as map_file:
            map=json.load(map_file)
        labels_number=len(map)
        print(f"Labels number: {labels_number}")
        model = BertForSequenceClassification.from_pretrained(model_path,num_labels=labels_number).to("cuda:0")
        generator = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=0)
        predicted_test_labels = generator(test_texts)
        predicted_test_labels = prune_prefixes_from_labels(predicted_test_labels)
        result = calculate_metrics(
            predictions=predicted_test_labels,
            labels=test_labels,
            average_type=average_type,
        )
        return result

    def predict(self, text, model_path=None):
        if model_path is None:
            model_path = self.model_path
        with open(f"{model_path}/map.json") as map_file:
            map = json.load(map_file)
        labels_number = len(map)
        model = BertForSequenceClassification.from_pretrained(model_path,num_labels=labels_number).to("cuda:0")
        generator = pipeline(task="text-classification", model=model, tokenizer=self.tokenizer, device=0)
        # TODO: check how will it work if we cast predict on a previously trained model
        predicted_labels = generator(text)
        predicted_numeric_labels = prune_prefixes_from_labels(predicted_labels)
        result = map_result_to_text(predicted_numeric_labels, model_path)
        return result

    def evaluate_with_division_between_languages(self, test_df, model_path=None, average_type="weighted"):
        evaluation_results = {}
        print("COMMENCING EVALUATION")
        print(test_df["lang"].unique())
        for language in test_df["lang"].unique():
            print(f"Language: {language}")
            lang_df = test_df[test_df["lang"] == language]
            evaluation_results["language"] = self.evaluate(
                test_df=lang_df, model_path=model_path, average_type=average_type
            )
        return evaluation_results

    def model_init(self, trial):
        return BertForSequenceClassification.from_pretrained(self.model_type).to("cuda:0")

    def perform_hyperparameter_search(
        self,
        space,
        train_df,
        model_path=None,
        config_path="./config/base_config.yaml",
    ):
        if model_path is None:
            model_path=self.model_path
        trainer = self.create_trainer(
            train_df=train_df,
            model_path=model_path,
            config_path=config_path,
            model_init=self.model_init,
        )
        best_trial = trainer.hyperparameter_search(
            direction="maximize", backend="optuna", hp_space=space, n_trials=50, compute_objective=get_f1_from_metrics
        )
        return best_trial
