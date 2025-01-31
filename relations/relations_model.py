import json

import optuna

from relations.relations_dataset import RelationsDataset
from relations.relations_utility_functions import (
    prune_prefixes_from_labels,
    map_result_to_text,
    calculate_metrics,
    get_texts_and_labels,
    compute_metrics, remove_tags_from_dataframe
)
from sklearn.model_selection import train_test_split
from transformers import (
    Trainer,
    AutoTokenizer,
    TrainingArguments,
    pipeline, AutoModelForSequenceClassification, EarlyStoppingCallback,
)
from utils.config_parser import get_training_args
from utils.evaluation import get_f1_from_metrics
from utils.enhancement import enhance_with_brackets


class RelationsModel():
    def __init__(self, model_path="./re", model_type="bert-base-multilingual-cased"):
        self.model_path = model_path
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)

    def train(
        self,
        train_df,
        model_path=None,
        training_arguments=None,
        split=0.2,
        config_path="./config/base_config.yaml",
        remove_tags=True,
        early_stopping=True
    ):
        if model_path is None:
            model_path = self.model_path

        trainer = self.create_trainer(
            train_df=train_df,
            model_path=model_path,
            training_arguments=training_arguments,
            split=split,
            config_path=config_path,
            remove_tags=remove_tags,
            early_stopping=early_stopping
        )
        trainer.train()
        trainer.save_model(model_path)

    def create_trainer(
        self,
        train_df,
        model_path,
        config_path,
        training_arguments=None,
        split=0.2,
        model_init=None,
        remove_tags=True,
        early_stopping=True
    ):
        training_args=None
        if not training_arguments:
            if config_path:
                training_arguments = get_training_args(config_path=config_path, model_type="re")
        if remove_tags:
            train_df=remove_tags_from_dataframe(train_df)
        texts, labels = get_texts_and_labels(train_df, model_path)
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=split)
        tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)
        train_dataset = RelationsDataset(train_encodings, train_labels)
        val_dataset = RelationsDataset(val_encodings, val_labels)
        with open(f"{model_path}/map.json") as map_file:
            map = json.load(map_file)
        labels_number = len(map)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_type, num_labels=labels_number
        ).to("cuda:0")
        if training_arguments:
            not_none_params = {k: v for k, v in training_arguments.items() if v is not None}
            training_args = TrainingArguments(**not_none_params)
        callbacks=None
        if early_stopping:
            callbacks=[EarlyStoppingCallback(1, 0.0)]
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        return trainer

    def evaluate(self, df, model_path=None, average_type="micro", enhancement_func=enhance_with_brackets):
        if model_path is None:
            model_path = self.model_path
        enhanced_test_df = df.copy()
        enhancement_input = enhanced_test_df[['text', 'entity_1', 'entity_2']].to_dict(orient='records')
        enhancement_test = enhancement_func(enhancement_input)
        enhanced_test_df['text'] = enhancement_test
        texts, labels = get_texts_and_labels(enhanced_test_df, model_path, read=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        with open(f"{model_path}/map.json") as map_file:
            map = json.load(map_file)
        labels_number = len(map)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=labels_number).to("cuda:0")
        generator = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=0)
        predicted_test_labels = generator(texts)
        predicted_test_labels = prune_prefixes_from_labels(predicted_test_labels)
        result = calculate_metrics(
            predictions=predicted_test_labels,
            labels=labels,
            average_type=average_type,
        )
        return result

    def predict(self, sentences, model_path=None):
        if model_path is None:
            model_path = self.model_path
        with open(f"{model_path}/map.json") as map_file:
            map = json.load(map_file)
        labels_number = len(map)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=labels_number).to("cuda:0")
        generator = pipeline(task="text-classification", model=model, tokenizer=self.tokenizer, device=0)
        predicted_labels = generator(sentences)
        predicted_numeric_labels = prune_prefixes_from_labels(predicted_labels)
        result = map_result_to_text(predicted_numeric_labels, model_path)
        return result

    def model_init(self, trial):
        with open(f"{self.model_path}/map.json") as map_file:
            map = json.load(map_file)
        labels_number = len(map)
        return AutoModelForSequenceClassification.from_pretrained(self.model_type, num_labels=labels_number).to("cuda:0")

    def get_study_name(self,trial: optuna.Trial):
        return f"{self.__class__.__name__}_{trial.number}"

    def perform_hyperparameter_search(
        self,
        space,
        train_df,
        study_name,
        number_of_trials=50,
        model_path=None,
        config_path="./config/base_config.yaml",
        storage='sqlite:///example.db',
        load_if_exists=True,
        remove_tags=False
    ):
        if model_path is None:
            model_path = self.model_path
        trainer = self.create_trainer(
            train_df=train_df,
            model_path=model_path,
            config_path=config_path,
            model_init=self.model_init,
            remove_tags=remove_tags
        )
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=space,
            n_trials=number_of_trials,
            compute_objective=get_f1_from_metrics,
            storage=storage,
            load_if_exists=load_if_exists,
            study_name=study_name
        )
        return best_trial
