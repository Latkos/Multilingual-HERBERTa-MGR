import torch
from tokenizers.trainers import Trainer

from base_model.base_model import BaseModel
from relations.relations_dataset import RelationsDataset
from relations.relations_utility_functions import (
    prune_prefixes_from_labels,
    map_result_to_text,
    calculate_metrics,
    get_texts_and_labels,
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
        self.last_trainer = None

    def train(self, train_df, model_path=None, training_arguments=None, split=0.2, config_path='./config/base_config.yaml'):
        if training_arguments is None:
            training_arguments=get_training_args(config_path=config_path,model_type="re")
        if model_path is None:
            model_path = self.model_path
        if training_arguments is None:
            training_arguments = {}
        train_texts, train_labels = get_texts_and_labels(train_df, model_path)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=split
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)
        train_dataset = RelationsDataset(train_encodings, train_labels)
        val_dataset = RelationsDataset(val_encodings, val_labels)
        labels_number = len(set(train_labels))
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
        )
        trainer.train()
        trainer.save_model(model_path)
        return model

    def evaluate(self, test_df, model_path=None, average_type="weighted"):
        if model_path is None:
            model_path = self.model_path
        test_texts, test_labels = get_texts_and_labels(test_df, model_path)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = BertForSequenceClassification.from_pretrained(
            model_path, num_labels=21
        ).to("cuda:0")
        generator = pipeline(
            task="text-classification", model=model, tokenizer=tokenizer, device=0
        )
        predicted_test_labels = generator(test_texts)
        predicted_test_labels = prune_prefixes_from_labels(predicted_test_labels)
        result=calculate_metrics(
            predictions=predicted_test_labels,
            labels=test_labels,
            average_type=average_type,
        )
        return predicted_test_labels

    def predict(self, text, model_path=None):
        if model_path is None:
            model_path = self.model_path
        model = BertForSequenceClassification.from_pretrained(model_path).to("cuda:0")
        generator = pipeline(
            task="text-classification", model=model, tokenizer=self.tokenizer, device=0
        )
        predicted_labels = generator(text)
        predicted_numeric_labels = prune_prefixes_from_labels(predicted_labels)
        result = map_result_to_text(predicted_numeric_labels, model_path)
        return result

    def evaluate_with_division_between_languages(self, df):
        pass
