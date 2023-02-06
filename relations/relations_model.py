from tokenizers.trainers import Trainer
import pandas as pd

from relations.relations_dataset import RelationsDataset
from relations.relations_utility_functions import prune_prefixes_from_labels, map_result_to_text, \
    calculate_metrics, get_texts_and_labels
from sklearn.model_selection import train_test_split
from transformers import (
    Trainer,
    BertForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)

from old_relations.helper_functions_relations import (
    get_texts_and_labels,
    prune_prefixes_from_labels,
    calculate_metrics,
    map_result_to_text,
)

from old_relations.relations_dataset import RelationsDataset



class RelationsModel:
    def __init__(self, model_path, model_name="bert-base-multilingual-cased"):
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def train(self, train_df, model_path, training_arguments=None, split=0.2):
        if training_arguments is None:
            training_arguments = {}
        df = pd.read_csv(train_df, sep="\t")
        train_texts, train_labels = get_texts_and_labels(df, model_path)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=split
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)
        train_dataset = RelationsDataset(train_encodings, train_labels)
        val_dataset = RelationsDataset(val_encodings, val_labels)
        not_none_params = {k: v for k, v in training_arguments.items() if v is not None}
        training_args = TrainingArguments(**not_none_params)
        labels_number = len(set(train_labels))
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased", num_labels=labels_number
        ).to("cuda:0")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        trainer.train()
        trainer.save_model(model_path)
        return model

    def evaluate_model(self,test_df, model_path=None):
        if model_path is None:
            model_path=self.model_path
        test_texts, test_labels = get_texts_and_labels(test_df, model_path)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        labels_number = len(set(test_labels))
        model = BertForSequenceClassification.from_pretrained(
            model_path, num_labels=labels_number
        ).to("cuda:0")
        generator = pipeline(
            task="text-classification", model=model, tokenizer=tokenizer, device=0
        )
        predicted_test_labels = generator(test_texts)
        predicted_test_labels = prune_prefixes_from_labels(predicted_test_labels)
        calculate_metrics(test_labels, predicted_test_labels)
        return predicted_test_labels


    def predict(self, text, model_path=None):
        if model_path is  None:
            model_path=self.model_path
        model=BertForSequenceClassification.from_pretrained(model_path).to("cuda:0")
        generator = pipeline(
            task="text-classification", model=model, tokenizer=self.tokenizer, device=0
        )
        predicted_labels = generator(text)
        predicted_numeric_labels = prune_prefixes_from_labels(predicted_labels)
        result = map_result_to_text(predicted_numeric_labels, model_path)
        return result

