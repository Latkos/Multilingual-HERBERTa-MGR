import sys

import nltk
import numpy as np
from datasets import load_metric, Dataset
from transformers import DataCollatorForTokenClassification, AutoTokenizer, TrainingArguments, \
    AutoModelForTokenClassification, Trainer

from ner.training import compute_metrics
from new_named_entity import ner_config
from new_named_entity.named_entity_utility_functions import split_dataset, create_dataset_from_dataframe


class NamedEntityModel:
    def __init__(self, model_path="./", model_name="ner",model_type='bert-base-multilingual-cased'):
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)


    def train(self, train_df, training_arguments=None, split=0.2):
        ds=create_dataset_from_dataframe(train_df)
        ds = ds.map(self.tokenize_adjust_labels, batched=True)
        ds=ds.remove_columns(column_names=['id','entity_1','entity_2','label','text','lang'])
        train_ds,val_ds=split_dataset(ds,split)
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased",
                                                                num_labels=len(ner_config.label_mapping)).to("cuda:0")
        training_args = TrainingArguments(
            output_dir="./ner/result1",
            evaluation_strategy="steps",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=7,
            weight_decay=0.01,
            logging_steps=1000,
            run_name="first_run",
            save_strategy='no'
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )
        trainer.train()
        trainer.save_model(self.model_name)

    def evaluate_model(self,test_df, model_path=None):
        pass

    def predict(self, text, model_path=None):
        pass

    def tokenize_adjust_labels(self, all_samples_per_split):
        tokenized_samples = self.tokenizer.batch_encode_plus(all_samples_per_split["tokens"], is_split_into_words=True)
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
                    adjusted_label_ids.append(existing_label_ids[i])
            total_adjusted_labels.append(adjusted_label_ids)
        tokenized_samples["labels"] = total_adjusted_labels
        return tokenized_samples

