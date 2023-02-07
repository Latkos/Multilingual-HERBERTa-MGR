import sys

import nltk
import numpy as np
from datasets import load_metric, Dataset
from transformers import DataCollatorForTokenClassification, AutoTokenizer, TrainingArguments, \
    AutoModelForTokenClassification, Trainer


class NamedEntityModel:
    def __init__(self, model_path="./", model_name="ner",model_type='bert-base-multilingual-cased',
                 label_mapping=None, metric=None):
        if label_mapping is None:
            label_mapping = {"O": 0, 'B-Entity1': 1, 'I-Entity1': 2, 'B-Entity2': 3, 'I-Entity2': 4}
        self.label_mapping=label_mapping
        self.label_names=list(label_mapping.keys())
        if metric is None:
            metric = load_metric("seqeval")
        self.metric = metric
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)

    def create_tokens_and_labels(self, text):
        replacement_map = {"<e1>": " StartEntity1 ", "</e1>": " StopEntity1 ", "<e2>": " StartEntity2 ",
                           "</e2>": " StopEntity2 "}
        for word, replacement in replacement_map.items():
            text = text.replace(word, replacement)
        text_tokens = nltk.word_tokenize(text)
        # TODO unhardcode this
        special_elements = ['StartEntity1','StopEntity1','StartEntity2','StopEntity2']
        tokens, labels = self.extract_entities(text_tokens, special_elements)
        return tokens, labels

    def extract_entities(self, tokens, special_elements):
        new_tokens = [token for token in tokens if token not in special_elements]
        labels = ['O'] * len(new_tokens)
        entity_starts = [i for i, token in enumerate(tokens) if "StartEntity" in token]
        entity_ends = [i for i, token in enumerate(tokens) if "StopEntity" in token]
        for start, end in zip(entity_starts, entity_ends):
            entity_label = tokens[start].split("Start")[1]
            try:
                start = new_tokens.index(tokens[start + 1])
                end = new_tokens.index(tokens[end - 1]) + 1
            except Exception as e:
                print("CAUGHT AN ERROR")
                print(str(e))
                print("TOKENS:",tokens)
                sys.exit(1)
            labels[start] = "B-" + entity_label
            for i in range(start + 1, end):
                labels[i] = "I-" + entity_label
        return new_tokens, labels

    def preprocess_dataframe(self,df):
        df['tokens'], df['labels'] = zip(*df['text'].map(self.create_tokens_and_labels))
        df['ner_tags'] = df['labels'].apply(lambda x: [self.label_mapping[i] for i in x])
        return df

    def compute_metrics(self,p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_names[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        flattened_results = {
            "overall_precision": results["overall_precision"],
            "overall_recall": results["overall_recall"],
            "overall_f1": results["overall_f1"],
            "overall_accuracy": results["overall_accuracy"],
        }
        for k in results.keys():
            if (k not in flattened_results.keys()):
                flattened_results[k + "_f1"] = results[k]["f1"]
        return flattened_results

    def split_dataset(self, ds, split=0.2):
        split_ds = ds.train_test_split(split)
        train = split_ds["train"]
        val = split_ds["test"]
        return train, val

    def train(self, train_df, training_arguments=None, split=0.2):
        ds=self.create_dataset_from_dataframe(train_df)
        ds = ds.map(self.tokenize_adjust_labels, batched=True)
        ds=ds.remove_columns(column_names=['id','entity_1','entity_2','label','text','lang'])
        print(ds['tokens'][0])
        print(ds['tokens'][1])
        train_ds,val_ds=self.split_dataset(ds,split)
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased",
                                                                num_labels=len(self.label_mapping))
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
            compute_metrics=self.compute_metrics
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
        print(len(tokenized_samples["input_ids"]))
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

    def create_dataset_from_dataframe(self,df):
        df=self.preprocess_dataframe(df)
        dataset=Dataset.from_pandas(df)
        return dataset