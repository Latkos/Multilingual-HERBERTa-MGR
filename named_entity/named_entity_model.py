from transformers import (
    DataCollatorForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForTokenClassification,
    Trainer,
    pipeline,
    BertForTokenClassification,
)

from base_model.base_model import BaseModel
from named_entity import ner_config
from named_entity.named_entity_utility_functions import (
    split_dataset,
    create_dataset_from_dataframe,
    get_model_output_as_sentence,
    compute_metrics,
)
from utils.config_parser import get_training_args


class NamedEntityModel(BaseModel):
    def __init__(self, model_path="./ner", model_type="bert-base-multilingual-cased"):
        self.model_path = model_path
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)

    def preprocess_data(self, df):
        ds = create_dataset_from_dataframe(df)
        ds = ds.map(self.tokenize_adjust_labels, batched=True)
        ds = ds.remove_columns(
            column_names=["id", "entity_1", "entity_2", "label", "text", "lang"]
        )
        return ds

    def train(self, train_df, training_arguments=None, split=0.2, config_path='./config/base_config.yaml'):
        if training_arguments is None:
            training_arguments=get_training_args(config_path=config_path,model_type="ner")
        ds = self.preprocess_data(train_df)
        train_ds, val_ds = split_dataset(ds, split)
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_type, num_labels=len(ner_config.label_mapping)
        ).to("cuda:0")
        not_none_params = {k: v for k, v in training_arguments.items() if v is not None}
        training_args=TrainingArguments(**not_none_params)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model(self.model_path)

    def evaluate(self, test_df, model_path=None):
        if model_path is None:
            model_path = self.model_path
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        test_args = TrainingArguments(
            output_dir="./ner/result1",
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=16,
        )
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        trainer = Trainer(model=model, args=test_args,
                          data_collator=data_collator,
                          compute_metrics=compute_metrics)
        test_ds = self.preprocess_data(test_df)
        result = trainer.evaluate(test_ds)
        print("EVALUATION RESULT: ", result)
        return result

    def predict(self, sentences, model_path):
        model = BertForTokenClassification.from_pretrained(model_path)
        tokenizer = self.tokenizer
        token_classifier = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
        groups = token_classifier(sentences)
        result = []
        if isinstance(groups[0], dict):
            result.append(get_model_output_as_sentence(groups))
        else:
            for i in groups:
                result.append(get_model_output_as_sentence(i))
        return result

    def tokenize_adjust_labels(self, all_samples_per_split):
        tokenized_samples = self.tokenizer.batch_encode_plus(
            all_samples_per_split["tokens"], is_split_into_words=True
        )
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
