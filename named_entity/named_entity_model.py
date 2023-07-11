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
from config import general_config
from named_entity.named_entity_utility_functions import (
    split_dataset,
    create_dataset_from_dataframe,
    get_model_output_as_sentence,
    compute_metrics,
    tokenize_adjust_labels,
)
from utils.config_parser import get_training_args
from utils.evaluation import get_f1_from_metrics


class NamedEntityModel(BaseModel):
    def __init__(self, model_path="./ner", model_type="bert-base-multilingual-cased"):
        super().__init__(model_path, model_type)
        self.model_path = model_path
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)

    def train(
        self,
        train_df,
        training_arguments=None,
        split=0.2,
        config_path="./config/base_config.yaml",
    ):
        trainer = self.create_trainer(train_df=train_df, training_arguments=training_arguments, split=split, config_path=config_path)
        trainer.train()
        trainer.save_model(self.model_path)

    def create_trainer(
        self,
        train_df,
        training_arguments=None,
        config_path="./config/base_config.yaml",
        split=0.2,
        model_init=None,
    ):
        if training_arguments is None:
            training_arguments = get_training_args(config_path=config_path, model_type="ner")
        ds = self.preprocess_data(train_df)
        train_ds, val_ds = split_dataset(ds, split)
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_type, num_labels=len(general_config.label_mapping)
        ).to("cuda:0")
        not_none_params = {k: v for k, v in training_arguments.items() if v is not None}
        training_args = TrainingArguments(**not_none_params)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            model_init=model_init,
        )
        return trainer

    def preprocess_data(self, df):
        ds = create_dataset_from_dataframe(df)
        ds = ds.map(tokenize_adjust_labels(self.tokenizer), batched=True)
        ds = ds.remove_columns(column_names=["id", "entity_1", "entity_2", "label", "text", "lang"])
        return ds

    def evaluate(self, df, model_path=None):
        if model_path is None:
            model_path = self.model_path
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        test_args = TrainingArguments(
            output_dir=".placeholder",
            do_train=False,
            do_predict=True,
        )
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        trainer = Trainer(
            model=model,
            args=test_args,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        ds = self.preprocess_data(df)
        result = trainer.evaluate(ds)
        print("EVALUATION RESULT: ", result)
        return result

    def predict(self, sentences, model_path=None):
        if model_path is None:
            model_path = self.model_path
        model = BertForTokenClassification.from_pretrained(model_path)
        tokenizer = self.tokenizer
        token_classifier = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=0,
        )
        groups = token_classifier(sentences)
        result = []
        if isinstance(groups[0], dict):
            result.append(get_model_output_as_sentence(groups))
        else:
            for i in groups:
                result.append(get_model_output_as_sentence(i))
        return result

    def perform_hyperparameter_search(
        self,
        train_df,
        space,
        config_path="./config/base_config.yaml",
        training_arguments=None,
        number_of_trials=50,
        split=0.2,
    ):
        trainer = self.create_trainer(
            train_df=train_df,
            training_arguments=training_arguments,
            split=split,
            config_path=config_path,
            model_init=self.model_init,
        )
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=space,
            n_trials=number_of_trials,
            compute_objective=get_f1_from_metrics,
        )
        return best_trial

    def model_init(self, trial):
        return AutoModelForTokenClassification.from_pretrained(
            self.model_type, num_labels=len(general_config.label_mapping)
        ).to("cuda:0")
