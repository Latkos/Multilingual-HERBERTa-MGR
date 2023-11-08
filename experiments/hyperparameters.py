def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 12, 16, 20, 24]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.2),
    }

def optuna_hp_space_scientific(trial):
    return {
        "learning_rate": trial.suggest_categorical("learning_rate",[2e-5,3e-5,4e-5,5e-5]),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 6),
        "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.0,0.05,0.10, 0.15, 0.20]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0,0.001,0.01,0.1,0.2,0.3]),
    }


import pandas as pd
from transformers import TrainingArguments


def test_hyperparameters_impact(model, train_df, test_df):
    default_hyperparams = {
        "output_dir": "ner",
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 6,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "num_train_epochs": 6,
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "load_best_model_at_end": True,
        "metric_for_best_model": "overall_f1",
        "report_to": "wandb"
    }
    hyperparameter_grid = {
        "learning_rate": [2e-5, 3e-5, 4e-5, 5e-5],
        "num_train_epochs": [1, 2, 3, 4, 5, 6],
        "warmup_ratio": [0.0, 0.05, 0.10, 0.15, 0.20],
        "weight_decay": [0.0, 0.001, 0.01, 0.1, 0.2, 0.3]
    }
    results=[]
    for hyperparam, values in hyperparameter_grid.items():
        for value in values:
            hyperparams = default_hyperparams.copy()
            hyperparams[hyperparam] = value
            training_arguments = TrainingArguments(
                **hyperparams
            )
            model.train(train_df=train_df,training_arguments=training_arguments)
            eval_result = model.evaluate(df=test_df)
            results = results.append({
                "Hyperparameter": hyperparam,
                "Value": value,
                "Evaluation Result": eval_result
            })
    print(results)
    results_df=pd.DataFrame(results)
    return results_df
