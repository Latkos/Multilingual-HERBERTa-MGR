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
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [1]),
        "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.0,0.05,0.10, 0.15, 0.20]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0,0.001,0.01,0.1,0.2,0.3]),
    }


