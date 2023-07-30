def optuna_hp_space(trial):
    return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 12, 16, 20, 24]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.2),
    }

def test_hyperparameters(optuna_hp_space=optuna_hp_space):

    def perform_hyperparameter_search(
        self,
        train_df,
        space,
        config_path="./config/base_config.yaml",
        training_arguments=None,
        number_of_trials=50,
        split=0.2,
    ):
