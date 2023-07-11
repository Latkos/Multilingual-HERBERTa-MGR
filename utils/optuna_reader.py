import re
import pandas as pd


def read_optuna_logs(file_name):
    trial_numbers = []
    metrics = []
    trial_runtimes = []
    with open(file_name, encoding="utf-8") as file:
        for line in file:
            trial_number = re.search("Trial (.*) finished", line)
            if trial_number:
                trial_numbers.append(trial_number.group(1))
            metric = re.search("finished with value: (.*) and parameters", line)
            if metric:
                metrics.append(metric.group(1))
            trial_runtime = re.search("'train_runtime': (.*), 'train_samples_per_second':", line)
            if trial_runtime:
                trial_runtimes.append(trial_runtime.group(1))
    df = pd.DataFrame(
        data={
            "trial_number": trial_numbers,
            "metric": metrics,
            "trial_runtime": trial_runtimes,
        }
    )
    df["metric"] = df["metric"].astype(float)
    df["trial_runtime"] = df["trial_runtime"].astype(float)
    return df
