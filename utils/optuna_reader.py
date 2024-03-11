import ast
import re
import pandas as pd


def read_optuna_logs(file_name):
    trial_numbers = []
    metrics = []
    trial_runtimes = []
    parameters=[]
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
            trial_parameters_match = re.search(r"parameters: ({.*})", line)
            if trial_parameters_match is not None:
                trial_parameters_str = re.search(r"parameters: ({.*})", line).group(1)
                trial_parameters = ast.literal_eval(trial_parameters_str)
                parameters.append(trial_parameters)
    data = {
        "trial_number": trial_numbers,
        "metric": metrics,
        "trial_runtime": trial_runtimes,
    }
    combined_data = {**data, **{key: [d[key] for d in parameters] for key in parameters[0]}}
    df = pd.DataFrame(combined_data)
    df["metric"] = df["metric"].astype(float)
    df["trial_runtime"] = df["trial_runtime"].astype(float)
    df["trial_number"] = df["trial_number"].astype(int)
    df['trial_number'] -= df['trial_number'].min()
    return df
