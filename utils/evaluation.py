import pandas as pd

def evaluate_with_division_between_column(model, test_df, column_name, average_type="micro"):
    print(f"Evaluating with division between {column_name} ")
    evaluation_results = {}
    for unique_value in test_df[column_name].unique():
        print(f"{column_name}: {unique_value}")
        subset_df = test_df[test_df[column_name] == unique_value]
        evaluation_results[unique_value] = model.evaluate(
            test_df=subset_df, average_type=average_type
        )
    df = pd.DataFrame(list(evaluation_results.items()), columns=['relation', 'f1'])
    return df

def get_f1_from_metrics(metrics):
    f1 = metrics.get("eval_overall_f1")
    if f1 is None:
        f1 = metrics["eval_f1"]
    return f1
