
def evaluate_with_division_between_column(model, test_df, column_name, model_path=None, average_type="micro"):
    if model_path is None:
        model_path = model.model_path
    print(f"COMMENCING EVALUATION DIVIDED BY {column_name} ")
    print(test_df[column_name].unique())
    evaluation_results = {}
    for unique_value in test_df[column_name].unique():
        print(f"{column_name}: {unique_value}")
        lang_df = test_df[test_df[column_name] == unique_value]
        evaluation_results[unique_value] = model.evaluate(
            test_df=lang_df, model_path=model_path, average_type=average_type
        )
    return evaluation_results

def get_f1_from_metrics(metrics):
    f1 = metrics["eval_overall_f1"]
    print(f"f1: {f1}")
    return f1