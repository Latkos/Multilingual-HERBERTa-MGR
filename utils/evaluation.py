import pandas as pd
from sklearn.metrics import f1_score

from utils.prediction import predict_joint_models


def evaluate_with_division_between_column(model, test_df, column_name, average_type="micro"):
    print(f"Evaluating with division between {column_name} ")
    evaluation_results = {}
    for unique_value in test_df[column_name].unique():
        print(f"{column_name}: {unique_value}")
        subset_df = test_df[test_df[column_name] == unique_value]
        evaluation_results[unique_value] = model.evaluate(
            df=subset_df, average_type=average_type
        )
    df = pd.DataFrame(list(evaluation_results.items()), columns=['relation', 'f1'])
    return df

def get_f1_from_metrics(metrics):
    f1 = metrics.get("eval_overall_f1")
    if f1 is None:
        f1 = metrics["eval_f1"]
    return f1

def evaluate_joint_models(file_path, ner_model, re_model, enhance_function):
    test_df = pd.read_csv(file_path, sep='\t')
    predictions_df = predict_joint_models(test_df, ner_model, re_model, enhance_function)
    true_values = list(zip(test_df['entity_1'], test_df['entity_2'], test_df['label']))
    predicted_values = list(zip(predictions_df['predicted_entity_1'], predictions_df['predicted_entity_2'],
                                predictions_df['predicted_relation']))
    binary_true = [1 if true == pred else 0 for true, pred in zip(true_values, predicted_values)]
    binary_pred = [1] * len(binary_true)
    combined_f1 = f1_score(binary_true, binary_pred)
    return combined_f1
