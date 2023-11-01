import pandas as pd


def test_model_quality_depending_on_dataset_size(model, train_df, test_df, sizes, random_state=None):
    results = []
    for size in sizes:
        print(f"Testing {model.__class__.__name__.lower()}, size: {size}")
        sampled_train_df = train_df.sample(n=size, random_state=random_state)
        model_path = f"../models/{model.__class__.__name__.lower()}_size_quality_tests_{size}"
        model.train(train_df=sampled_train_df, model_path=model_path)
        test_df_filtered = test_df.copy()
        test_df_filtered = test_df_filtered[test_df_filtered['label'].isin(sampled_train_df['label'].unique())]
        evaluation_result = model.evaluate(test_df_filtered, model_path)
        f1_column = evaluation_result.get("eval_overall_f1", evaluation_result.get("f1"))
        results.append({"size": size, "f1": evaluation_result[f1_column]})
    results_df = pd.DataFrame(results)
    return results_df
