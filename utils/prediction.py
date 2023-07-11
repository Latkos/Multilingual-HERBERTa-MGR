import pandas as pd

def add_predictions_and_correctness_label_to_dataframe(model, test_df):
    original_entities_df = test_df.reset_index()
    sentences = test_df["text"].tolist()
    prediction_results = model.predict(sentences)
    results_df = pd.DataFrame(prediction_results)
    results_df = results_df.reset_index()
    return original_entities_df, results_df
