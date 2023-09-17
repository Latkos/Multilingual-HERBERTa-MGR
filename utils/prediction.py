import pandas as pd


def add_predictions_and_correctness_label_to_dataframe(model, test_df):
    original_entities_df = test_df.reset_index()
    sentences = test_df["text"].tolist()
    prediction_results = model.predict(sentences)
    results_df = pd.DataFrame(prediction_results)
    results_df = results_df.reset_index()
    return original_entities_df, results_df


def predict_joint_models(test_df, ner_model, re_model,enhance_function):
    final_predictions = test_df
    sentences = test_df["text"].tolist()
    ner_predictions = ner_model.predict(sentences=sentences)
    text=enhance_function(ner_predictions)
    re_prediction_result = re_model.predict(sentences=text)
    final_predictions["predicted_entity_1"] = [d["entity_1"] for d in ner_predictions]
    final_predictions["predicted_entity_2"] = [d["entity_2"] for d in ner_predictions]
    final_predictions["predicted_relation"] = re_prediction_result
    return final_predictions
