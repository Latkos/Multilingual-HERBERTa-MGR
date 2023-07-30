import pandas as pd


# def add_predictions_and_correctness_label_to_dataframe(model, test_df):
#     original_entities_df = test_df.reset_index()
#     sentences = test_df["text"].tolist()
#     prediction_results = model.predict(sentences)
#     results_df = pd.DataFrame(prediction_results)
#     results_df = results_df.reset_index()
#     return original_entities_df, results_df


def evalute_predictions_of_combined_models(test_df, ner_model, re_model):
    final_predictions = test_df
    sentences = test_df["text"].tolist()
    ner_predictions = ner_model.predict(sentences=sentences)
    text = []
    for prediction in ner_predictions:
        text.append(prediction["text"])
    print("NER evaluation results:")
    print(ner_model.evaluate(test_df))
    re_prediction_result = re_model.predict(sentences=text)
    print(re_prediction_result)
    final_predictions["predicted_entity_1"] = [d["entity_1"] for d in ner_predictions]
    final_predictions["predicted_entity_2"] = [d["entity_1"] for d in ner_predictions]
    final_predictions["predicted_relation"] = re_prediction_result
    print("NER evaluation results:")
    print(re_model.evaluate(test_df))
    print(final_predictions)
    return final_predictions
