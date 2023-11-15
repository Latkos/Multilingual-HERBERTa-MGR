import pickle

import pandas as pd
import torch
from relations.relations_utility_functions import remove_tags_from_dataframe

def add_predictions_and_correctness_label_to_dataframe(model, test_df):
    original_entities_df = test_df.reset_index()
    sentences = test_df["text"].tolist()
    prediction_results = model.predict(sentences)
    results_df = pd.DataFrame(prediction_results)
    results_df = results_df.reset_index()
    return original_entities_df, results_df


def predict_joint_models(test_df, ner_model, re_model, enhance_function):
    final_predictions = test_df.copy()
    sentences = test_df["text"].tolist()
    ner_predictions = ner_model.predict(sentences=sentences)
    text = enhance_function(ner_predictions)
    re_prediction_result = re_model.predict(sentences=text)
    final_predictions["predicted_entity_1"] = [d["entity_1"] for d in ner_predictions]
    final_predictions["predicted_entity_2"] = [d["entity_2"] for d in ner_predictions]
    final_predictions["predicted_relation"] = re_prediction_result
    return final_predictions


def save_results(filename, data):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)


def train_re_on_ner(ner_model, re_model, train_df, test_df, enhancement_func, results_file, read, train_ner=False,
                    model_path=None):
    if read:
        with open(results_file, 'rb') as file:
            results = pickle.load(file)
    else:
        if train_ner:
            ner_model.train(train_df)
        results = ner_model.predict(sentences=train_df['text'].tolist())
        save_results(results_file, results)
    enhanced_train_df = train_df.copy()
    enhanced_input = enhancement_func(results)
    enhanced_train_df['text'] = enhanced_input
    print(train_df['text'].head(5))
    print("*************************")
    print(enhanced_train_df['text'].head(5))
    print("*************************")
    re_model.train(train_df=enhanced_train_df, model_path=model_path, remove_tags=False)
    results = re_model.evaluate(df=test_df, model_path=model_path, enhancement_func=enhancement_func)
    torch.cuda.empty_cache()
    return results
