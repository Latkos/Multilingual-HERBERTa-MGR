import os
import shutil

import pandas as pd
import torch


def test_ner_quality_depending_on_dataset_size(
    model, train_df, test_df, sizes, random_state=None, delete_after_training=True
):
    results = []
    for size in sizes:
        print(f"Testing NER, size: {size}")
        if size > len(train_df):
            print(f"Not enough training examples to train the model for size {size}")
            continue
        sampled_train_df = train_df.sample(n=size, random_state=random_state)
        model_path = f"models/ner_size_quality_tests_{size}"
        model.train(train_df=sampled_train_df, model_path=model_path)
        torch.cuda.empty_cache()
        evaluation_result = model.evaluate(test_df, model_path)
        f1 = evaluation_result["eval_overall_f1"]
        results.append({"size": size, "f1": f1})
        if delete_after_training:
            shutil.rmtree(model_path)
    results_df = pd.DataFrame(results)
    return results_df


def test_re_quality_depending_on_dataset_size(
    model, train_df, test_df, sizes, random_state=None, delete_after_training=True, remove_tags=False
):
    results = []
    for size in sizes:
        print(f"Testing RE, size: {size}")
        if size > len(train_df):
            print(f"Not enough training examples to train the model for size {size}")
            continue
        sampled_train_df = train_df.sample(n=size, random_state=random_state)
        model_path = f"models/re_size_quality_tests_{size}"
        model.train(train_df=sampled_train_df, model_path=model_path, remove_tags=remove_tags)
        torch.cuda.empty_cache()
        test_df_filtered = test_df.copy()
        test_df_filtered = test_df_filtered[test_df_filtered["label"].isin(sampled_train_df["label"].unique())]
        evaluation_result = model.evaluate(test_df_filtered, model_path)
        f1 = evaluation_result["f1"]
        results.append({"size": size, "f1": f1})
        if delete_after_training:
            shutil.rmtree(model_path)

    results_df = pd.DataFrame(results)
    return results_df
