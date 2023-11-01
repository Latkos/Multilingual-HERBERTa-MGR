import pandas as pd
import torch

from utils.enhancement import enhance_with_brackets
from utils.preprocessing import filter_out_wrong_data


def filter_and_set_proportions(df, proportions):
    target_languages = list(proportions.keys())
    df_filtered = df[df['lang'].isin(target_languages)].copy()
    current_proportions = df_filtered['lang'].value_counts(normalize=True).to_dict()

    for language in target_languages:
        if language not in current_proportions:
            current_proportions[language] = 0.0
        target_proportion = proportions[language]

        if current_proportions[language] > target_proportion:
            num_samples_to_keep = int(target_proportion * len(df_filtered))
            df_filtered = df_filtered[df_filtered['lang'] != language]
            df_filtered = pd.concat([
                df_filtered[df_filtered['lang'] == language].sample(n=num_samples_to_keep, replace=False),
                df_filtered[df_filtered['lang'] != language]
            ])

    return df_filtered


def train_and_evaluate_on_language_subsets(training_files, testing_files, ner_model, re_model,
                                           enhance_function=enhance_with_brackets, average_type='micro',
                                           skip_training=False):
    train_dfs = [pd.read_csv(file, sep='\t') for file in training_files]
    test_dfs = [pd.read_csv(file, sep='\t') for file in testing_files]
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    train_df = filter_out_wrong_data(train_df)
    test_df = filter_out_wrong_data(test_df)
    if len(train_df) >= 675000:
        train_df = train_df.sample(n=675000)
    language_names = '+'.join([file.split('_')[0] for file in training_files])
    language_names=language_names.replace("data/","")
    print(f"Language names: {language_names}")
    if not skip_training:
        ner_model.train(train_df, model_path=f"models/ner_{language_names}")
    ner_model.evaluate(df=test_df, model_path=f"models/ner_{language_names}")
    results = ner_model.predict(sentences=train_df['text'].tolist(), model_path=f"models/ner_{language_names}")
    enhanced_input = enhance_function(results)
    train_df['text'] = enhanced_input
    if not skip_training:
        re_model.train(train_df, model_path=f"models/re_{language_names}", remove_tags=False)
    test_df = test_df[test_df['label'].isin(train_df['label'].unique())]
    re_model.evaluate(df=test_df, model_path=f"models/re_{language_names}", average_type=average_type)
    torch.cuda.empty_cache()
