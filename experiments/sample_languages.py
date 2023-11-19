import pandas as pd
import torch

from named_entity.named_entity_model import NamedEntityModel
from relations.relations_model import RelationsModel
from utils.enhancement import enhance_with_brackets
from utils.overlap import remove_overlapping_entities
from utils.preprocessing import filter_out_wrong_data


# def filter_and_set_proportions(df, proportions):
#     target_languages = list(proportions.keys())
#     df_filtered = df[df['lang'].isin(target_languages)].copy()
#     current_proportions = df_filtered['lang'].value_counts(normalize=True).to_dict()
#
#     for language in target_languages:
#         if language not in current_proportions:
#             current_proportions[language] = 0.0
#         target_proportion = proportions[language]
#
#         if current_proportions[language] > target_proportion:
#             num_samples_to_keep = int(target_proportion * len(df_filtered))
#             df_filtered = df_filtered[df_filtered['lang'] != language]
#             df_filtered = pd.concat([
#                 df_filtered[df_filtered['lang'] == language].sample(n=num_samples_to_keep, replace=False),
#                 df_filtered[df_filtered['lang'] != language]
#             ])
#
#     return df_filtered


def perform_four_variations_linguistic(
    training_languages,
    testing_languages,
    ner_model,
    re_model,
    enhance_function=enhance_with_brackets,
    downsample_number=None,
    average_type="micro",
):
    results = []
    # monolingual
    train_df, test_df = create_datasets_from_language_lists(
        testing_languages, testing_languages, downsample_number=None
    )
    results.append(
        train_and_evaluate_on_language_subsets(
            train_df,
            test_df,
            ner_model,
            re_model,
            enhance_function=enhance_function,
            average_type=average_type,
            skip_ner_training=False,
        )
    )
    # zero-shot
    train_df, test_df = create_datasets_from_language_lists(
        training_languages, testing_languages, downsample_number=downsample_number
    )
    results.append(
        train_and_evaluate_on_language_subsets(
            train_df,
            test_df,
            ner_model,
            re_model,
            enhance_function=enhance_function,
            average_type=average_type,
            skip_ner_training=False,
        )
    )
    # transfer
    train_df, test_df = create_datasets_from_language_lists(
        training_languages + testing_languages,
        testing_languages,
        downsample_number=downsample_number,
    )
    results.append(
        train_and_evaluate_on_language_subsets(
            train_df,
            test_df,
            ner_model,
            re_model,
            enhance_function=enhance_function,
            average_type=average_type,
            skip_ner_training=False,
        )
    )
    # multilingual
    multilingual_ner = NamedEntityModel(model_path="models/base_ner")
    multilingual_re = RelationsModel(model_path="models/re_brackets")
    train_df, test_df = create_datasets_from_language_lists(
        training_languages, testing_languages, downsample_number=None
    )
    multilingual_ner_result = multilingual_ner.evaluate(model_path="models/base_ner", df=test_df)
    multilingual_re_result = multilingual_re.evaluate(model_path="models/re_brackets", df=test_df)
    multilingual_results = {
        "training_languages": "all",
        "testing_languages": testing_languages,
        "ner_result": multilingual_ner_result,
        "re_result": multilingual_re_result,
    }
    results.append(multilingual_results)
    print(f"RESULTS: {results}")
    print("**************")
    return results


def create_datasets_from_language_lists(training_languages, testing_languages, downsample_number=None):
    training_files = [f"data/{lang}_corpora_train.tsv" for lang in training_languages]
    testing_files = [f"data/{lang}_corpora_test.tsv" for lang in testing_languages]
    train_dfs = [pd.read_csv(file, sep="\t") for file in training_files]
    test_dfs = [pd.read_csv(file, sep="\t") for file in testing_files]
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    if downsample_number is not None:
        to_downsample = set(train_df["lang"].unique()) - set(test_df["lang"].unique())
        downsampled = train_df[train_df["lang"].isin(to_downsample)].sample(
            n=min(len(train_df), downsample_number), random_state=42
        )
        train_df = pd.concat([downsampled, train_df[~train_df["lang"].isin(to_downsample)]], ignore_index=True)
    train_df = filter_out_wrong_data(train_df)
    test_df = filter_out_wrong_data(test_df)
    return train_df, test_df


def train_and_evaluate_on_language_subsets(
    train_df,
    test_df,
    ner_model,
    re_model,
    enhance_function=enhance_with_brackets,
    average_type="micro",
    skip_ner_training=False,
):
    training_language_names = "+".join(train_df["lang"].unique())
    testing_language_names = "+".join(test_df["lang"].unique())
    print(f"Training names: {training_language_names}")
    print(f"Testing names: {testing_language_names}")
    if not skip_ner_training:
        ner_model.train(train_df, model_path=f"models/ner_{training_language_names}")
    ner_result = ner_model.evaluate(df=test_df, model_path=f"models/ner_{training_language_names}")
    results = ner_model.predict(sentences=train_df["text"].tolist(), model_path=f"models/ner_{training_language_names}")
    enhanced_input = enhance_function(results)
    train_df["text"] = enhanced_input
    re_model.train(train_df, model_path=f"models/re_{training_language_names}", remove_tags=False)
    test_df = test_df[test_df["label"].isin(train_df["label"].unique())]
    re_result = re_model.evaluate(
        df=test_df, model_path=f"models/re_{training_language_names}", average_type=average_type
    )
    torch.cuda.empty_cache()
    results = {
        "training_languages": training_language_names,
        "testing_languages": testing_language_names,
        "ner_result": ner_result,
        "re_result": re_result,
    }
    return results


def run_experiments_linguistic():
    # Same language family
    ner_model = NamedEntityModel()
    re_model = RelationsModel()
    perform_four_variations_linguistic(
        ["es", "it", "pt"], ["fr"], ner_model, re_model, enhance_function=enhance_with_brackets, average_type="micro"
    )
    perform_four_variations_linguistic(
        ["de", "en-full", "sv"],
        ["fr"],
        ner_model,
        re_model,
        enhance_function=enhance_with_brackets,
        average_type="micro",
        downsample_number=128373,
    )

    # Cross-script
    ner_model = NamedEntityModel()
    re_model = RelationsModel()
    perform_four_variations_linguistic(
        ["ar"], ["ru"], ner_model, re_model, enhance_function=enhance_with_brackets, average_type="micro"
    )
    perform_four_variations_linguistic(
        ["pl"],
        ["ru"],
        ner_model,
        re_model,
        enhance_function=enhance_with_brackets,
        average_type="micro",
        downsample_number=9304,
    )

    # Complex language and simple language
    perform_four_variations_linguistic(
        ["pl"],
        ["es"],
        ner_model,
        re_model,
        enhance_function=enhance_with_brackets,
        average_type="micro",
        downsample_number=4483,
    )
    perform_four_variations_linguistic(
        ["sv"],
        ["es"],
        ner_model,
        re_model,
        enhance_function=enhance_with_brackets,
        average_type="micro",
    )

    # SVO
    perform_four_variations_linguistic(
        ["ko"], ["fa"], ner_model, re_model, enhance_function=enhance_with_brackets, average_type="micro"
    )
    perform_four_variations_linguistic(
        ["pt"],
        ["fa"],
        ner_model,
        re_model,
        enhance_function=enhance_with_brackets,
        average_type="micro",
        downsample_number=18712,
    )

    # Lexical Overlap
    perform_four_variations_linguistic(
        ["fr"], ["it"], ner_model, re_model, enhance_function=enhance_with_brackets, average_type="micro"
    )
    # We do not call perform_four_variations_linguistic here, because we need to filter out overlapping entities
    # And because we are only interested in zero-shot scenario, few-shot does not make much sense here and the rest will not change

    train_df=pd.read_csv('../data/fr_corpora_train.tsv',sep='\t')
    test_df=pd.read_csv('../data/it_corpora_test.tsv',sep='\t')
    train_df=remove_overlapping_entities(train_df,test_df)
    train_and_evaluate_on_language_subsets(
        train_df,
        test_df,
        ner_model,
        re_model,
        enhance_function=enhance_with_brackets,
        average_type='micro',
    )



