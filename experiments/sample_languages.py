import pandas as pd
import torch

from named_entity.named_entity_model import NamedEntityModel
from relations.relations_model import RelationsModel
from utils.enhancement import enhance_with_special_characters
from utils.overlap import remove_overlapping_entities
from utils.preprocessing import filter_out_wrong_data




def perform_four_variations_linguistic(
    training_languages,
    testing_languages,
    ner_model,
    re_model,
    enhance_function=enhance_with_special_characters,
    downsample_number=None,
    downsample_main=None,
    average_type="micro",
    train_monolingual=True,
    number_of_runs=1
):
    torch.cuda.empty_cache()
    final_results = {
        "monolingual": {"ner_result": 0, "re_result": 0},
        "zero_shot": {"ner_result": 0, "re_result": 0},
        "transfer": {"ner_result": 0, "re_result": 0},
        "multilingual": {"ner_result": 0, "re_result": 0}
    }

    for _ in range(number_of_runs):
        # Monolingual
        if train_monolingual:
            train_df, test_df = create_datasets_from_language_lists(
                testing_languages, testing_languages, downsample_number=None, downsample_main=downsample_main
            )
            monolingual_result = train_and_evaluate_on_language_subsets(
                train_df, test_df, ner_model, re_model, enhance_function, average_type, False
            )
            final_results["monolingual"]["ner_result"] += monolingual_result["ner_result"]
            final_results["monolingual"]["re_result"] += monolingual_result["re_result"]
        torch.cuda.empty_cache()

        # Zero-shot
        train_df, test_df = create_datasets_from_language_lists(
            training_languages, testing_languages, downsample_number, downsample_main
        )
        zero_shot_result = train_and_evaluate_on_language_subsets(
            train_df, test_df, ner_model, re_model, enhance_function, average_type, False
        )
        final_results["zero_shot"]["ner_result"] += zero_shot_result["ner_result"]
        final_results["zero_shot"]["re_result"] += zero_shot_result["re_result"]
        torch.cuda.empty_cache()

        # Transfer
        train_df, test_df = create_datasets_from_language_lists(
            training_languages + testing_languages, testing_languages, downsample_number, downsample_main
        )
        transfer_result = train_and_evaluate_on_language_subsets(
            train_df, test_df, ner_model, re_model, enhance_function, average_type, False
        )
        final_results["transfer"]["ner_result"] += transfer_result["ner_result"]
        final_results["transfer"]["re_result"] += transfer_result["re_result"]
    torch.cuda.empty_cache()

    # Multilingual (evaluated only once, as it does not depend on training languages)
    multilingual_ner = NamedEntityModel(model_path="models/base_ner")
    multilingual_re = RelationsModel(model_path="models/re_entity_with_special_characters")
    train_df, test_df = create_datasets_from_language_lists(
        training_languages, testing_languages, downsample_number=None
    )
    multilingual_ner_result = multilingual_ner.evaluate(model_path="models/base_ner", df=test_df)
    multilingual_re_result = multilingual_re.evaluate(model_path="models/re_entity_with_special_characters", df=test_df)
    final_results["multilingual"]["ner_result"] = multilingual_ner_result["eval_overall_f1"]
    final_results["multilingual"]["re_result"] = multilingual_re_result["f1"]
    for key in final_results:
        if key != "multilingual":  # Multilingual is not averaged
            final_results[key]["ner_result"] /= number_of_runs
            final_results[key]["re_result"] /= number_of_runs

    return final_results

def filter_relations(train_df, test_df):
    test_relations = set(test_df['label'].unique())
    filtered_train_df = train_df[train_df['label'].isin(test_relations)]
    return filtered_train_df


def create_datasets_from_language_lists(training_languages, testing_languages, downsample_number=None, downsample_main=None):
    training_files = [f"data/{lang}_corpora_train.tsv" for lang in training_languages]
    testing_files = [f"data/{lang}_corpora_test.tsv" for lang in testing_languages]
    train_dfs = [pd.read_csv(file, sep="\t") for file in training_files]
    test_dfs = [pd.read_csv(file, sep="\t") for file in testing_files]
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    train_df=filter_relations(train_df,test_df)
    # Check if all languages in test_df are present in train_df
    if downsample_main is not None:
        to_downsample = set(test_df["lang"].unique())
        downsampled = train_df[train_df["lang"].isin(to_downsample)].sample(
            n=min(len(train_df[train_df["lang"].isin(to_downsample)]), downsample_main)
        )
        train_df = pd.concat([downsampled, train_df[~train_df["lang"].isin(to_downsample)]], ignore_index=True)
    print(f"Dataset from {training_languages} has {len(train_df)} examples")
    if downsample_number is not None:
        to_downsample = set(train_df["lang"].unique()) - set(test_df["lang"].unique())
        downsampled = train_df[train_df["lang"].isin(to_downsample)].sample(
            n=min(len(train_df), downsample_number)
        )
        train_df = pd.concat([downsampled, train_df[~train_df["lang"].isin(to_downsample)]], ignore_index=True)
    train_df = filter_out_wrong_data(train_df)
    test_df = filter_out_wrong_data(test_df)
    print(f"After all filtering train dataset from {training_languages} has {len(train_df)} examples")
    return train_df, test_df

def train_and_evaluate_on_language_subsets(
    train_df,
    test_df,
    ner_model,
    re_model,
    enhance_function=enhance_with_special_characters,
    average_type="micro",
    skip_ner_training=False
):
    torch.cuda.empty_cache()
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
        "ner_result": ner_result['eval_overall_f1'],
        "re_result": re_result['f1'],
    }
    return results

def run_experiments_linguistic():
    # Same language family
    ner_model = NamedEntityModel()
    re_model = RelationsModel()
    perform_four_variations_linguistic(
        ["es", "it", "pt"], ["fr"], ner_model, re_model, enhance_function=enhance_with_special_characters, average_type="micro"
    )
    perform_four_variations_linguistic(
        ["de", "en-full", "sv"],
        ["fr"],
        ner_model,
        re_model,
        enhance_function=enhance_with_special_characters,
        average_type="micro",
        downsample_number=128373,
    )

    # Cross-script
    ner_model = NamedEntityModel()
    re_model = RelationsModel()
    perform_four_variations_linguistic(
        ["ar"], ["ru"], ner_model, re_model, enhance_function=enhance_with_special_characters, average_type="micro"
    )
    perform_four_variations_linguistic(
        ["pl"],
        ["ru"],
        ner_model,
        re_model,
        enhance_function=enhance_with_special_characters,
        average_type="micro",
        downsample_number=9304,
    )

    # Complex language and simple language
    perform_four_variations_linguistic(
        ["pl"],
        ["es"],
        ner_model,
        re_model,
        enhance_function=enhance_with_special_characters,
        average_type="micro",
        downsample_number=4483,
    )
    perform_four_variations_linguistic(
        ["sv"],
        ["es"],
        ner_model,
        re_model,
        enhance_function=enhance_with_special_characters,
        average_type="micro",
    )

    # SVO
    perform_four_variations_linguistic(
        ["ko"], ["fa"], ner_model, re_model, enhance_function=enhance_with_special_characters, average_type="micro"
    )
    perform_four_variations_linguistic(
        ["pt"],
        ["fa"],
        ner_model,
        re_model,
        enhance_function=enhance_with_special_characters,
        average_type="micro",
        downsample_number=18712,
    )

    # Lexical Overlap
    perform_four_variations_linguistic(
        ["fr"], ["it"], ner_model, re_model, enhance_function=enhance_with_special_characters, average_type="micro"
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
        enhance_function=enhance_with_special_characters,
        average_type='micro',
    )



