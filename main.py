import sys

import torch

from experiments.calculate_overlap import get_overlaps
from experiments.enhancing_with_ner import test_enhancing_text_used_to_train_re
from experiments.hyperparameters import optuna_hp_space, optuna_hp_space_scientific
from named_entity.named_entity_model import NamedEntityModel
from relations.relations_model import RelationsModel
import pandas as pd

from utils.preprocessing import filter_out_wrong_data

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train_df = pd.read_csv("merged_train.tsv", sep="\t")
    test_df = pd.read_csv("merged_test.tsv", sep="\t")
    train_df=train_df.sample(frac=0.2,random_state=42)
    train_df = filter_out_wrong_data(train_df)
    test_df = filter_out_wrong_data(test_df)
    ner_model=NamedEntityModel(model_path='models/hyperparameter_default_ner')
    re_model=RelationsModel(model_path='models/hyperparameter_default_re')
    re_model.train(train_df=train_df, remove_tags=False)
    re_model.evaluate(df=test_df)
    print("*******************")
    # re_model.perform_hyperparameter_search(space=optuna_hp_space_scientific,train_df=train_df, study_name="re_hyperparameter_search_scientific")



    # re_model_default=RelationsModel(model_path="models/default_re_50")
    # re_model_chosen=RelationsModel(model_path="models/chosen_re_50")
    #
    # re_model_default.train(train_df=train_df,remove_tags=False,config_path='config/base_config_old.yaml')
    # re_model_default.evaluate(df=test_df)
    # print("**************************************")
    # re_model_chosen.train(train_df=train_df,remove_tags=False,config_path='config/base_config.yaml')
    # re_model_chosen.evaluate(df=test_df)
    # test_enhancing_text_used_to_train_re(train_df=train_df,test_df=test_df,ner_model=ner_model,re_model=re_model)


    # get_overlaps(train_df=train_df,test_df=test_df)
    # train_df = train_df.sample(frac=0.5, random_state=42)
    # train_df = filter_out_wrong_data(train_df)
    # test_df = filter_out_wrong_data(test_df)
    # print(f"Size of training: {len(train_df)}")
    # print(f"Size of test: {len(test_df)}")
    # ner_model=NamedEntityModel()
    # re_model=RelationsModel()
    # Training on English, testing on English (single-language testing)
    # train_and_evaluate_on_language_subsets(ner_model=ner_model,re_model=re_model,training_files=['data/en-full'],testing_files=['data/en-full_corpora_test.tsv'],
    #                                        )
    # print("********************")
    # print("Training on other Romance, testing on French test (Romance language proximity)")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model,re_model=re_model,training_files=['data/es_corpora_train.tsv','data/it_corpora_train.tsv','data/pt_corpora_train.tsv'],testing_files=['data/fr_corpora_test.tsv'])
    # print("********************")
    # print("Training on other Romance, testing on French train (Romance language proximity)")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model,re_model=re_model,training_files=['data/es_corpora_train.tsv','data/it_corpora_train.tsv','data/pt_corpora_train.tsv'],testing_files=['data/fr_corpora_train.tsv'], skip_training=True)
    # print("********************")
    # print("Training on other Romance and French, testing on French test (Romance language proximity)")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model,re_model=re_model,training_files=['data/es_corpora_train.tsv','data/it_corpora_train.tsv','data/pt_corpora_train.tsv','data/fr_corpora_train.tsv'],testing_files=['data/fr_corpora_test.tsv'])
    # print("********************")
    # print("Monolingual French")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model,re_model=re_model,training_files=['data/fr_corpora_train.tsv'],testing_files=['data/fr_corpora_test.tsv'])
    # print("********************")
    # print("Training on German and Dutch, testing on English test (Germanic languages)")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model, re_model=re_model,
    #                                        training_files=['data/de_corpora_train.tsv',
    #                                                        'data/nl_corpora_train.tsv'],
    #                                        testing_files=['data/en-full_corpora_test.tsv'])
    # print("********************")
    # print("Training on German and Dutch and English, testing on English test (Germanic languages)")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model, re_model=re_model,
    #                                        training_files=['data/de_corpora_train.tsv',
    #                                                        'data/nl_corpora_train.tsv',
    #                                                        'data/en-full_corpora_train.tsv'],
    #                                        testing_files=['data/en-full_corpora_test.tsv'])
    # print("********************")
    #
    # print("Training on Arabic, testing on Korean test (one script to another script)")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model, re_model=re_model,
    #                                        training_files=['data/ar_corpora_train.tsv'],
    #                                        testing_files=['data/ko_corpora_test.tsv'])
    # print("********************")
    #
    # print("Training on Arabic, testing on Korean train (one script to another script) - Skip Training")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model, re_model=re_model,
    #                                        training_files=['data/ar_corpora_train.tsv'],
    #                                        testing_files=['data/ko_corpora_train.tsv'], skip_training=True)
    # print("********************")
    #
    # print("Training on Arabic and Korean, testing on Korean test (one script to another script)")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model, re_model=re_model,
    #                                        training_files=['data/ar_corpora_train.tsv', 'data/ko_corpora_train.tsv'],
    #                                        testing_files=['data/ko_corpora_test.tsv'])
    # print("********************")
    #
    # print("Training on Arabic and Russian, testing on French test (two different scripts to common script)")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model, re_model=re_model,
    #                                        training_files=['data/ar_corpora_train.tsv',
    #                                                        'data/ru_corpora_train.tsv'],
    #                                        testing_files=['data/fr_corpora_test.tsv'])
    # print("********************")
    #
    # print(
    #     "Training on Arabic and Russian, testing on French train (two different scripts to common script) - Skip Training")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model, re_model=re_model,
    #                                        training_files=['data/ar_corpora_train.tsv',
    #                                                        'data/ru_corpora_train.tsv'],
    #                                        testing_files=['data/fr_corpora_train.tsv'], skip_training=True)
    # print("********************")
    #
    # print("Training on Arabic and Russian and French, testing on French test (two different scripts to common script)")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model, re_model=re_model,
    #                                        training_files=['data/ar_corpora_train.tsv',
    #                                                        'data/ru_corpora_train.tsv',
    #                                                        'data/fr_corpora_train.tsv'],
    #                                        testing_files=['data/fr_corpora_test.tsv'])
    # print("********************")
    #
    # print("Training on Persian, testing on English (uncommon languages)")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model, re_model=re_model,
    #                                        training_files=['data/fa_corpora_train.tsv'],
    #                                        testing_files=['data/en-full_corpora_test.tsv'])
    # print("********************")
    #
    # print("Training on Persian and English, testing on English (uncommon languages)")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model, re_model=re_model,
    #                                        training_files=['data/fa_corpora_train.tsv',
    #                                                        'data/en-full_corpora_train.tsv'],
    #                                        testing_files=['data/en-full_corpora_test.tsv'])
    # print("********************")
    #
    # print("Training on Polish, testing on Spanish test (complex language to simple language)")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model, re_model=re_model,
    #                                        training_files=['data/pl_corpora_train.tsv'],
    #                                        testing_files=['data/es_corpora_test.tsv'])
    # print("********************")
    #
    # print("Training on Polish, testing on Spanish train (complex language to simple language) - Skip Training")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model, re_model=re_model,
    #                                        training_files=['data/pl_corpora_train.tsv'],
    #                                        testing_files=['data/es_corpora_train.tsv'], skip_training=True)
    # print("********************")
    #
    # print("Training on Polish and Spanish, testing on Spanish test (complex language to simple language)")
    # train_and_evaluate_on_language_subsets(ner_model=ner_model, re_model=re_model,
    #                                        training_files=['data/pl_corpora_train.tsv', 'data/es_corpora_train.tsv'],
    #                                        testing_files=['data/es_corpora_test.tsv'])
    #
    # languages = [
    #     # 'ar', 'de', 'es', 'fa', 'fr', 'it', 'ko',
    #              'nl', 'pl', 'pt', 'ru', 'sv',
    #              'uk']
    #
    # for language in languages:
    #     print(f"Monolingual {language}")
    #     train_file = f'data/{language}_corpora_train.tsv'
    #     test_file = f'data/{language}_corpora_test.tsv'
    #     train_and_evaluate_on_language_subsets(ner_model=ner_model, re_model=re_model, training_files=[train_file],
    #                                            testing_files=[test_file])

    # re_results=test_model_quality_depending_on_dataset_size(model=re_model, train_df=train_df, test_df=test_df, sizes=[0.0001,0.001,0.005,0.01,0.02,0.05,0.1,0.2])
    # re_results.to_csv('re_results_size_new.csv',index=False)
    # ner_results=test_model_quality_depending_on_dataset_size(model=ner_model, train_df=train_df, test_df=test_df, sizes=[0.0001,0.001,0.005,0.01,0.02,0.05,0.1,0.2])
    # ner_results.to_csv('ner_results_size.csv',index=False)

    # ner_model.evaluate(df=test_df)
    # re_model.evaluate(df=test_df)
    # test_df=test_df.sample(n=1)
    # predict_joint_models(
    #     test_df=test_df,
    #     ner_model=ner_model,
    #     re_model=re_model,
    #     enhance_function=enhance_with_nothing
    # )
    # print(ner_model.predict(["John Smith has a child called Jake."]))
