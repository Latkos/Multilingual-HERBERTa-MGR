from named_entity.named_entity_model import NamedEntityModel
from relations.relations_model import RelationsModel
from utils.config_parser import get_training_args
from utils import csv_merger
import pandas as pd

from utils.prediction import evalute_predictions_of_combined_models
from utils.preprocessing import filter_out_wrong_data

if __name__ == "__main__":
    # train_df=pd.read_csv('merged_train.tsv',sep='\t')
    # test_df=pd.read_csv('merged_test.tsv',sep='\t')
    # train_df=train_df.sample(frac=0.02,random_state=42)
    # train_df = filter_out_wrong_data(train_df)
    # test_df = filter_out_wrong_data(test_df)
    # re_model=RelationsModel(m  odel_path='models/re_hyperparameter_search')
    # re_model.perform_hyperparameter_search(space=optuna_hp_space,train_df=train_df)

    train_df = pd.read_csv("merged_train.tsv", sep="\t")
    test_df = pd.read_csv("merged_test.tsv", sep="\t")
    train_df = train_df.sample(frac=0.02)
    train_df = filter_out_wrong_data(train_df)
    test_df = filter_out_wrong_data(test_df)
    ner_model=NamedEntityModel(model_path="models/ner_size_quality_tests_0.6/")
    re_model=RelationsModel(model_path="")
    evalute_predictions_of_combined_models(
        test_df=test_df,
        ner_model=ner_model,
        re_model=re_model,
    )
    # print(ner_model.predict(["John Smith has a child called Jake."]))
