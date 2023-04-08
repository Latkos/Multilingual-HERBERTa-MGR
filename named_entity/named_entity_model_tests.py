import pandas as pd
from named_entity.named_entity_model import NamedEntityModel


def test_ner_model_quality_depending_on_dataset_size(train_df, test_df, sizes):
    results = []
    for size in sizes:
        print(f"Size: {size}")
        sampled_train_df = train_df.sample(frac=size)
        model_path = f"ner_size_quality_tests_{size}"
        ner_model = NamedEntityModel(model_path=model_path)
        ner_model.train(train_df=sampled_train_df)
        evaluation_result = ner_model.evaluate(test_df, model_path)
        results.append({"size": size, "f1": evaluation_result["eval_overall_f1"]})
    results_df = pd.DataFrame(results)
    return results_df
