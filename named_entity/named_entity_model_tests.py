from named_entity.named_entity_model import NamedEntityModel


def test_ner_model_quality_depending_on_dataset_size(train_df, test_df, sizes):
    for size in sizes:
        print(f'Size: {size}')
        sampled_train_df = train_df.sample(size)
        model_path = f'ner_size_quality_tests_{size}'
        ner_model = NamedEntityModel(model_path=model_path)
        model = ner_model.train(train_df=sampled_train_df)
        print(ner_model.evaluate(test_df, model_path))
