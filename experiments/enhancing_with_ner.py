from utils.enhancement import enhance_with_nothing, enhance_with_entity, enhance_with_brackets, \
    enhance_with_entity_differentiated, enhance_with_special_characters, enhance_entities_only
from utils.prediction import train_re_on_ner, save_results


def test_enhancing_text_used_to_train_re(train_df, test_df, ner_model, re_model,results_file='results_optimized_ner.txt',read=True, train_ner=False):
    enhancements = [
        ("brackets", enhance_with_brackets),
        ("no_enhance", enhance_with_nothing),
        ("entity", enhance_with_entity),
        ("entity_differentiated", enhance_with_entity_differentiated),
        ("entity_with_special_characters", enhance_with_special_characters),
        ("entity_only", enhance_entities_only)
    ]
    if train_ner:
        ner_model.train(train_df=train_df)
        results = ner_model.predict(sentences=train_df['text'].tolist())
        save_results(results_file, results)
    f1_values=[]
    for enhancement_name, enhancement_func in enhancements:
        result=train_re_on_ner(ner_model, re_model, train_df, test_df, enhancement_func, results_file, read, model_path=f"models/re_{enhancement_name}")
        f1=result['f1']
        f1_values.append(f1)
        print(f"For enhancement {enhancement_name} the F1 is equal to {f1}")
    return f1_values