import torch
from utils.enhancement import enhance_with_nothing, enhance_with_entity, enhance_with_brackets, \
    enhance_with_entity_differentiated
import pickle

def save_results(filename, data):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)

def train_and_evaluate(re_model, train_df, test_df, model_name):
    print(f"Training {model_name}")
    print(train_df.head(10)['text'].tolist())
    re_model.train(train_df, model_path=f"models/{model_name}")
    re_model.evaluate(df=test_df, model_path=f"models/{model_name}")

def test_enhancing_text_used_to_train_re(train_df, test_df, ner_model, re_model):
    results = ner_model.predict(sentences=train_df['text'].tolist())
    save_results('results_optimized_ner.txt', results)
    # with open('results_optimized_ner.txt', 'rb') as file:
    #     results = pickle.load(file)
    enhancements = [
        ("brackets", enhance_with_brackets),
        ("no_enhance", enhance_with_nothing),
        ("entity", enhance_with_entity),
        ("entity_differentiated", enhance_with_entity_differentiated)
    ]

    for enhancement_name, enhancement_func in enhancements:
        enhanced_train_df = train_df.copy()
        enhanced_input = enhancement_func(results)
        enhanced_train_df['text'] = enhanced_input
        train_and_evaluate(re_model, enhanced_train_df, test_df, f"re_enhancing_{enhancement_name}_second_tryout")
        torch.cuda.empty_cache()
