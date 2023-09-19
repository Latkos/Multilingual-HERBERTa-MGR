import pandas as pd
from utils.enhancement import enhance_with_nothing, enhance_with_entity, enhance_with_brackets, enhance_with_entity_differentiated


def test_enhancing_text_used_to_train_re(train_df,test_df,ner_model,re_model):
    results=ner_model.predict(sentences=train_df['text'].tolist())
    input_no_enhance=enhance_with_nothing(results)
    input_enhance_with_entity=enhance_with_entity(results)
    input_enhance_with_entity_differentiated=enhance_with_entity_differentiated(results)
    input_enhance_with_brackets=enhance_with_brackets(results)

    df_no_enhance = pd.DataFrame({'text': input_no_enhance})
    df_entity = pd.DataFrame({'text': input_enhance_with_entity})
    df_entity_differentiated = pd.DataFrame({'text': input_enhance_with_entity_differentiated})
    df_brackets = pd.DataFrame({'text': input_enhance_with_brackets})
    train_df_no_enhance = pd.concat([train_df.drop(columns='text'), df_no_enhance], axis=1)
    train_df_entity = pd.concat([train_df.drop(columns='text'), df_entity], axis=1)
    train_df_entity_differentiated = pd.concat([train_df.drop(columns='text'), df_entity_differentiated], axis=1)
    train_df_brackets = pd.concat([train_df.drop(columns='text'), df_brackets], axis=1)
    re_model.train(train_df_no_enhance, model_path='models/re_enhancing_no_enhance')
    re_model.evaluate(df=test_df,model_path='models/re_enhancing_no_enhance')
    re_model.train(train_df_entity, model_path='models/re_enhancing_entity')
    re_model.evaluate(df=test_df,model_path='models/re_enhancing_entity')
    re_model.train(train_df_entity_differentiated, model_path='models/re_enhancing_entity_differentiated')
    re_model.evaluate(df=test_df,model_path='models/re_enhancing_entity_differentiated')
    re_model.train(train_df_brackets, model_path='models/re_enhancing_brackets')
    re_model.evaluate(df=test_df,model_path='models/re_enhancing_brackets')
