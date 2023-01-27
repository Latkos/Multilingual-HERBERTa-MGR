from transformers import AutoTokenizer, BertForTokenClassification, pipeline

from ner.ner_config import NERConfig


def load_tokenizer(model_path):
    """Load tokenizer from saved model file

    Args:
        model_name (string): Model name

    Returns:
        huggingface.Tokenizer: Tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def load_model(model_path):
    """Load Model from saved model file

    Args:
        model_name (string): Model name

    Returns:
        huggingface.ForTokenClassification: Token classification
    """
    model = BertForTokenClassification.from_pretrained(
        model_path, num_labels=len(NERConfig.LABEL_NAMES)
    )
    return model


def get_entities_sentence(list_entity_groups):
    """
    Postprocessing for prediction result.

    Args:
        list_entity_groups (list): List of predicted group for words
        e.g:
            {
                'entity_group': 'LABEL_0',
                'score': 0.99553,
                'word': 'developed by',
                'start': 190,
                'end': 202
            },
            {
                'entity_group': 'LABEL_1',
                'score': 0.7543062,
                'word': 'TI',
                'start': 203,
                'end': 205
            },
            {
                'entity_group': 'LABEL_0',
                'score': 0.9995715,
                'word': '.',
                'start': 205,
                'end': 206
            }]

    Returns:
        dict:
            'ENTITY_1': string coresponded with entity 1,
            'ENTITY_2': string coresponded with entity 2,
            'TEXT': input text with <e1>, </e1>, <e2>, </e2> tags set
    """
    words_list = []
    entity_1_words = []
    entity_2_words = []
    flag = 0

    for group in list_entity_groups:
        if group.get("entity_group") == "LABEL_1":
            if flag == 0:
                words_list.append("<e1>")
            word = group.get("word")
            words_list.append(word)
            entity_1_words.append(word)
            flag = 1
        elif group.get("entity_group") == "LABEL_2":
            word = group.get("word")
            words_list.append(word)
            entity_1_words.append(word)
        elif group.get("entity_group") == "LABEL_3":
            if flag == 0:
                words_list.append("<e2>")
            word = group.get("word")
            words_list.append(word)
            entity_2_words.append(word)
            flag = 2
        elif group.get("entity_group") == "LABEL_4":
            word = group.get("word")
            words_list.append(word)
            entity_2_words.append(word)
        elif group.get("entity_group") == "LABEL_0":
            word = group.get("word")
            if flag == 1:
                words_list.append("</e1>")
                flag = 0
            elif flag == 2:
                words_list.append("</e2>")
                flag = 0
            words_list.append(word)

    sentence = " ".join(words_list)
    entity_1 = " ".join(entity_1_words)
    entity_2 = " ".join(entity_2_words)
    return {"ENTITY_1": entity_1, "ENTITY_2": entity_2, "TEXT": sentence}


def prediction(model_name, sentences):
    """
    Prediction a entity 1 and entity 2 for sentence or list sentences

    Args:
        model_name (string): Saved model name
        sentences (list or string): text to predict entity 1 and entity 2

    Returns:
        list: list of dict for each sentence with entity_1, entity_2 and
        input sentence with <e1>, </e1>, <e2>, </e2> tags set.
    """

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)
    result = []

    token_classifier = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )

    groups = token_classifier(sentences)

    if isinstance(groups[0], dict):
        result.append(get_entities_sentence(groups))
    else:
        for i in groups:
            result.append(get_entities_sentence(i))

    return result
