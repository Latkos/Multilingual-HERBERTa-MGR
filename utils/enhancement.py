def enhance_with_nothing(ner_predictions):
    text = []
    for prediction in ner_predictions:
        text.append(prediction["text"])
    return text