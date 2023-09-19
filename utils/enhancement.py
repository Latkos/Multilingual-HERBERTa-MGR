def enhance_with_nothing(ner_predictions):
    text = []
    for prediction in ner_predictions:
        text.append(prediction["text"])
    return text

def enhance_with_entity_differentiated(ner_predictions):
    enhanced_text = []
    for prediction in ner_predictions:
        enhanced_text.append(prediction["text"].replace(prediction["entity_1"], "Entity1").replace(prediction["entity_2"], "Entity2"))
    return enhanced_text

def enhance_with_brackets(ner_predictions):
    enhanced_text = []
    for prediction in ner_predictions:
        enhanced_text.append(
            prediction["text"]
            .replace(prediction["entity_1"], f"<e1>{prediction['entity_1']}</e1>")
            .replace(prediction["entity_2"], f"<e2>{prediction['entity_2']}</e2>")
        )
    return enhanced_text

def enhance_with_entity(ner_predictions):
    enhanced_text = []
    for prediction in ner_predictions:
        enhanced_text.append(prediction["text"].replace(prediction["entity_1"], "Entity").replace(prediction["entity_2"], "Entity"))
    return enhanced_text
