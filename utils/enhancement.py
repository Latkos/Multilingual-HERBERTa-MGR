def enhance_with_nothing(ner_predictions):
    text = []
    for prediction in ner_predictions:
        text.append(prediction["text"])
    return text

def enhance_with_entity_differentiated(ner_predictions):
    enhanced_text = []
    for prediction in ner_predictions:
        if prediction["entity_1"]!="":
            prediction["text"]=prediction["text"].replace(prediction["entity_1"],"Entity1")
        if prediction["entity_2"]!="":
            prediction["text"]=prediction["text"].replace(prediction["entity_2"],"Entity2")
        enhanced_text.append(prediction["text"])
    return enhanced_text

def enhance_with_brackets(ner_predictions):
    enhanced_text = []
    for prediction in ner_predictions:
        if prediction["entity_1"]!="":
            prediction["text"]=prediction["text"].replace(prediction["entity_1"],f"<e1>{prediction['entity_1']}</e1>")
        if prediction["entity_2"]!="":
            prediction["text"]=prediction["text"].replace(prediction["entity_2"],f"<e2>{prediction['entity_2']}</e2>")
        enhanced_text.append(prediction["text"])
    return enhanced_text

def enhance_with_entity(ner_predictions):
    enhanced_text = []
    for prediction in ner_predictions:
        if prediction["entity_1"]!="":
            prediction["text"]=prediction["text"].replace(prediction["entity_1"],"Entity")
        if prediction["entity_2"]!="":
            prediction["text"]=prediction["text"].replace(prediction["entity_2"],"Entity")
        enhanced_text.append(prediction["text"])
    return enhanced_text

def enhance_with_special_characters (ner_predictions):
    enhanced_text = []
    for prediction in ner_predictions:
        if prediction["entity_1"]!="":
            prediction["text"]=prediction["text"].replace(prediction["entity_1"],f"${prediction['entity_1']}$")
        if prediction["entity_2"]!="":
            prediction["text"]=prediction["text"].replace(prediction["entity_2"],f"#{prediction['entity_2']}#")
        enhanced_text.append(prediction["text"])
    return enhanced_text

def enhance_entities_only(ner_predictions):
    enhanced_text = []
    for prediction in ner_predictions:
        prediction["text"] = f"{prediction['entity_1']} {prediction['entity_2']}"
        enhanced_text.append(prediction["text"])
    return enhanced_text
