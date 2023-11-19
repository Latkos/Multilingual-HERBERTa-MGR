import re

def extract_and_replace(text, tag, replacement):
    pattern = re.compile(f"{re.escape(tag)}(.*?){re.escape(tag[0] + '/' + tag[1:])}")
    return pattern.sub(replacement, text)

def enhance_with_nothing(predictions):
    enhanced_text = []
    for prediction in predictions:
        text = prediction['text'].replace('<e1>', '')
        text = text.replace('</e1>', '')
        text = text.replace('<e2>', '')
        text = text.replace('</e2>', '')
        enhanced_text.append(text)
    return enhanced_text

def enhance_with_entity_differentiated(predictions):
    enhanced_text = []
    for prediction in predictions:
        text = extract_and_replace(extract_and_replace(prediction['text'], "<e1>", "Entity1"), "<e2>", "Entity2")
        enhanced_text.append(text)
    return enhanced_text

def enhance_with_brackets(predictions):
    enhanced_text = []
    for prediction in predictions:
        text = extract_and_replace(extract_and_replace(prediction['text'], "<e1>", lambda m: f"<e1>{m.group(1)}</e1>"), "<e2>", lambda m: f"<e2>{m.group(1)}</e2>")
        enhanced_text.append(text)
    return enhanced_text

def enhance_with_entity(predictions):
    enhanced_text = []
    for prediction in predictions:
        text = extract_and_replace(extract_and_replace(prediction['text'], "<e1>", "Entity"), "<e2>", "Entity")
        enhanced_text.append(text)
    return enhanced_text

def enhance_with_special_characters(predictions):
    enhanced_text = []
    for prediction in predictions:
        text = extract_and_replace(extract_and_replace(prediction['text'], "<e1>", lambda m: f"${m.group(1)}$"), "<e2>", lambda m: f"#{m.group(1)}#")
        enhanced_text.append(text)
    return enhanced_text

def enhance_entities_only(ner_predictions):
    enhanced_text = []
    for prediction in ner_predictions:
        text = f"{prediction['entity_1']} # {prediction['entity_2']}"
        enhanced_text.append(text)
    return enhanced_text