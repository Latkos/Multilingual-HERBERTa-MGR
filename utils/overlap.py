import pandas as pd

def compute_overlap(entity_dict):
    overlap_percentages = {}
    languages = list(entity_dict.keys())
    for i in range(len(languages)):
        for j in range(i + 1, len(languages)):
            lang1 = languages[i]
            lang2 = languages[j]
            print(f"{lang1}, {lang2}")
            if isinstance(entity_dict[lang1], set):
                intersection = len(entity_dict[lang1].intersection(entity_dict[lang2]))
                union = len(entity_dict[lang1].union(entity_dict[lang2]))
            else:
                intersection = sum(1 for e in entity_dict[lang1] if e in entity_dict[lang2])
                union = len(entity_dict[lang1]) + len(entity_dict[lang2])

            overlap_percent = round((intersection / union) * 100,4)
            overlap_percentages[(lang1, lang2)] = overlap_percent
    df = pd.DataFrame(list(overlap_percentages.items()), columns=['Languages', 'Value'])
    df[['Language1', 'Language2']] = pd.DataFrame(df['Languages'].tolist(), index=df.index)
    del df['Languages']
    return df


def compute_both_entity_overlap(df):
    distinct_language_entities = {}  # For distinct entities
    all_language_entities = {}  # For all entities
    for lang, group in df.groupby('lang'):
        distinct_entities = set(group['entity_1']).union(set(group['entity_2']))
        distinct_language_entities[lang] = distinct_entities
        all_entities = group['entity_1'].tolist() + group['entity_2'].tolist()
        all_language_entities[lang] = all_entities
    distinct_overlap = compute_overlap(distinct_language_entities)
    all_overlap = compute_overlap(all_language_entities)
    return distinct_overlap, all_overlap
