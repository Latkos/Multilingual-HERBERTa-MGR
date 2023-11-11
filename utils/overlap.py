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
    df = pd.DataFrame(list(overlap_percentages.items()), columns=['languages', 'value'])
    df[['language1', 'language2']] = pd.DataFrame(df['languages'].tolist(), index=df.index)
    del df['languages']
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

def get_overlaps(train_df,test_df,distinct_path='results/total_overlap_distinct.csv',all_path='results/total_overlap_all.csv'):
    df = pd.concat([train_df, test_df], ignore_index=True)
    total_overlap_distinct, total_overlap_all=compute_both_entity_overlap(df)
    total_overlap_distinct.to_csv(distinct_path,index=False)
    total_overlap_all.to_csv(all_path,index=False)

def create_full_matrix(file_path):
    df = pd.read_csv(file_path)
    matrix = df.pivot(index='language1', columns='language2', values='value')
    full_matrix = matrix.combine_first(matrix.T)
    full_matrix = full_matrix.round(1)
    return full_matrix

