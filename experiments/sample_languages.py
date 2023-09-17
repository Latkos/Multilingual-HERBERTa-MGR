import pandas as pd


def filter_and_set_proportions(df, proportions):
    target_languages = list(proportions.keys())
    df_filtered = df[df['lang'].isin(target_languages)].copy()
    current_proportions = df_filtered['lang'].value_counts(normalize=True).to_dict()

    for language in target_languages:
        if language not in current_proportions:
            current_proportions[language] = 0.0
        target_proportion = proportions[language]

        if current_proportions[language] > target_proportion:
            num_samples_to_keep = int(target_proportion * len(df_filtered))
            df_filtered = df_filtered[df_filtered['lang'] != language]
            df_filtered = pd.concat([
                df_filtered[df_filtered['lang'] == language].sample(n=num_samples_to_keep, replace=False),
                df_filtered[df_filtered['lang'] != language]
            ])

    return df_filtered
