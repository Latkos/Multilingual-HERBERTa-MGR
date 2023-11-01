from utils.overlap import compute_both_entity_overlap
import pandas as pd


def get_overlaps(train_df,test_df):
    df = pd.concat([train_df, test_df], ignore_index=True)
    print("For train_df")
    train_overlap_distinct, train_overlap_all=compute_both_entity_overlap(train_df)
    train_overlap_distinct.to_csv('results/train_overlap_distinct.csv',index=False)
    train_overlap_all.to_csv('results/train_overlap_all.csv',index=False)
    test_overlap_distinct, test_overlap_all=compute_both_entity_overlap(test_df)
    print(test_overlap_distinct)
    test_overlap_distinct.to_csv('results/test_overlap_distinct.csv',index=False)
    test_overlap_all.to_csv('results/test_overlap_all.csv',index=False)
    total_overlap_distinct, total_overlap_all=compute_both_entity_overlap(df)
    total_overlap_distinct.to_csv('results/total_overlap_distinct.csv',index=False)
    total_overlap_all.to_csv('results/total_overlap_all.csv',index=False)

