import pandas as pd

def filter_out_wrong_data(df):
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype("int")
    df = df[~df["text"].str.replace(" ", "").str.contains("<e1></e1>")]
    df = df[~df["text"].str.replace(" ", "").str.contains("<e2></e2>")]
    return df
