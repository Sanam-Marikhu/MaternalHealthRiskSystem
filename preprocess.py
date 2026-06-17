import pandas as pd

def clean_data(path):
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Drop rows where target is missing
    df = df.dropna(subset=["Risk_Level"])

    # Fill numeric missing values with median
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df