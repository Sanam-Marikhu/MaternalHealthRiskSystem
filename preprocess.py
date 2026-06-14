import pandas as pd

def clean_data(path):
    df = pd.read_csv(path)

    # 1. Check missing values
    df = df.dropna()

    # 2. Remove duplicates
    df = df.drop_duplicates()

    # 3. Fix data types
    df["Age"] = df["Age"].astype(int)
    df["SystolicBP"] = df["SystolicBP"].astype(int)
    df["DiastolicBP"] = df["DiastolicBP"].astype(int)

    return df