import pandas as pd
from datetime import datetime


def save_prediction(
        age,
        risk_level,
        confidence):

    new_record = pd.DataFrame([{
        "Timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Age": age,
        "Risk_Level": risk_level,
        "Confidence": round(confidence, 2)
    }])

    new_record.to_csv(
        "prediction_history.csv",
        mode="a",
        header=False,
        index=False
    )