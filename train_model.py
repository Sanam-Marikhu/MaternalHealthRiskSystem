import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("dataset/maternal_health.csv")

# Encode target
encoder = LabelEncoder()
df["RiskLevel"] = encoder.fit_transform(df["RiskLevel"])

# Split data
X = df.drop("RiskLevel", axis=1)
y = df["RiskLevel"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Save model
import joblib
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")

# Simple prediction demo
print("\nPrediction Demo:")

sample1 = [[25, 120, 80, 6.0, 98.0, 80]]
prediction = model.predict(sample1)

print("Predicted Risk:", encoder.inverse_transform(prediction)[0])

sample2 = [[25, 140, 100, 8.0, 98.0, 80]]
prediction = model.predict(sample2)

print("Predicted Risk:", encoder.inverse_transform(prediction)[0])
