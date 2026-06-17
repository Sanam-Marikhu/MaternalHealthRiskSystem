import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import clean_data
import matplotlib.pyplot as plt

# Load dataset
df = clean_data("dataset/high_risk_pregnancy.csv")

# Clean column names (important for consistency)
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Encode target
encoder = LabelEncoder()
df["Risk_Level"] = encoder.fit_transform(df["Risk_Level"])

# FEATURES & TARGET
X = df.drop("Risk_Level", axis=1)
y = df["Risk_Level"]

# CROSS VALIDATION (REALISTIC SCORE)
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    min_samples_split=5,
    max_depth=10
)
cv_scores = cross_val_score(model, X, y, cv=5)

print("\nCross Validation Accuracy:")
print("Mean:", np.mean(cv_scores))
print("Scores:", cv_scores)


# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# TRAIN MODEL
model.fit(X_train, y_train)


# PREDICTIONS
y_pred = model.predict(X_test)

# EVALUATION
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)

# FEATURE IMPORTANCE
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(feature_importance)

# FEATURE IMPORTANCE PLOT
sorted_idx = importance.argsort()

plt.figure(figsize=(8, 5))
plt.barh(X.columns[sorted_idx], importance[sorted_idx])
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

# Save model

joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")
print("Model saved successfully!")
