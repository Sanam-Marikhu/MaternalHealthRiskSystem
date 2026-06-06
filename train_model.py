import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("dataset/maternal_health.csv")

# Encode RiskLevel
encoder = LabelEncoder()
df['RiskLevel'] = encoder.fit_transform(df['RiskLevel'])

# Separate features and target
X = df.drop("RiskLevel", axis=1)
y = df["RiskLevel"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")

print("Model saved successfully!")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

print(feature_importance.sort_values(
    by='Importance',
    ascending=False
))

# # Custom Prediction

# print("\n===== Maternal Health Risk Prediction =====")

# age = int(input("Enter Age: "))
# systolic = int(input("Enter Systolic BP: "))
# diastolic = int(input("Enter Diastolic BP: "))
# bs = float(input("Enter Blood Sugar Level: "))
# temp = float(input("Enter Body Temperature: "))
# heart_rate = int(input("Enter Heart Rate: "))

# # Create input data
# custom_data = pd.DataFrame([{
#     'Age': age,
#     'SystolicBP': systolic,
#     'DiastolicBP': diastolic,
#     'BS': bs,
#     'BodyTemp': temp,
#     'HeartRate': heart_rate
# }])
# # Predict
# prediction = model.predict(custom_data)

# # Convert prediction back to label
# risk_level = encoder.inverse_transform(prediction)

# print("\nPredicted Risk Level:", risk_level[0])