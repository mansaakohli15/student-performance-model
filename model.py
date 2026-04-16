import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('student_data.csv')

print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features (X) and target (y) - predict average of math, reading, writing
data['average_score'] = (data['math score'] + data['reading score'] + data['writing score']) / 3

X = data.drop(['math score', 'reading score', 'writing score', 'average_score'], axis=1)
y = data['average_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"\nTraining R² Score: {r2_score(y_train, y_train_pred):.4f}")
print(f"Testing R² Score: {r2_score(y_test, y_test_pred):.4f}")
print(f"\nMean Absolute Error (Test): {mean_absolute_error(y_test, y_test_pred):.2f}")
print(f"Root Mean Squared Error (Test): {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")

# Feature importance (coefficients)
print("\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)
features = X.columns
coefs = model.coef_
for feature, coef in zip(features, coefs):
    print(f"{feature:30}: {coef:.3f}")

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Average Score")
plt.ylabel("Predicted Average Score")
plt.title("Actual vs Predicted Student Performance")
plt.tight_layout()
plt.show()

# Predict for a sample student
print("\n" + "="*50)
print("SAMPLE PREDICTION")
print("="*50)
sample_student = [[1, 2, 3, 1, 1]]  # Example: female, group C, bachelor's, standard, completed
predicted = model.predict(sample_student)
print(f"Predicted average score: {predicted[0]:.2f}")