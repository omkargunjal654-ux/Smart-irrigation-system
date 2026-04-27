# Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
df = pd.read_csv("irrigation_dataset.csv")

# 2. Convert text to numbers
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# 3. Split data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 4. Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# 5. Take user input
print("Enter values:")

user_input = []
for col in X.columns:
    val = input(f"{col}: ")
    
    try:
        val = float(val)   # try number
    except:
        val = 0            # if text, put 0 (simple fix)
    
    user_input.append(val)

# 6. Prediction
prediction = model.predict([user_input])

# 7. Output
print("\nIrrigation Decision:", prediction[0])
# -------------------------------
# ADDITION FOR PRESENTATION 3
# -------------------------------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Split data for evaluation
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# MODEL 1: Decision Tree
# -------------------------------
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# -------------------------------
# MODEL 2: Random Forest
# -------------------------------
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# -------------------------------
# RESULTS
# -------------------------------
print("\n--- MODEL PERFORMANCE ---")
print("Decision Tree Accuracy:", dt_acc)
print("Random Forest Accuracy:", rf_acc)

# Confusion Matrix
print("\nConfusion Matrix (Random Forest):\n", confusion_matrix(y_test, rf_pred))

# Classification Report
print("\nClassification Report (Random Forest):\n", classification_report(y_test, rf_pred))

# -------------------------------
# GRAPH (MODEL COMPARISON)
# -------------------------------
models = ['Decision Tree', 'Random Forest']
accuracy = [dt_acc, rf_acc]

plt.figure()
plt.bar(models, accuracy)
plt.title("Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()