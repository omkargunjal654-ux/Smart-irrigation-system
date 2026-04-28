import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- APP CONFIG ---
st.set_page_config(page_title="Irrigation Intelligence", layout="wide")
st.title("🌱 Smart Irrigation Prediction System")

# --- 1. LOAD DATA ---
@st.cache_data # Caches data so it doesn't reload on every click
def load_data():
    try:
        df = pd.read_csv("irrigation_dataset.csv")
        return df
    except FileNotFoundError:
        st.error("CSV file not found! Please ensure 'irrigation_dataset.csv' is in the same folder.")
        return None

df = load_data()

if df is not None:
    # --- 2. PREPROCESSING ---
    le = LabelEncoder()
    # We create a copy to keep original text for display if needed
    processed_df = df.copy()
    
    for col in processed_df.columns:
        if processed_df[col].dtype == 'object':
            processed_df[col] = le.fit_transform(processed_df[col])

    X = processed_df.iloc[:, :-1]
    y = processed_df.iloc[:, -1]

    # Split for training/testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. MODEL TRAINING ---
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)

    # --- 4. SIDEBAR USER INPUT ---
    st.sidebar.header("User Input Parameters")
    user_data = []
    
    for col in X.columns:
        # Use number_input for numerical features
        val = st.sidebar.number_input(f"Enter {col}", value=0.0)
        user_data.append(val)

    # --- 5. PREDICTION ---
    if st.sidebar.button("Predict Irrigation"):
        prediction = rf_model.predict([user_data])
        # Mapping back if target was encoded (assuming 1=Yes, 0=No or similar)
        result = "YES (Irrigate)" if prediction[0] == 1 else "NO (Wait)"
        st.success(f"### Prediction: {result}")

    # --- 6. DASHBOARD VISUALS ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance")
        rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
        dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
        
        fig, ax = plt.subplots()
        ax.bar(['Decision Tree', 'Random Forest'], [dt_acc, rf_acc], color=['skyblue', 'green'])
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)

    with col2:
        st.subheader("Dataset Preview")
        st.write(df.head(10))