import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- Set wide layout ---
st.set_page_config(layout="wide")

# --- Your Provided Code (Adapted for Streamlit) ---

# Load data
# Note: Ensure '502纳入.xlsx' is in the same directory or provide the full path
try:
    df = pd.read_excel('502纳入.xlsx')
except FileNotFoundError:
    st.error("Error: File '502纳入.xlsx' not found. Please ensure it's in the same directory.")
    st.stop()

# Rename columns
df.rename(columns={'SOFA':'SOFA',
                   'PO2/FiO2':'PO2/FiO2(mmHg)',
                   'K': 'K(mmol/L)',
                   '24-hour fluid balance':'24-hour fluid balnce(ml)',
                   'Cr':'Cr(umol/L)',
                   'D-Dimer':'D-Dimer(mg/L)',
                   'AST':'AST(IU/L)',
                   'GLU':'GLU(mmol/L)',
                   'pH':'pH',},inplace=True)

# Define variables
continuous_vars = [
    'SOFA',
    'PO2/FiO2(mmHg)',
    'K(mmol/L)',
    '24-hour fluid balnce(ml)',
    'Cr(umol/L)',
    'D-Dimer(mg/L)',
    'AST(IU/L)',
    'GLU(mmol/L)',
    'pH',
]

# Combine all variables for unified input
all_vars = continuous_vars

# Check if all required columns exist
missing_cols = [col for col in all_vars + ['急性肾衰竭'] if col not in df.columns]
if missing_cols:
    st.error(f"Error: The following columns are missing from the dataset: {missing_cols}")
    st.stop()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_vars),
    ])

# Apply preprocessing
X_processed = preprocessor.fit_transform(df[continuous_vars])

# Get feature names
feature_names = continuous_vars

X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
X = X_processed_df
y = df['急性肾衰竭']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

# --- Streamlit App Interface ---

# Centered Title
st.markdown("<h1 style='text-align: center;'>Logistic Regression Model for Predicting AKI Secondary to ARDS </h1>", unsafe_allow_html=True)

# --- 1. User Input for X values (Unified, 3 columns) ---
st.header("1. Enter Patient Data")

user_input = {}
input_valid = True

# Create input fields for all variables in 3 columns
input_cols = st.columns(3)
for i, var in enumerate(all_vars):
    with input_cols[i % 3]:
        # Handle different variables with appropriate ranges
        if var == "SOFA":
            user_val = st.number_input(f"{var}", value=None, step=1.0, placeholder="e.g., 6.0")
        elif var == "PO2/FiO2(mmHg)":
            user_val = st.number_input(f"{var}",  value=None, step=0.01, placeholder="e.g., 200.0")
        elif var == "K(mmol/L)":
            user_val = st.number_input(f"{var}", value=None, step=0.01, placeholder="e.g., 4.0")
        elif var == "24-hour fluid balnce(ml)":
            user_val = st.number_input(f"{var}", value=None, step=0.01, placeholder="e.g., 1500.0")
        elif var == "Cr(umol/L)":
            user_val = st.number_input(f"{var}", value=None, step=0.01, placeholder="e.g., 100.0")
        elif var == "D-Dimer(mg/L)":
            user_val = st.number_input(f"{var}", value=None, step=0.01, placeholder="e.g., 1.5")
        elif var == "AST(IU/L)":
            user_val = st.number_input(f"{var}", value=None, step=0.01, placeholder="e.g., 40.0")
        elif var == "GLU(mmol/L)":
            user_val = st.number_input(f"{var}", value=None, step=0.01, placeholder="e.g., 6.0")
        elif var == "pH":
            user_val = st.number_input(f"{var}", value=None, step=0.01, placeholder="e.g., 7.4")
        else:
            user_val = st.number_input(f"{var}", value=None, step=0.01, placeholder="please enter")
        
        if user_val is None:
            input_valid = False
        user_input[var] = user_val

# --- 2. Model Parameter Display ---
st.header("2. Model Parameters")

# Display model information
st.info("""
**Model Type:** Logistic Regression with L2 regularization  
**Class Weight:** Balanced (to handle imbalanced classes)  
""")

# --- 3. Prediction Button and Logic ---
if st.button("Train Model and Predict"):
    if not input_valid:
        st.error("Error: Please fill in all the required fields.")
    else:
        # Create a DataFrame from user input
        input_data = pd.DataFrame([user_input])
        
        # --- Train the model ---
        try:
            # Initialize and train the model
            model = LogisticRegression(
                random_state=999,
                penalty='l2',
                class_weight='balanced',
            )
            
            with st.spinner("Training model..."):
                model.fit(X_train, y_train)
            
            st.success("Model trained successfully!")
            
            # Apply the same preprocessing to input data
            input_processed = preprocessor.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_processed)[0]
            prediction_proba = model.predict_proba(input_processed)[0]
            
            # Display results
            st.header("Prediction Result")
            
            # Create two columns for results display
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 1:
                    st.error("**Prediction: Acute Kidney Injury (AKI)**")
                else:
                    st.success("**Prediction: No Acute Kidney Injury**")
            
            with result_col2:
                prob_aki = prediction_proba[1] * 100
                st.metric(
                    label="Probability of Acute Kidney Injury", 
                    value=f"{prob_aki:.1f}%",
                    delta=f"{(prob_aki - 50):+.1f}%" if prob_aki != 50 else None
                )
            
            # Display probability breakdown
            st.subheader("Probability Breakdown")
            with prob_col1:
                st.metric(label="Probability of AKI", value=f"{prediction_proba[1]*100:.1f}%")
            with prob_col2:
                st.metric(label="Probability of AKI", value=f"{prediction_proba[1]*100:.1f}%")    
        except Exception as e:
            st.error(f"An error occurred during model training or prediction: {str(e)}")
            st.error("Please check if all input values are valid and try again.")


# --- Disclaimer Section at the Bottom ---
st.markdown("---") # Horizontal line separator
st.header("Disclaimer and Data Definitions")

disclaimer_text = """
**Clinical Note:** This tool provides a predictive assessment based on statistical modeling and should be used as a decision support aid only. Clinical judgment should always supersede algorithmic predictions.

**Variable Definitions (measured within first 24 hours of ICU admission):**

* **AST**: Peak AST (aspartate aminotransferase) value
* **Cr**: Peak creatinine value  
* **GLU**: Peak venous blood glucose level
* **K+**: Peak serum potassium level
* **D-Dimer**: Peak D-Dimer level
* **SOFA**: Sequential Organ Failure Assessment score
* **pH**: pH value from arterial blood gas
* **PO2/FiO2**: Ratio of partial pressure of oxygen to fraction of inspired oxygen from arterial blood gas
* **24-hour fluid balance**: Net fluid balance (input minus output)


"""
st.markdown(disclaimer_text)

# Add footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>AKI Prediction Tool v1.0 | For clinical decision support only</div>", unsafe_allow_html=True)
