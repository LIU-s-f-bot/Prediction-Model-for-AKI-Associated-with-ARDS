import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# --- Set wide layout ---
st.set_page_config(layout="wide")

# --- Your Provided Code (Adapted for Streamlit) ---

# Load data
# Note: Ensure '502纳入.xlsx' is in the same directory or provide the full path
try:
    df = pd.read_excel('502纳入.xlsx')
except FileNotFoundError:
    st.error("Error, file not found")
    st.stop()

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

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_vars),
        ])

# Apply preprocessing
X_processed = preprocessor.fit_transform(df)

# Get feature names
try:
    feature_names = (
        continuous_vars +
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_vars))
    )
except AttributeError:
    feature_names = (
        continuous_vars +
        list(preprocessor.named_transformers_['cat'].get_feature_names(categorical_vars))
    )

X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
X = X_processed_df
y = df['急性肾衰竭']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)


# --- Streamlit App Interface ---

# Centered Title
st.markdown("<h1 style='text-align: center;'>Support Vector Machine model for predicting ARDS patients with Late acute AKI</h1>", unsafe_allow_html=True)

# --- 1. User Input for X values (Unified, 3 columns) ---
st.header("1. Enter Patient Data")

user_input = {} # summarize user input data
input_valid = True # Flag to check if all inputs are valid
# Create input fields for all variables in 3 columns
# Combine continuous and categorical for unified handling in layout
input_cols = st.columns(3) # Changed to 3 columns
for i, var in enumerate(all_vars):
    with input_cols[i % 3]: # Cycle through 3 columns
        if var in continuous_vars:
            # Handle continuous variables - No default value
            if var =="FiO2":
                user_val = st.number_input(f"{var}",value=None,format="%.4f", step=0.01, placeholder="please enter,e.g.,0.6")
            else:
                user_val = st.number_input(f"{var}", value=None, format="%.4f",step=0.01, placeholder="please enter")
            if user_val is None:
                input_valid = False
                #st.warning(f"请输入 {var} 的值")
            user_input[var] = user_val
        else: # Handle categorical variables
            # Get categories for categorical variables
            fitted_encoder = preprocessor.named_transformers_['cat']
            try:
                # Find the index of the categorical variable in the transformer's input list
                cat_var_index = categorical_vars.index(var)
                options = fitted_encoder.categories_[cat_var_index]
            except (AttributeError, ValueError, IndexError):
                # Fallback if categories_ is not directly available or index error
                options = np.unique(df[var].astype(str))
            # Default to the first category or a placeholder
            # UI - No default selection, user must choose
            selected_option = st.selectbox(f"{var}", options=options, index=None, placeholder="please enter")
            if selected_option is None:
                input_valid = False
                #st.warning(f"请选择 {var} 的值")
            user_input[var] = selected_option

# --- 2. Model Parameter Display (Fixed, no user selection) ---
#st.header("2. Model Parameters (Fixed)")

# Display fixed parameters
# Store fixed parameters
#FIXED_KERNEL = 'linear'
#FIXED_C = 1.0
#FIXED_CLASS_WEIGHT = 'balanced'

#col1, col2, col3 = st.columns(3)
#with col1:
    #st.metric(label="Kernel", value=FIXED_KERNEL)
#with col2:
 #   st.metric(label="Regularization Parameter (C)", value=FIXED_C )
#with col3:
 #   st.metric(label="Class Weight", value=FIXED_CLASS_WEIGHT)

# --- 3. Prediction Button and Logic ---
if st.button("Train Model and Predict"):
    if not input_valid:
        st.error("error, please check all X is inputed")
    else:
        # Create a DataFrame from user input
        input_data = pd.DataFrame([user_input])

        # --- Train the model with fixed parameters ---
        try:
            # Use the train/test split defined earlier
            # model
            model = LogisticRegression(
            random_state=999,
            penalty='l2',
            class_weight='balanced')
            model.fit(X_train, y_train) # Train on the training set
            st.success("Model trained successfully with fixed parameters!")

            # Apply the same preprocessing pipeline to input data
            input_processed = preprocessor.transform(input_data)

            # Make prediction using the newly trained model
            # Prediction probabilities
            prediction_proba = svc.predict_proba(input_processed)[0]
            
            # Display results
            st.header("Prediction Result")
            # Assuming class 1 is 'Acute Kidney Injury'
            prob_label = "Predicted Probability of Acute Kidney Injury"
            st.metric(label=prob_label, value=f"{prediction_proba[1]*100:.2f}%") # Displaying probability of class 1

        except Exception as e:
            st.error(f"An error occurred during model training or prediction: {e}")


# --- Disclaimer Section at the Bottom ---
st.markdown("---") # Horizontal line separator
disclaimer_text = """
**Disclaimer:**

Supplement:
*  AST : Peak AST value within the first 24 hours of ICU admission.
*  Cr : Peak Cr value within the first 24 hours of ICU admission.
*  GLU : Peak venous blood glucose level within the first 24 hours of ICU admission.
*  K+ : Peak serum potassium level within the first 24 hours of ICU admission.
*  D-Dimer : Peak D-Dimer level within the first 24 hours of ICU admission.
*  SOFA : SOFA within the first 24 hours of ICU admission.
*  pH : pH value from arterial blood gas obtained within 24 hours post-ICU admission.
*  PO2/FiO2 : PO2/FiO2 value from arterial blood gas obtained within 24 hours post-ICU admission.
*  24-hour fluid balance : The fluid balance (input-output) for the first 24 hours in the ICU.
"""
st.markdown(disclaimer_text)




