import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load and preprocess training data
@st.cache_data
def load_and_preprocess_data():
    train = pd.read_csv('train.csv')
    
    # Clean up any leading/trailing whitespace in column names
    train.columns = train.columns.str.strip()
    
    # Handle the target variable first
    if 'NObeyesdad' in train.columns:
        # Clean target variable values
        train['NObeyesdad'] = train['NObeyesdad'].str.strip()
        
        # Create consistent label mapping
        obesity_mapping = {
            'Insufficient_Weight': 0,
            'Normal_Weight': 1,
            'Overweight_Level_I': 2,
            'Overweight_Level_II': 3,
            'Obesity_Type_I': 4,
            'Obesity_Type_II': 5,
            'Obesity_Type_III': 6
        }
        
        # Map target variable
        train['NObeyesdad'] = train['NObeyesdad'].map(obesity_mapping)
    
    # Define all categorical mappings
    categorical_mappings = {
        'family_history_with_overweight': {'yes': 1, 'no': 0},
        'FAVC': {'yes': 1, 'no': 0},
        'SMOKE': {'yes': 1, 'no': 0},
        'SCC': {'yes': 1, 'no': 0},
        'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
        'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
        'MTRANS': {
            'Public_Transportation': 0,
            'Automobile': 1,
            'Walking': 2,
            'Bike': 3,
            'Motorbike': 4
        }
    }
    
    # Apply categorical mappings
    for col, mapping in categorical_mappings.items():
        if col in train.columns:
            train[col] = train[col].str.strip().map(mapping).fillna(0).astype(int)
    
    # Handle gender (one-hot encoding)
    if 'Gender' in train.columns:
        train['Gender'] = train['Gender'].str.strip()
        train['Gender_Female'] = (train['Gender'] == 'Female').astype(int)
        train['Gender_Male'] = (train['Gender'] == 'Male').astype(int)
        train.drop('Gender', axis=1, inplace=True)
    
    # Drop ID column if exists
    if 'id' in train.columns:
        train.drop('id', axis=1, inplace=True)
    
    return train

train_df = load_and_preprocess_data()

# Train model
@st.cache_resource
def train_model():
    X = train_df.drop('NObeyesdad', axis=1)
    y = train_df['NObeyesdad']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# Obesity class mapping (consistent with training data preprocessing)
obesity_classes = {
    0: 'Insufficient Weight',
    1: 'Normal Weight',
    2: 'Overweight Level I',
    3: 'Overweight Level II',
    4: 'Obesity Type I',
    5: 'Obesity Type II',
    6: 'Obesity Type III'
}

# Streamlit app
st.title("Obesity Risk Prediction")
st.markdown("Predict your obesity risk based on lifestyle and health factors")

# Input form
with st.form("prediction_form"):
    st.header("Personal Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=14, max_value=100, value=30)
        height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.7, step=0.01)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        gender = st.selectbox("Gender", ["Female", "Male"])
        family_history = st.selectbox("Family History with Overweight", ["no", "yes"])
        favc = st.selectbox("Frequent High Caloric Food Consumption", ["no", "yes"])
    
    with col2:
        smoke = st.selectbox("Do you smoke?", ["no", "yes"])
        scc = st.selectbox("Calories Consumption Monitoring", ["no", "yes"])
        caec = st.selectbox("Food Consumption Between Meals", ["no", "Sometimes", "Frequently", "Always"])
        calc = st.selectbox("Alcohol Consumption", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Transportation Used", ["Public_Transportation", "Automobile", "Walking", "Bike", "Motorbike"])
    
    st.header("Physical Activity and Eating Habits")
    faf = st.slider("Physical Activity Frequency (days/week)", 0, 7, 3)
    tue = st.slider("Time Using Technology Devices (hours/day)", 0, 24, 2)
    fcvc = st.slider("Vegetable Consumption Frequency (1-3)", 1, 3, 2)
    ncp = st.slider("Number of Main Meals Daily", 1, 5, 3)
    ch2o = st.slider("Daily Water Consumption (liters)", 0.0, 5.0, 1.5, step=0.5)
    
    submit_button = st.form_submit_button("Predict Obesity Risk")

# Prediction
if submit_button:
    # Create input DataFrame with raw values
    input_data = pd.DataFrame({
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'Gender': [gender],
        'family_history_with_overweight': [family_history],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'CAEC': [caec],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'SCC': [scc],
        'FAF': [faf],
        'TUE': [tue],
        'CALC': [calc],
        'MTRANS': [mtrans]
    })
    
    # Preprocess the input data
    def preprocess_input(input_df):
        # Apply the same transformations as training data
        processed = input_df.copy()
        
        # Categorical mappings
        categorical_mappings = {
            'family_history_with_overweight': {'yes': 1, 'no': 0},
            'FAVC': {'yes': 1, 'no': 0},
            'SMOKE': {'yes': 1, 'no': 0},
            'SCC': {'yes': 1, 'no': 0},
            'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
            'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
            'MTRANS': {
                'Public_Transportation': 0,
                'Automobile': 1,
                'Walking': 2,
                'Bike': 3,
                'Motorbike': 4
            }
        }
        
        for col, mapping in categorical_mappings.items():
            if col in processed.columns:
                processed[col] = processed[col].map(mapping).fillna(0).astype(int)
        
        # Handle gender
        if 'Gender' in processed.columns:
            processed['Gender_Female'] = (processed['Gender'] == 'Female').astype(int)
            processed['Gender_Male'] = (processed['Gender'] == 'Male').astype(int)
            processed.drop('Gender', axis=1, inplace=True)
        
        # Ensure all expected columns exist
        expected_columns = model.feature_names_in_
        for col in expected_columns:
            if col not in processed.columns:
                processed[col] = 0  # Add missing columns with default value
        
        return processed[expected_columns]
    
    try:
        processed_input = preprocess_input(input_data)
        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)
        
        # Display results
        st.subheader("Prediction Results")
        predicted_class = prediction[0]
        class_name = obesity_classes.get(predicted_class, "Unknown")
        st.metric("Predicted Obesity Class", class_name)
        
        # Show probabilities
        st.write("Class Probabilities:")
        proba_df = pd.DataFrame({
            'Class': [obesity_classes.get(i, f"Unknown {i}") for i in range(7)],
            'Probability': [f"{p*100:.1f}%" for p in prediction_proba[0]]
        }).sort_values('Probability', ascending=False)
        st.table(proba_df)
        
        # BMI calculation
        bmi = weight / (height ** 2)
        st.write(f"Your BMI: {bmi:.1f}")
        
        # Interpretation
        st.subheader("Health Interpretation")
        if predicted_class == 0:
            st.info("Underweight: Consider consulting a nutritionist for dietary advice.")
        elif predicted_class == 1:
            st.success("Normal Weight: Maintain your healthy habits!")
        elif predicted_class in [2, 3]:
            st.warning("Overweight: Consider increasing physical activity and improving diet.")
        else:
            st.error("Obese: Please consult a healthcare professional for guidance.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.write("Please check your input values and try again.")

# Add info sidebar
st.sidebar.markdown("""
### About This App
This tool predicts obesity risk using machine learning based on:
- Body measurements
- Lifestyle factors
- Eating habits
- Physical activity patterns

**Note:** For educational purposes only. Consult a healthcare professional for medical advice.
""")