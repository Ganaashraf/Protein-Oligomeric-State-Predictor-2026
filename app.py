import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Protein Assembly Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. LOAD MACHINE LEARNING ASSETS
# ==========================================
# We use @st.cache_resource so the app loads the model only once (makes it faster)
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('rf_protein_model.pkl')
        scaler = joblib.load('scaler.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        top_features = joblib.load('top_features.pkl')
        return model, scaler, target_encoder, top_features
    except FileNotFoundError:
        st.error("Error: Could not find the .pkl files. Make sure your notebook saved them in the same folder as app.py!")
        st.stop()

model, scaler, target_encoder, top_features = load_assets()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Predictor Tool", "About the Project"])

if page == "About the Project":
    st.title("🧬 Macromolecular Structure Project")
    st.write("""
    ### Project Overview
    This web application deploys a Machine Learning model built to predict the **Oligomeric State** (Monomer, Dimer, Multimer) of human proteins based on structural and biochemical data extracted from the RCSB Protein Data Bank.
    
    ### Model Details
    - **Algorithm:** Tuned Random Forest Classifier
    - **Feature Selection:** Embedded Method (Top 15 features based on importance)
    - **Dataset Size:** 11,832 records
    """)
    st.info("Navigate to the 'Predictor Tool' in the sidebar to test the model.")

# ==========================================
# 4. PREDICTOR TOOL PAGE
# ==========================================
elif page == "Predictor Tool":
    st.title("🔬 Protein Oligomeric State Predictor")
    st.write("Enter the structural and biochemical parameters below to predict how this protein will assemble.")
    
    st.markdown("---")

    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)
    user_inputs = {}
    
    st.sidebar.header("Adjust Parameters")
    
    # Split the input fields evenly between the two columns
    half_point = len(top_features) // 2
    
    for i, feature in enumerate(top_features):
        if i < half_point:
            with col1:
                user_inputs[feature] = st.number_input(f"Input: {feature}", value=0.0, format="%.3f")
        else:
            with col2:
                user_inputs[feature] = st.number_input(f"Input: {feature}", value=0.0, format="%.3f")
                
    st.markdown("---")

    # ==========================================
    # 5. PREDICTION LOGIC
    # ==========================================
    # Center the predict button
    col_empty1, col_btn, col_empty2 = st.columns([1, 1, 1])
    with col_btn:
        predict_button = st.button("🚀 Predict Oligomeric State", use_container_width=True)

    if predict_button:
        with st.spinner("Analyzing structural parameters..."):
            # Convert user inputs to a dataframe
            input_df = pd.DataFrame([user_inputs])
            
            # The scaler needs all original columns, so we create a dummy dataframe filled with 0s
            dummy_df = pd.DataFrame(np.zeros((1, len(scaler.feature_names_in_))), columns=scaler.feature_names_in_)
            
            # Update the dummy dataframe with the user's inputs
            for col in input_df.columns:
                if col in dummy_df.columns:
                    dummy_df[col] = input_df[col]
            
            # Apply the scaler
            scaled_input_full = scaler.transform(dummy_df)
            
            # Extract ONLY the top features needed for the model
            final_input = pd.DataFrame(scaled_input_full, columns=scaler.feature_names_in_)[top_features]
            
            # Make the Prediction
            prediction_encoded = model.predict(final_input)
            prediction_label = target_encoder.inverse_transform(prediction_encoded)[0]
            prediction_proba = model.predict_proba(final_input)[0]
            
            # ==========================================
            # 6. DISPLAY RESULTS
            # ==========================================
            st.success(f"### Predicted Assembly State: **{prediction_label.upper()}**")
            
            st.write("#### Confidence Probabilities:")
            
            # Create a bar chart for probabilities
            fig, ax = plt.subplots(figsize=(8, 3))
            classes = target_encoder.classes_
            
            sns.barplot(x=prediction_proba * 100, y=classes, palette='viridis', ax=ax)
            
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probability (%)')
            ax.set_ylabel('State')
            ax.set_title('Model Confidence per Class')
            
            # Add percentage text to bars
            for i, p in enumerate(ax.patches):
                ax.annotate(f'{p.get_width():.1f}%', 
                            (p.get_width() + 1, p.get_y() + p.get_height() / 2.), 
                            va='center')
                
            st.pyplot(fig)