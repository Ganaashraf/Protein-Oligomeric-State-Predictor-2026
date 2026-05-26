import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Protein Assembly Predictor", page_icon="🧬", layout="wide")

# ==========================================
# 2. LOAD MACHINE LEARNING ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('rf_protein_model.pkl')
        scaler = joblib.load('scaler.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        top_features = joblib.load('top_features.pkl')
        return model, scaler, target_encoder, top_features
    except Exception as e:
        st.error(f"Missing Files Error: {e}")
        st.stop()

model, scaler, target_encoder, top_features = load_assets()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Predictor Tool", "About the Project"])

if page == "About the Project":
    st.title("🧬 Macromolecular Structure Project")
    st.write("This web application deploys a Machine Learning model built to predict the **Oligomeric State** of human proteins.")
    st.info("Navigate to the 'Predictor Tool' in the sidebar to test the model.")

# ==========================================
# 4. PREDICTOR TOOL PAGE
# ==========================================
elif page == "Predictor Tool":
    st.title("🔬 Protein Oligomeric State Predictor")
    st.markdown("---")

    col1, col2 = st.columns(2)
    user_inputs = {}
    
    st.sidebar.header("Adjust Parameters")
    
    # Generate input fields safely
    half = len(top_features) // 2
    for i, feature in enumerate(top_features):
        if i < half:
            with col1:
                user_inputs[feature] = st.number_input(f"{feature}", value=10.0)
        else:
            with col2:
                user_inputs[feature] = st.number_input(f"{feature}", value=10.0)
                
    st.markdown("---")

    # ==========================================
    # 5. PREDICTION LOGIC
    # ==========================================
    if st.button("🚀 Predict Oligomeric State", use_container_width=True):
        with st.spinner("Analyzing parameters..."):
            
            # Format inputs
            input_df = pd.DataFrame([user_inputs])
            
            # Scale inputs safely
            scaled_input = scaler.transform(input_df)
            final_input = pd.DataFrame(scaled_input, columns=input_df.columns)
            
            # Make prediction
            pred_encoded = model.predict(final_input)
            pred_label = target_encoder.inverse_transform(pred_encoded)[0]
            pred_proba = model.predict_proba(final_input)[0]
            
            st.success(f"### Predicted State: **{pred_label.upper()}**")
            
            # Graph
            fig, ax = plt.subplots(figsize=(6, 2))
            sns.barplot(x=pred_proba * 100, y=target_encoder.classes_, palette='viridis', ax=ax)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probability (%)')
            st.pyplot(fig)