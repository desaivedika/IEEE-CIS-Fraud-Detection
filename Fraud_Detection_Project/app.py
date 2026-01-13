import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Setup
st.set_page_config(page_title="IEEE Fraud System", page_icon="🛡️", layout="wide")

# 2. Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('fraud_model_v1.pkl')
    cols = joblib.load('model_features.pkl')
    return model, cols

model, cols = load_assets()

# 3. Sidebar - Settings and Debugging
st.sidebar.title("🛡️ System Settings")
st.sidebar.markdown("Use this slider to adjust how 'sensitive' the fraud alert is.")
threshold = st.sidebar.slider("Fraud Alert Threshold", 0.0, 1.0, 0.05, help="If the probability is higher than this, show RED.")

st.sidebar.write("---")
show_debug = st.sidebar.checkbox("Show Technical Debug Info", value=True)

# 4. Main Interface
st.title("🛡️ IEEE-CIS Fraud Detection Dashboard")
st.markdown("### Real-Time Transaction Risk Assessment")

with st.container():
    st.write("Enter the details from the transaction below:")
    col1, col2 = st.columns(2)
    
    with col1:
        amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
    with col2:
        card1 = st.number_input("Card ID (card1)", min_value=0, value=1234)

# 5. Run Prediction
if st.button("Analyze Transaction", use_container_width=True):
    # Prepare data
    input_df = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
    input_df['TransactionAmt'] = amt
    input_df['card1'] = card1
    
    # Get Probability
    prob = model.predict_proba(input_df)[0][1]
    
    # 6. Display Results
    st.write("---")
    
    # Create three columns for metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Status", "Fraud Check Complete")
    m2.metric("Risk Score", f"{prob:.2%}")
    m3.metric("Threshold", f"{threshold:.2%}")

    # The Visual Progress Bar
    st.write("**Risk Level Indicator:**")
    st.progress(prob)

    # The Logic for Red/Green
    if prob >= threshold:
        st.error(f"🚨 FRAUD ALERT: High Risk Detected!")
        st.markdown(f"**Action:** Flagged for manual review. This transaction has a **{prob:.2%}** probability of being fraudulent.")
    else:
        st.success(f"✅ TRANSACTION APPROVED: Low Risk")
        st.markdown(f"**Action:** Proceed with payment. Risk is below the {threshold:.2%} safety limit.")

    # Technical Debugging section
    if show_debug:
        with st.expander("See Technical Data"):
            st.write(f"Raw Model Output (Class 1 Probability): `{prob}`")
            st.write("Features used for prediction:")
            st.write(input_df[['TransactionAmt', 'card1']])