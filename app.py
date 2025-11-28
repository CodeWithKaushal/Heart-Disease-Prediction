import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="HeartGuard AI | Advanced Cardiac Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #2563eb;
        --secondary-color: #10b981;
        --danger-color: #ef4444;
        --background-color: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
        background-color: var(--background-color);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stApp {
        background-color: var(--background-color);
        background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
        background-size: 20px 20px;
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--card-bg);
        border-right: 1px solid #e2e8f0;
        box-shadow: 4px 0 24px rgba(0,0,0,0.02);
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: 1.5rem;
        background: none;
        -webkit-text-fill-color: var(--text-primary);
    }
    
    /* Headings */
    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    
    h1 {
        font-size: 2.5rem; 
        margin-bottom: 1rem;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 { font-size: 1.5rem; margin-bottom: 0.75rem; }
    h3 { font-size: 1.25rem; margin-bottom: 0.5rem; }
    
    /* Cards/Containers */
    .stForm {
        background-color: var(--card-bg);
        padding: 2.5rem;
        border-radius: 1.5rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.01);
        border: 1px solid #f1f5f9;
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input, 
    .stSelectbox > div > div > div {
        border-radius: 0.75rem;
        border-color: #cbd5e1;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
        background-color: #f8fafc;
    }
    
    .stNumberInput > div > div > input:focus, 
    .stSelectbox > div > div > div:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.1);
        background-color: #ffffff;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 0.75rem;
        height: 3.5rem;
        font-weight: 600;
        background: linear-gradient(135deg, var(--primary-color), #1d4ed8);
        color: white;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.875rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
    }
    
    /* Metrics & Info */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .stAlert {
        border-radius: 0.75rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Custom Classes */
    .section-header {
        color: var(--text-secondary);
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-header span {
        background-color: #eff6ff;
        color: var(--primary-color);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .section-header::after {
        content: "";
        flex: 1;
        height: 2px;
        background: linear-gradient(to right, #e2e8f0, transparent);
    }
    
    .result-card {
        padding: 2rem;
        border-radius: 1rem;
        margin-top: 1rem;
        position: relative;
        overflow: hidden;
        animation: fadeIn 0.5s ease-out;
    }
    
    .high-risk {
        background: linear-gradient(135deg, #fff5f5 0%, #fff1f2 100%);
        border: 1px solid #fecdd3;
    }
    
    .low-risk {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #bbf7d0;
    }
    
    .recommendation-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #f1f5f9;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        animation: fadeIn 0.6s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model


@st.cache_resource
def load_model(model_name):
    try:
        with open(model_name, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(
            f"Model file '{model_name}' not found. Please train the model first.")
        return None


# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")

    st.markdown("### Model Selection")
    model_choice = st.selectbox(
        "Choose Prediction Model",
        ["LogisticRegression", "XGBoostClassifier", "SVC"],
        help="Select the machine learning algorithm for analysis",
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### ℹ️ System Info")
    st.info("""
    **HeartGuard AI v2.0**
    
    Advanced clinical decision support system utilizing ensemble machine learning for cardiac risk stratification.
    
    **Model Performance (AUC):**
    - Logistic Regression: 0.85
    - XGBoost: 0.83
    - SVC: 0.84
    """)

    st.markdown("---")
    st.caption("© 2025 HeartGuard Medical Systems. For investigational use only.")

model = load_model(model_choice)

# Main Content
st.title("🫀 HeartGuard AI")
st.markdown("### Clinical Risk Assessment Dashboard")
st.markdown(
    "Enter patient clinical parameters below to generate a comprehensive risk assessment report.")

# Create a form for inputs
with st.form("prediction_form"):
    # Group 1: Personal & Vitals
    st.markdown('<div class="section-header"><span>01</span> Patient Demographics & Vitals</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        age = st.number_input("Age", 1, 120, 50)
    with c2:
        sex = st.selectbox(
            "Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    with c3:
        trestbps = st.number_input("Resting BP (mm Hg)", 50, 250, 120)
    with c4:
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

    st.markdown("")

    # Group 2: Cardiac Symptoms
    st.markdown('<div class="section-header"><span>02</span> Cardiac Symptoms & History</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
        exang = st.selectbox("Exercise Induced Angina", [
                             0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with c2:
        thalach = st.number_input("Max Heart Rate", 50, 250, 150)
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, 0.1)

    with c3:
        slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: [
                             "Upsloping", "Flat", "Downsloping"][x])
        ca = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])

    st.markdown("")

    # Group 3: Other Tests
    st.markdown('<div class="section-header"><span>03</span> Other Clinical Tests</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [
                           0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with c2:
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2],
                               format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
    with c3:
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                            format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])

    st.markdown("")

    # Submit Button
    submit_btn = st.form_submit_button(
        "Generate Risk Assessment", type="primary")

# Prediction Logic
if submit_btn:
    if model is not None:
        input_data = pd.DataFrame({
            'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
            'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
            'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
        })

        try:
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(
                input_data) if hasattr(model, 'predict_proba') else None

            # Results Display
            st.markdown("### 📊 Analysis Results")

            r1, r2 = st.columns([1, 2])

            with r1:
                # Gauge Chart
                if prediction_proba is not None:
                    prob_disease = prediction_proba[0][1]
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_disease * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Risk Probability",
                               'font': {'size': 24, 'color': '#1e293b'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
                            'bar': {'color': "#ef4444" if prob_disease > 0.5 else "#10b981"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#e2e8f0",
                            'steps': [
                                {'range': [0, 50], 'color': "#f0fdf4"},
                                {'range': [50, 100], 'color': "#fff1f2"}],
                        }
                    ))
                    fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={'color': "#1e293b", 'family': "Inter"}
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with r2:
                st.markdown("<br>", unsafe_allow_html=True)
                if prediction[0] == 1:
                    st.markdown("""
                    <div class="result-card high-risk">
                        <h3 style="color: #991b1b; margin-top: 0;">⚠️ High Risk Detected</h3>
                        <p style="font-size: 1.1rem; color: #7f1d1d;">The analysis indicates a <strong>high probability</strong> of coronary artery disease based on the provided clinical parameters.</p>
                    </div>
                    
                    <div class="recommendation-card">
                        <h4 style="margin-top: 0; color: #1e293b;">🩺 Clinical Recommendations</h4>
                        <ul style="color: #475569; margin-bottom: 0;">
                            <li><strong>Immediate Consultation:</strong> Schedule an appointment with a cardiologist.</li>
                            <li><strong>Diagnostic Tests:</strong> Consider further tests like ECG, Echocardiogram, or Stress Test.</li>
                            <li><strong>Lifestyle Changes:</strong> Monitor diet, reduce stress, and avoid smoking/alcohol.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-card low-risk">
                        <h3 style="color: #065f46; margin-top: 0;">✅ Low Risk Detected</h3>
                        <p style="font-size: 1.1rem; color: #064e3b;">The analysis indicates a <strong>low probability</strong> of coronary artery disease based on the provided clinical parameters.</p>
                    </div>
                    
                    <div class="recommendation-card">
                        <h4 style="margin-top: 0; color: #1e293b;">🛡️ Preventive Measures</h4>
                        <ul style="color: #475569; margin-bottom: 0;">
                            <li><strong>Regular Checkups:</strong> Continue with annual health screenings.</li>
                            <li><strong>Healthy Diet:</strong> Maintain a balanced diet rich in fruits, vegetables, and whole grains.</li>
                            <li><strong>Physical Activity:</strong> Aim for at least 150 minutes of moderate exercise per week.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Please ensure all inputs are valid and try again.")
