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

# Custom CSS for Dark Mode Professional UI
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-color: #0f172a;
        --card-bg: #1e293b;
        --sidebar-bg: #0b1120;
        --primary: #38bdf8;
        --secondary: #818cf8;
        --success: #34d399;
        --danger: #fb7185;
        --text-main: #f1f5f9;
        --text-muted: #94a3b8;
        --border: #334155;
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: var(--text-main);
        background-color: var(--bg-color);
    }
    
    /* Main Background */
    .stApp {
        background-color: var(--bg-color);
        background-image: 
            radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.1) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(129, 140, 248, 0.1) 0px, transparent 50%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border);
    }
    
    /* Headings */
    h1, h2, h3 {
        color: var(--text-main);
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    
    h1 {
        background: linear-gradient(to right, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        padding-bottom: 0.5rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: var(--text-muted);
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: var(--primary);
        border-bottom: 2px solid var(--primary);
    }
    
    /* Cards/Containers */
    .stForm {
        background-color: var(--card-bg);
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid var(--border);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input, 
    .stSelectbox > div > div > div {
        background-color: #0f172a;
        color: var(--text-main);
        border: 1px solid var(--border);
        border-radius: 0.5rem;
    }
    
    .stNumberInput > div > div > input:focus, 
    .stSelectbox > div > div > div:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3.5rem;
        font-weight: 600;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: #0f172a;
        border: none;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(56, 189, 248, 0.3);
    }
    
    /* Result Cards */
    .result-container {
        background-color: var(--card-bg);
        border-radius: 1rem;
        padding: 2rem;
        border: 1px solid var(--border);
        margin-top: 2rem;
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.875rem;
        margin-bottom: 1rem;
    }
    
    .risk-high {
        background-color: rgba(251, 113, 133, 0.2);
        color: var(--danger);
        border: 1px solid var(--danger);
    }
    
    .risk-low {
        background-color: rgba(52, 211, 153, 0.2);
        color: var(--success);
        border: 1px solid var(--success);
    }
    
    .metric-value {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    
    .recommendation-item {
        display: flex;
        align-items: start;
        gap: 1rem;
        padding: 1rem;
        background-color: rgba(15, 23, 42, 0.5);
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid var(--primary);
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
        st.error(f"Model file '{model_name}' not found. Please train the model first.")
        return None

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.markdown("### Model Configuration")
    model_choice = st.selectbox(
        "Select Algorithm",
        ["LogisticRegression", "XGBoostClassifier", "SVC"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Accuracy")
    
    acc_data = {
        "LogisticRegression": "69.3%",
        "XGBoostClassifier": "98.5%",
        "SVC": "100.0%"
    }
    
    best_params = {
        "LogisticRegression": {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'},
        "XGBoostClassifier": {'learning_rate': 0.08, 'max_depth': 2, 'n_estimators': 2000},
        "SVC": {'C': 10, 'gamma': 0.01}
    }
    
    st.metric("Model Accuracy (Test Set)", acc_data.get(model_choice, "N/A"))
    
    with st.expander("See Best Parameters"):
        st.json(best_params.get(model_choice, {}))

    st.markdown("---")
    st.markdown("### 📈 Dataset Statistics")
    st.markdown("""
    <div style="background-color: #1e293b; padding: 1rem; border-radius: 0.5rem; border: 1px solid #334155;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #94a3b8;">Total Samples:</span>
            <span style="font-weight: 700; color: #f1f5f9;">1025</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #94a3b8;">Training Set:</span>
            <span style="font-weight: 700; color: #34d399;">820</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #94a3b8;">Test Set:</span>
            <span style="font-weight: 700; color: #fb7185;">205</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("This tool is for educational purposes only. Always consult a healthcare professional.")

model = load_model(model_choice)

# Main Content
st.title("HeartGuard AI")
st.markdown("<p style='font-size: 1.2rem; color: #94a3b8; margin-bottom: 2rem;'>Advanced Cardiac Risk Assessment System</p>", unsafe_allow_html=True)

# Input Form with Tabs
with st.form("prediction_form"):
    
    tab1, tab2, tab3 = st.tabs(["👤 Patient Details", "❤️ Symptoms", "🔬 Clinical Tests"])
    
    with tab1:
        st.markdown("#### Demographics & Vitals")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 1, 120, 50)
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 120)
        with c2:
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
            
    with tab2:
        st.markdown("#### Cardiac Symptoms")
        c1, c2 = st.columns(2)
        with c1:
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
            thalach = st.number_input("Max Heart Rate Achieved", 50, 250, 150)
        with c2:
            exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, 0.1)

    with tab3:
        st.markdown("#### Test Results")
        c1, c2, c3 = st.columns(3)
        with c1:
            slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
            fbs = st.selectbox("Fasting BS > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with c2:
            ca = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])
            restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
        with c3:
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])

    st.markdown("###")
    submit_btn = st.form_submit_button("Analyze Patient Data", type="primary")

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
            prediction_proba = model.predict_proba(input_data) if hasattr(model, 'predict_proba') else None
            
            prob_value = prediction_proba[0][1] * 100 if prediction_proba is not None else 0
            is_high_risk = prediction[0] == 1
            
            # Result Container
            st.markdown("---")
            
            c1, c2 = st.columns([1, 1.5])
            
            with c1:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Probability", 'font': {'size': 20, 'color': '#94a3b8'}},
                    number={'font': {'size': 40, 'color': '#f1f5f9'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                        'bar': {'color': "#fb7185" if prob_value > 50 else "#34d399"},
                        'bgcolor': "#1e293b",
                        'borderwidth': 2,
                        'bordercolor': "#334155",
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(52, 211, 153, 0.1)"},
                            {'range': [50, 100], 'color': "rgba(251, 113, 133, 0.1)"}
                        ],
                    }
                ))
                fig.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#f1f5f9", 'family': "Outfit"}
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown(f"""
                <div class="result-container">
                    <div class="risk-badge {'risk-high' if is_high_risk else 'risk-low'}">
                        { '⚠️ High Risk Detected' if is_high_risk else '✅ Low Risk Detected' }
                    </div>
                    <div class="metric-value" style="color: { '#fb7185' if is_high_risk else '#34d399' }">
                        {prob_value:.1f}%
                    </div>
                    <p style="color: #94a3b8; font-size: 1.1rem;">
                        Probability of coronary artery disease based on the provided clinical parameters.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 📋 Clinical Recommendations")
                if is_high_risk:
                    st.markdown("""
                    <div class="recommendation-item">
                        <div style="font-size: 1.5rem;">🏥</div>
                        <div>
                            <div style="font-weight: 600; color: #f1f5f9;">Immediate Consultation</div>
                            <div style="color: #94a3b8; font-size: 0.9rem;">Refer patient to a cardiologist for comprehensive evaluation.</div>
                        </div>
                    </div>
                    <div class="recommendation-item">
                        <div style="font-size: 1.5rem;">🩺</div>
                        <div>
                            <div style="font-weight: 600; color: #f1f5f9;">Diagnostic Imaging</div>
                            <div style="color: #94a3b8; font-size: 0.9rem;">Recommended: Stress Echocardiography or Coronary Angiography.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="recommendation-item">
                        <div style="font-size: 1.5rem;">📅</div>
                        <div>
                            <div style="font-weight: 600; color: #f1f5f9;">Routine Monitoring</div>
                            <div style="color: #94a3b8; font-size: 0.9rem;">Schedule annual cardiac health screening.</div>
                        </div>
                    </div>
                    <div class="recommendation-item">
                        <div style="font-size: 1.5rem;">🥗</div>
                        <div>
                            <div style="font-weight: 600; color: #f1f5f9;">Lifestyle Maintenance</div>
                            <div style="color: #94a3b8; font-size: 0.9rem;">Encourage heart-healthy diet and regular physical activity.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
