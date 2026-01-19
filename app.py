# ML Resume Screening App with SHAP Explanations
# Using Streamlit for UI and Gemini for AI explanations

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai

# page config
st.set_page_config(page_title="Candidate Screening Predictor", page_icon="📄", layout="wide")

# Modern Professional UI with Teal/Cyan Theme - Dark Mode Optimized
st.markdown("""
<style>
/* Import Poppins - Modern Professional Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

/* CSS Variables for easy theming */
:root {
    --primary-gradient: linear-gradient(135deg, #0d9488 0%, #0891b2 50%, #06b6d4 100%);
    --accent-color: #f59e0b;
    --accent-gradient: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
    --bg-dark: #0f172a;
    --bg-card: rgba(15, 23, 42, 0.85);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border-color: rgba(148, 163, 184, 0.2);
    --success-color: #10b981;
    --error-color: #ef4444;
    --glow-teal: rgba(13, 148, 136, 0.4);
    --glow-amber: rgba(245, 158, 11, 0.3);
}

/* Global styles */
.stApp {
    font-family: 'Poppins', sans-serif;
    background: var(--bg-dark);
    background-image: 
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(13, 148, 136, 0.3), transparent),
        radial-gradient(ellipse 60% 40% at 100% 100%, rgba(8, 145, 178, 0.2), transparent),
        radial-gradient(ellipse 40% 30% at 0% 80%, rgba(245, 158, 11, 0.15), transparent);
    min-height: 100vh;
}

/* Main container styling */
.main .block-container {
    background: var(--bg-card);
    border-radius: 24px;
    padding: 2.5rem 3rem;
    margin: 1.5rem auto;
    border: 1px solid var(--border-color);
    box-shadow: 
        0 25px 50px -12px rgba(0, 0, 0, 0.5),
        0 0 80px rgba(13, 148, 136, 0.1);
    backdrop-filter: blur(20px);
    animation: slideUp 0.6s ease-out;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px var(--glow-teal); }
    50% { box-shadow: 0 0 40px var(--glow-teal), 0 0 60px var(--glow-teal); }
}

@keyframes popIn {
    from { opacity: 0; transform: scale(0.9); }
    to { opacity: 1; transform: scale(1); }
}

/* Title styling */
h1 {
    font-family: 'Poppins', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.8rem !important;
    text-align: center;
    background: linear-gradient(135deg, #0d9488 0%, #06b6d4 40%, #f59e0b 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s linear infinite, fadeIn 0.8s ease-out;
    text-shadow: none;
    letter-spacing: -0.02em;
}

/* Subheader styling */
.stSubheader, h2, h3 {
    font-family: 'Poppins', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 600;
    letter-spacing: -0.01em;
}

h2 {
    font-size: 1.5rem !important;
    margin-bottom: 1rem !important;
}

h3 {
    font-size: 1.2rem !important;
    color: #67e8f9 !important;
}

/* Regular text */
p, span, label, .stMarkdown {
    font-family: 'Poppins', sans-serif !important;
    color: var(--text-secondary) !important;
}

/* Card sections with glassmorphism */
.stColumn > div {
    background: rgba(30, 41, 59, 0.6);
    border-radius: 20px;
    padding: 1.5rem;
    border: 1px solid var(--border-color);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 1rem;
}

.stColumn > div:hover {
    border-color: rgba(13, 148, 136, 0.5);
    box-shadow: 0 8px 30px rgba(13, 148, 136, 0.2);
    transform: translateY(-2px);
}

/* Button styling - Premium */
.stButton > button {
    font-family: 'Poppins', sans-serif !important;
    background: var(--primary-gradient);
    color: white !important;
    border: none;
    border-radius: 16px;
    padding: 1rem 2.5rem;
    font-weight: 600;
    font-size: 1.15rem;
    letter-spacing: 0.02em;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 8px 25px var(--glow-teal);
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 15px 40px var(--glow-teal);
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:active {
    transform: translateY(-1px) scale(0.98);
}

/* Input fields styling */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextArea > div > div > textarea {
    font-family: 'Poppins', sans-serif !important;
    background: rgba(30, 41, 59, 0.8) !important;
    border-radius: 14px !important;
    border: 2px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    transition: all 0.3s ease;
}

.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within,
.stTextArea > div > div > textarea:focus {
    border-color: #0d9488 !important;
    box-shadow: 0 0 0 4px rgba(13, 148, 136, 0.2) !important;
}

/* Slider styling */
.stSlider > div > div > div {
    background: var(--primary-gradient) !important;
}

.stSlider > div > div > div > div {
    background: #f59e0b !important;
    box-shadow: 0 0 10px var(--glow-amber);
}

/* Checkbox styling */
.stCheckbox {
    font-family: 'Poppins', sans-serif !important;
}

.stCheckbox > label > div[data-testid="stCheckbox"] > div {
    background: rgba(30, 41, 59, 0.8);
    border-color: var(--border-color);
}

.stCheckbox > label > div[data-testid="stCheckbox"] > div[aria-checked="true"] {
    background: var(--primary-gradient) !important;
    border-color: #0d9488 !important;
}

/* Success box styling */
.stSuccess {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%) !important;
    border: 1px solid rgba(16, 185, 129, 0.4) !important;
    border-radius: 16px !important;
    animation: popIn 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
}

/* Error box styling */
.stError {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%) !important;
    border: 1px solid rgba(239, 68, 68, 0.4) !important;
    border-radius: 16px !important;
    animation: popIn 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.2);
}

/* Metric styling */
[data-testid="stMetricValue"] {
    font-family: 'Poppins', sans-serif !important;
    font-size: 2.5rem !important;
    font-weight: 700;
    background: linear-gradient(135deg, #0d9488 0%, #06b6d4 50%, #f59e0b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Info box styling */
.stInfo {
    background: linear-gradient(135deg, rgba(13, 148, 136, 0.15) 0%, rgba(6, 182, 212, 0.1) 100%) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(13, 148, 136, 0.3) !important;
    border-left: 4px solid #0d9488 !important;
}

.stInfo p {
    color: #67e8f9 !important;
}

/* Spinner animation */
.stSpinner > div {
    border-color: #0d9488 transparent #06b6d4 transparent !important;
}

/* Divider styling */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #0d9488, #f59e0b, transparent);
    margin: 2rem 0;
    opacity: 0.6;
}

/* Chart container */
.stPyplotChart {
    background: rgba(30, 41, 59, 0.8);
    border-radius: 20px;
    padding: 1.5rem;
    border: 1px solid var(--border-color);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    animation: slideUp 0.5s ease-out;
}

/* Multiselect tags */
.stMultiSelect span[data-baseweb="tag"] {
    font-family: 'Poppins', sans-serif !important;
    background: var(--primary-gradient) !important;
    border-radius: 10px !important;
    font-weight: 500;
    box-shadow: 0 2px 8px var(--glow-teal);
}

/* Smooth scroll */
html {
    scroll-behavior: smooth;
}

/* Footer styling */
.stCaption {
    font-family: 'Poppins', sans-serif !important;
    text-align: center;
    color: var(--text-secondary) !important;
    padding: 1.5rem;
    font-size: 0.9rem;
}

/* Label styling for inputs */
.stSelectbox label, .stMultiSelect label, .stSlider label, .stCheckbox label {
    font-family: 'Poppins', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

/* Dropdown menu styling */
[data-baseweb="popover"] {
    background: rgba(30, 41, 59, 0.95) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px);
}

[data-baseweb="menu"] {
    background: transparent !important;
}

[data-baseweb="menu"] li {
    font-family: 'Poppins', sans-serif !important;
    color: var(--text-primary) !important;
    transition: background 0.2s ease;
}

[data-baseweb="menu"] li:hover {
    background: rgba(13, 148, 136, 0.2) !important;
}

/* Section card helper class */
.section-card {
    background: rgba(30, 41, 59, 0.6);
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid var(--border-color);
    margin-bottom: 1.5rem;
}

/* Expander styling */
.streamlit-expanderHeader {
    font-family: 'Poppins', sans-serif !important;
    background: rgba(30, 41, 59, 0.6) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)

# gemini api key
genai.configure(api_key='AIzaSyBo3MKncRbB0vn3K0R1KMGHb4aISyCuZTs')

# load model and encoders
@st.cache_resource
def load_model():
    model = joblib.load('resume_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('degree_encoder.pkl')
    return model, tfidf, le

model, tfidf, le = load_model()

# all available skills matching the TF-IDF features (50 total)
all_skills = [
    'Python', 'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch',
    'Scikit-learn', 'Pandas', 'NumPy', 'SQL', 'AWS', 'GCP', 'Azure',
    'Docker', 'Docker Compose', 'Kubernetes', 'CI/CD', 'Git', 'Linux',
    'NLP', 'GPT', 'HuggingFace', 'Transformer Models', 'BERT',
    'Computer Vision', 'OpenCV', 'Speech Recognition',
    'Flask', 'FastAPI', 'Streamlit', 'REST APIs', 'Model Deployment',
    'Spark', 'Hadoop', 'Tableau', 'Power BI', 'Data Visualization',
    'Statistics', 'Regression', 'Classification', 'Clustering', 'EDA',
    'Feature Engineering', 'XGBoost', 'LightGBM', 'Lightning',
    'Matplotlib', 'Seaborn', 'SQLAlchemy', 'JSON',
    'Reinforcement Learning', 'Time Series', 'Signal Processing'
]

# job title options
job_titles = [
    'None/Fresher', 'Data Scientist', 'ML Engineer', 'Machine Learning Engineer',
    'AI Researcher', 'Research Scientist', 'NLP Engineer', 'Computer Vision Engineer',
    'Data Analyst', 'Applied ML Engineer', 'Senior Data Scientist', 'Other Non-ML Role'
]

# which titles count as ML roles
ml_roles = [
    'Data Scientist', 'ML Engineer', 'Machine Learning Engineer', 'AI Researcher',
    'Research Scientist', 'NLP Engineer', 'Computer Vision Engineer', 'Data Analyst',
    'Applied ML Engineer', 'Senior Data Scientist'
]

# normalize skills - handle abbreviations
def normalize_skills(text):
    mapping = {
        'ml': 'Machine Learning', 'dl': 'Deep Learning', 'nlp': 'NLP',
        'cv': 'Computer Vision', 'tf': 'TensorFlow', 'pytorch': 'PyTorch',
        'sklearn': 'Scikit-learn', 'aws': 'AWS', 'gcp': 'GCP', 'k8s': 'Kubernetes',
        'sql': 'SQL', 'bert': 'BERT', 'gpt': 'GPT', 'eda': 'EDA',
        'bi': 'Power BI', 'tableau': 'Tableau'
    }
    
    skills = [s.strip() for s in text.split(',')]
    result = []
    for skill in skills:
        lower = skill.lower().strip()
        if lower in mapping:
            result.append(mapping[lower])
        else:
            result.append(skill.strip().title())
    return ', '.join(result)

# gemini explanation function using shap values
def get_explanation(pred, prob, contributions, shap_values_raw):
    try:
        gemini = genai.GenerativeModel('models/gemini-2.5-flash')
        
        result = "SCREENED IN (Suitable for ML Role)" if pred == 1 else "SCREENED OUT (Not Suitable)"
        
        # map feature names to human-readable descriptions
        feature_descriptions = {
            'is_ml_title': 'Not having current ML/Data role' if contributions.get('is_ml_title', 0) <= 0 else 'Having current ML/Data role',
            'years_experience': 'Years of experience',
            'degree_encoded': 'Education level',
            'has_portfolio': 'No portfolio/GitHub' if contributions.get('has_portfolio', 0) <= 0 else 'Having portfolio/GitHub',
        }
        
        # format shap contributions with percentage
        positive_factors = []
        negative_factors = []
        for name, val in contributions.items():
            pct = abs(val) * 100
            # use human-readable name if available
            display_name = feature_descriptions.get(name, f"Skill: {name.title()}")
            if val > 0:
                positive_factors.append(f"- {display_name}: contributed +{pct:.1f}% towards selection")
            else:
                negative_factors.append(f"- {display_name}: reduced chances by -{pct:.1f}%")
        
        pos_text = "\n".join(positive_factors[:5]) if positive_factors else "None"
        neg_text = "\n".join(negative_factors[:5]) if negative_factors else "None"
        
        prompt = f"""Based on SHAP (SHapley Additive exPlanations) analysis, explain why this resume was predicted as: {result}

Confidence: {prob*100:.1f}%

**Positive Factors (helped the prediction):**
{pos_text}

**Negative Factors (hurt the prediction):**
{neg_text}

Write a clear explanation:
1. Start with overall verdict (1-2 sentences)
2. Explain which features helped and by how much (use the percentage values)
3. Explain which features hurt and by how much
4. Give 2-3 specific recommendations to improve

Use bullet points. Be specific about percentages. Keep it professional."""

        response = gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# create shap bar plot - dark theme with teal/amber colors
def make_shap_plot(shap_vals, features):
    # Set dark style for the plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1e293b')
    ax.set_facecolor('#1e293b')
    
    # get shap values for positive class
    if hasattr(shap_vals, 'shape') and len(shap_vals.shape) == 3:
        sv = shap_vals[0, :, 1]
    elif isinstance(shap_vals, list) and len(shap_vals) == 2:
        sv = shap_vals[1][0]
    else:
        sv = np.array(shap_vals).flatten()[:len(features)]
    
    # top 10 features only
    idx = np.argsort(np.abs(sv))[::-1][:10]
    raw_names = [features[i] for i in idx]
    vals = [sv[i] for i in idx]
    
    # map to human-readable names
    def get_display_name(name, val):
        if name == 'is_ml_title':
            return 'No ML Role' if val <= 0 else 'Has ML Role'
        elif name == 'has_portfolio':
            return 'No Portfolio' if val <= 0 else 'Has Portfolio'
        elif name == 'years_experience':
            return 'Years of Experience'
        elif name == 'degree_encoded':
            return 'Education Level'
        else:
            return f'Skill: {name.title()}'
    
    names = [get_display_name(raw_names[i], vals[i]) for i in range(len(raw_names))]
    # Teal for positive, coral/red for negative - matching the new theme
    colors = ['#0d9488' if v > 0 else '#ef4444' for v in vals]
    
    bars = plt.barh(range(len(names)), vals, color=colors, alpha=0.9, edgecolor='white', linewidth=0.5)
    
    # Style the axes
    plt.yticks(range(len(names)), names, fontsize=10, color='#f1f5f9', fontfamily='sans-serif')
    plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=11, color='#94a3b8', fontfamily='sans-serif')
    plt.title('Top Features Affecting Prediction', fontsize=13, color='#f1f5f9', fontweight='bold', fontfamily='sans-serif', pad=15)
    
    # Style spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.tick_params(colors='#94a3b8', which='both')
    
    # Add a subtle grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#475569')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return fig


# ============== MAIN APP ==============

st.title("📄 Candidate Screening Predictor")
st.markdown("""
<div style='text-align: left; margin: 0.5rem 0 0.5rem 0;'>
    <p style='font-size: 1.4rem; font-weight: 700; color: #f1f5f9; 
              background: linear-gradient(135deg, #0d9488 0%, #06b6d4 50%, #f59e0b 100%);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent;
              background-clip: text; letter-spacing: 0.02em;'>
        Check if your resume will be screened IN or OUT for AI/ML roles
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Input section with professional layout
st.markdown("""
<div style='text-align: left; margin-bottom: 1rem;'>
    <p style='color: #94a3b8; font-size: 1.1rem;'>
        Fill in your details below and let our AI analyze your resume fit for AI/ML roles
    </p>
</div>
""", unsafe_allow_html=True)

# Two column layout for main inputs
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🎓 Education & Experience")
    
    # years of exp
    years_exp = st.slider("Years of Experience", 0, 15, 3, help="Total years of relevant work experience")
    
    # degree
    degree = st.selectbox("Highest Degree", ['Bachelors', 'Masters', 'PhD'])

with col2:
    st.markdown("### 💼 Current Role")
    
    # current title
    title = st.selectbox("Current Job Title", job_titles, help="Select your current or most recent position")
    
    # portfolio
    portfolio = st.checkbox("✓ I have portfolio/GitHub", help="Check if you have an online portfolio or GitHub profile")

# Pro Tips section - centered below inputs
st.markdown("""
<div style='background: linear-gradient(135deg, rgba(13, 148, 136, 0.15) 0%, rgba(6, 182, 212, 0.1) 100%);
            border: 1px solid rgba(13, 148, 136, 0.4); border-left: 4px solid #f59e0b;
            border-radius: 16px; padding: 1.5rem; margin: 1.5rem auto; max-width: 600px; text-align: center;'>
    <p style='color: #f59e0b; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;'>💡 Pro Tips to Maximize Your Score</p>
    <p style='color: #94a3b8; font-size: 0.95rem; margin: 0;'>
        ✦ Select all relevant skills &nbsp;&nbsp; ✦ Portfolio/GitHub boosts visibility &nbsp;&nbsp; ✦ ML role adds credibility
    </p>
</div>
""", unsafe_allow_html=True)

# Skills section - full width with enhanced visibility
st.markdown("---")
st.markdown("""
<div style='text-align: left; margin-bottom: 1rem;'>
    <p style='font-size: 1.3rem; font-weight: 700; color: #0d9488; margin-bottom: 0.5rem;'>🛠️ Technical Skills</p>
    <p style='color: #67e8f9; font-size: 1rem;'>Select all the skills you possess from the list below</p>
</div>
""", unsafe_allow_html=True)

skills_selected = st.multiselect(
    "Select Your Skills",
    options=all_skills,
    default=['Python', 'Machine Learning'],
    help="Select all skills you have - the more relevant skills, the better!",
    label_visibility="collapsed"
)

st.markdown("---")

# predict button
st.markdown("""
<style>
    .stButton > button {
        font-size: 1.3rem !important;
        padding: 1.2rem 2rem !important;
        animation: pulse-glow 2s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)
if st.button("🚀 Get My AI/ML Screening Result", type="primary", use_container_width=True):
    with st.spinner("Analyzing..."):
        
        # prepare features - join selected skills
        skills_text = ', '.join(skills_selected)
        deg_enc = le.transform([degree])[0]
        port = 1 if portfolio else 0
        is_ml = 1 if title in ml_roles else 0
        
        # tfidf
        skills_vec = tfidf.transform([skills_text]).toarray()
        skills_df = pd.DataFrame(skills_vec, columns=tfidf.get_feature_names_out())
        
        # combine all features
        input_df = pd.DataFrame({
            'years_experience': [years_exp],
            'degree_encoded': [deg_enc],
            'has_portfolio': [port],
            'is_ml_title': [is_ml]
        })
        X = pd.concat([input_df.reset_index(drop=True), skills_df], axis=1)
        
        # predict
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        # shap values
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        
        # get values for positive class
        feature_names = X.columns.tolist()
        if hasattr(shap_vals, 'shape') and len(shap_vals.shape) == 3:
            sv = shap_vals[0, :, 1]
        elif isinstance(shap_vals, list) and len(shap_vals) == 2:
            sv = shap_vals[1][0]
        else:
            sv = np.array(shap_vals).flatten()[:len(feature_names)]
        
        # calculate contributions
        total = np.sum(np.abs(sv))
        contributions = {}
        for i, name in enumerate(feature_names):
            contributions[name] = sv[i] / total if total > 0 else 0
        
        # get top 10
        contributions = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
    
    # show results
    st.markdown("---")
    st.subheader("Prediction Result")
    
    c1, c2 = st.columns(2)
    
    with c1:
        if pred == 1:
            st.success("## ✅ SCREENED IN!")
            st.metric("Confidence", f"{proba[1]*100:.1f}%")
        else:
            st.error("## ❌ SCREENED OUT")
            st.metric("Confidence", f"{proba[0]*100:.1f}%")
    
    with c2:
        st.markdown("### Feature Contributions")
        for name, val in list(contributions.items())[:5]:
            display = name.replace('_', ' ').title() if any(x in name for x in ['exp', 'degree', 'portfolio', 'ml']) else f"Skill: {name}"
            sign = "🟢 +" if val > 0 else "� -"
            st.markdown(f"- **{display}**: {sign}{abs(val)*100:.1f}%")
    
    # explanation using gemini (moved above graph)
    st.markdown("---")
    st.subheader("📝 Why This Prediction?")
    
    with st.spinner("Generating explanation using SHAP values..."):
        conf = proba[1] if pred == 1 else proba[0]
        explanation = get_explanation(pred, conf, contributions, sv)
    st.markdown(explanation)
    
    # shap plot (moved below explanation, smaller)
    st.markdown("---")
    st.subheader("📊 Feature Impact Chart")
    fig = make_shap_plot(shap_vals, feature_names)
    st.pyplot(fig)

# footer
st.markdown("---")
st.caption("Built with Streamlit, SHAP, and Gemini AI | Model Accuracy: 99.89%")
