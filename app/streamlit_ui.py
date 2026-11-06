# app/streamlit_ui.py
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path: 
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from src.utils import setup_logger

ui_logger = setup_logger("streamlit_client_app")

st.set_page_config(
    page_title="Early ICU Outcome Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'assets_verified' not in st.session_state:
    with st.spinner("Connecting to the prediction API..."):
        try:
            check = requests.get("http://127.0.0.1:8000/health", timeout=2.0)
            st.session_state['assets_verified'] = True
        except Exception as e:
            st.session_state['assets_verified'] = False
            ui_logger.error(f"API connection failed: {str(e)}")

st.title("Early ICU Outcome Analysis")
st.markdown(
    "Enter a small set of patient observations to view the model's estimated "
    "mortality risk and the main factors associated with the estimate."
)
st.write("---")

if not st.session_state.get('assets_verified', False):
    st.error("The prediction API is not running. Start it with `uvicorn api.main:app --port 8000`.")
    st.stop()

# Initialize session state for patient presets
if 'preset_age' not in st.session_state:
    st.session_state['preset_age'] = 65
    st.session_state['preset_gender'] = "Male"
    st.session_state['preset_hr'] = 88.0
    st.session_state['preset_gcs'] = 12.0
    st.session_state['preset_sysbp'] = 115.0
    st.session_state['preset_temp'] = 37.2
    st.session_state['preset_bun_active'] = False
    st.session_state['preset_bun'] = 20.0
    st.session_state['preset_creat_active'] = False
    st.session_state['preset_creat'] = 1.0

with st.sidebar.expander("Quick Clinical Presets", expanded=True):
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("Stable Patient", use_container_width=True):
            st.session_state['preset_age'] = 55
            st.session_state['preset_gender'] = "Male"
            st.session_state['preset_hr'] = 72.0
            st.session_state['preset_gcs'] = 15.0
            st.session_state['preset_sysbp'] = 120.0
            st.session_state['preset_temp'] = 37.0
            st.session_state['preset_bun_active'] = True
            st.session_state['preset_bun'] = 16.0
            st.session_state['preset_creat_active'] = True
            st.session_state['preset_creat'] = 0.9
            st.rerun()
    with col_p2:
        if st.button("Septic Shock", use_container_width=True):
            st.session_state['preset_age'] = 75
            st.session_state['preset_gender'] = "Male"
            st.session_state['preset_hr'] = 140.0
            st.session_state['preset_gcs'] = 6.0
            st.session_state['preset_sysbp'] = 75.0
            st.session_state['preset_temp'] = 39.5
            st.session_state['preset_bun_active'] = True
            st.session_state['preset_bun'] = 85.0
            st.session_state['preset_creat_active'] = True
            st.session_state['preset_creat'] = 4.0
            st.rerun()

with st.sidebar.expander("Patient details", expanded=True):
    age_input = st.slider("Age (Years)", min_value=15, max_value=110, value=st.session_state['preset_age'])
    gender_idx = 1 if st.session_state['preset_gender'] == "Male" else 0
    gender_label = st.radio("Sex", options=["Female", "Male"], index=gender_idx)
    gender_encoded = 1 if gender_label == "Male" else 0

with st.sidebar.expander("Vital signs", expanded=True):
    st.caption("Values used for this prediction")
    hr_input = st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=220.0, value=float(st.session_state['preset_hr']))
    gcs_input = st.number_input("Glasgow Coma Scale (GCS)", min_value=3.0, max_value=15.0, value=float(st.session_state['preset_gcs']))
    sysbp_input = st.number_input("Systolic Blood Pressure (mmHg)", min_value=40.0, max_value=250.0, value=float(st.session_state['preset_sysbp']))
    temp_input = st.number_input("Temperature (°C)", min_value=30.0, max_value=43.0, value=float(st.session_state['preset_temp']))

with st.sidebar.expander("Laboratory values (optional)", expanded=st.session_state.get('preset_bun_active', False)):
    st.caption("Add values when they are available")
    bun_active = st.checkbox("Blood Urea Nitrogen (BUN)", value=st.session_state.get('preset_bun_active', False))
    bun_val = st.number_input("BUN (mg/dL)", min_value=1.0, max_value=200.0, value=float(st.session_state['preset_bun']), disabled=not bun_active)
    
    creat_active = st.checkbox("Serum Creatinine", value=st.session_state.get('preset_creat_active', False))
    creat_val = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=15.0, value=float(st.session_state['preset_creat']), step=0.1, disabled=not creat_active)
    
    plat_active = st.checkbox("Platelet Count")
    plat_val = st.number_input("Platelets (*10^3/µL)", min_value=5.0, max_value=1000.0, value=250.0, disabled=not plat_active)
    
    wbc_active = st.checkbox("White Blood Cell Count (WBC)")
    wbc_val = st.number_input("WBC (*10^3/µL)", min_value=0.1, max_value=100.0, value=7.5, step=0.1, disabled=not wbc_active)
    
    gluc_active = st.checkbox("Serum Glucose")
    gluc_val = st.number_input("Glucose (mg/dL)", min_value=10.0, max_value=800.0, value=120.0, disabled=not gluc_active)
    
    fio2_active = st.checkbox("Inspired Oxygen Fraction (FiO2)")
    fio2_val = st.slider("FiO2 Level", min_value=0.21, max_value=1.00, value=0.40, step=0.01, disabled=not fio2_active)

compiled_observations = [
    {"Parameter": "HR", "Value": float(hr_input)}, 
    {"Parameter": "GCS", "Value": float(gcs_input)},
    {"Parameter": "SysABP", "Value": float(sysbp_input)},
    {"Parameter": "Temp", "Value": float(temp_input)}
]
if bun_active: compiled_observations.append({"Parameter": "BUN", "Value": float(bun_val)})
if creat_active: compiled_observations.append({"Parameter": "Creatinine", "Value": float(creat_val)})
if plat_active: compiled_observations.append({"Parameter": "Platelets", "Value": float(plat_val)})
if wbc_active: compiled_observations.append({"Parameter": "WBC", "Value": float(wbc_val)})
if gluc_active: compiled_observations.append({"Parameter": "Glucose", "Value": float(gluc_val)})
if fio2_active: compiled_observations.append({"Parameter": "FiO2", "Value": float(fio2_val)})

api_payload = {
    "Age": float(age_input), 
    "Gender": int(gender_encoded), 
    "Observations": compiled_observations
}

left_col, right_col = st.columns([1.1, 0.9])

with left_col:
    st.subheader("Risk estimate")
    if st.button("Calculate risk", use_container_width=True):
        with st.spinner("Calculating..."):
            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=api_payload)
                if response.status_code == 200:
                    result = response.json()
                    prob = result["Mortality_Risk_Probability"]
                    lgbm_p = result.get("LGBM_Probability", prob)
                    lstm_p = result.get("LSTM_Probability", prob)
                    flag = result["Clinical_Status_Flag"]
                    cutoff = result.get("PhysioNet_Optimal_Cutoff", 0.3112)
                    suggestions = result.get("Review_Suggestions", [])
                    drivers = result["Primary_Risk_Drivers"]

                    st.write(" ")
                    met_col, prog_col = st.columns([1, 2])
                    with met_col:
                        st.metric(
                            label="Mortality Risk Score (Ensemble)", 
                            value=f"{prob * 100:.1f} %", 
                            delta=flag, 
                            delta_color="inverse" if ("HIGH" in flag or "CRITICAL" in flag) else "normal"
                        )
                    with prog_col:
                        st.write(" ")
                        st.write(" ")
                        st.progress(float(prob))
                        st.caption(f"Status: **{flag}** | Optimal Decision Threshold: {cutoff * 100:.1f}%")

                    m1_col, m2_col = st.columns(2)
                    with m1_col:
                        st.caption(f"🌲 **LightGBM (Tabular):** {lgbm_p * 100:.1f}%")
                    with m2_col:
                        st.caption(f"🧠 **PyTorch LSTM (Temporal):** {lstm_p * 100:.1f}%")

                    st.subheader("Clinical review prompts")
                    st.caption("Standardized review prompts based on model output and clinical parameters.")
                    for suggestion in suggestions:
                        st.info(suggestion)
                    
                    st.write("---")
                    st.subheader("Main risk factors")
                    st.caption("These are the features that contributed most to this prediction.")
                    
                    features_list, impacts_list, color_list = [], [], []
                    for item in drivers["escalating"]:
                        features_list.append(item["feature"])
                        impacts_list.append(item["impact"])
                        color_list.append("#ef4444") 
                        
                    for item in drivers["mitigating"]:
                        features_list.append(item["feature"])
                        impacts_list.append(item["impact"])
                        color_list.append("#10b981") 
                        
                    if features_list:
                        fig = go.Figure(go.Bar(
                            x=impacts_list[::-1],
                            y=features_list[::-1],
                            orientation='h',
                            marker_color=color_list[::-1],
                            text=[f"{v:+.3f}" for v in impacts_list[::-1]],
                            textposition='outside',
                            hovertemplate="<b>Driver:</b> %{y}<br><b>Attribution Impact:</b> %{x:+.4f}<extra></extra>"
                        ))
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis_title="Relative Attribution Metric Score",
                            height=380,
                            margin=dict(l=10, r=45, t=10, b=10),
                            xaxis=dict(showgrid=True, gridcolor='#f3f4f6', zeroline=True, zerolinecolor='#4b5563')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Inference server error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection failure: {str(e)}")

with right_col:
    st.subheader("Patient summary")
    st.caption("Values included in the request")
    
    registry_log = [
        {"Parameter": "Age", "Recorded Value": f"{age_input} Years"},
        {"Parameter": "Sex", "Recorded Value": gender_label},
        {"Parameter": "Heart Rate", "Recorded Value": f"{hr_input} bpm"},
        {"Parameter": "Glasgow Coma Scale", "Recorded Value": f"{gcs_input} Points"},
        {"Parameter": "Systolic BP", "Recorded Value": f"{sysbp_input} mmHg"},
        {"Parameter": "Temperature", "Recorded Value": f"{temp_input} °C"}
    ]
    if bun_active: registry_log.append({"Parameter": "BUN", "Recorded Value": f"{bun_val} mg/dL"})
    if creat_active: registry_log.append({"Parameter": "Creatinine", "Recorded Value": f"{creat_val} mg/dL"})
    if plat_active: registry_log.append({"Parameter": "Platelets", "Recorded Value": f"{plat_val} *10^3/µL"})
    if wbc_active: registry_log.append({"Parameter": "WBC Count", "Recorded Value": f"{wbc_val} *10^3/µL"})
    if gluc_active: registry_log.append({"Parameter": "Glucose", "Recorded Value": f"{gluc_val} mg/dL"})
    if fio2_active: registry_log.append({"Parameter": "Inspired Oxygen (FiO2)", "Recorded Value": f"{fio2_val * 100:.0f}%"})
    
    st.dataframe(pd.DataFrame(registry_log), use_container_width=True, hide_index=True)