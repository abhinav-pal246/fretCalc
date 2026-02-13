import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FRET Analysis Pro", layout="wide")

# CUSTOM CSS: Forces high-contrast visibility for metrics and professional dark theme
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    [data-testid="stMetricValue"] {
        color: #1e3a8a !important; 
        font-weight: bold;
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #4b5563 !important;
        font-weight: 600 !important;
    }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #e0e0e0; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    h1, h2, h3 { color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'auto_f0' not in st.session_state:
    st.session_state.auto_f0 = 3787.66 # Standard baseline from HSA study

# --- SIDEBAR: INPUTS ---
st.sidebar.header("🔬 Physical Constants")
# Constants derived from experimental setup [cite: 702]
k2 = st.sidebar.number_input("Orientation Factor (K²)", value=0.667, help="Standard value 2/3") # [cite: 702]
phi_d = st.sidebar.number_input("Donor Quantum Yield (ΦD)", value=0.118, format="%.3f") # [cite: 702]
n = st.sidebar.number_input("Refractive Index (n)", value=1.33) # [cite: 702]

st.sidebar.divider()
st.sidebar.header("📡 Experimental Inputs")

# F0 is automatically updated when the user clicks 'Auto-detect' below
f0 = st.sidebar.number_input("Initial Donor Intensity (F₀)", 
                             value=float(st.session_state.auto_f0),
                             help="Peak intensity from the Donor column") #

# F is manually entered by the user
f_val = st.sidebar.number_input("Quenched Intensity (F)", value=2278.21) #

# --- MAIN INTERFACE ---
st.title("🧪 Advanced FRET Calculator")
st.write("Determine Overlap Integral ($J$), Förster Distance ($R_0$), and Molecular Distance ($r$).")

uploaded_file = st.file_uploader("Upload Spectral Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
        
        st.subheader("🔗 Column Mapping")
        col1, col2, col3 = st.columns(3)
        wl_col = col1.selectbox("Wavelength Column (λ)", df.columns)
        id_col = col2.selectbox("Donor Intensity Column (ID)", df.columns)
        ea_col = col3.selectbox("Acceptor Molar Absorptivity Column (εA)", df.columns)

        # TRIGGER: Find max value in Donor Intensity column
        if st.button("🔍 Find Peak Donor Intensity (F₀)"):
            temp_id = pd.to_numeric(df[id_col], errors='coerce')
            max_intensity = temp_id.max()
            if not np.isnan(max_intensity):
                st.session_state.auto_f0 = max_intensity
                st.success(f"Detected Peak Intensity: {max_intensity:.2f}")
                st.rerun()
            else:
                st.error("Could not find a numeric peak in the selected column.")

        if st.button("🚀 Calculate FRET Parameters"):
            try:
                # Standard Data Cleaning
                for col in [wl_col, id_col, ea_col]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                clean_df = df.dropna(subset=[wl_col, id_col, ea_col]).sort_values(by=wl_col)

                # 1. Overlap Integral (J)
                area_id = simpson(y=clean_df[id_col], x=clean_df[wl_col])
                clean_df['norm_Id'] = clean_df[id_col] / area_id
                
                # Integrand calculation: ID(λ) * εA(λ) * λ⁴
                clean_df['J_integrand'] = clean_df['norm_Id'] * clean_df[ea_col] * (clean_df[wl_col]**4)
                j_value = simpson(y=clean_df['J_integrand'], x=clean_df[wl_col])
                
                # 2. Förster Distance (R₀) [cite: 700]
                r0 = 0.02108 * ((k2 * phi_d * (n**-4) * j_value) ** (1/6))
                
                # 3. Efficiency (E) and Distance (r)
                e_efficiency = 1 - (f_val / f0)
                r_distance = r0 * (((1 - e_efficiency) / e_efficiency) ** (1/6))

                # --- RESULTS DISPLAY ---
                st.divider()
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                
                # Format J to 3 decimal places in scientific notation
                formatted_j = "{:.3e}".format(j_value).replace("e+", " x 10^")
                
                res_col1.metric("Overlap Integral (J)", formatted_j)
                res_col2.metric("Förster Distance (R₀)", f"{r0:.4f} nm")
                res_col3.metric("Efficiency (E)", f"{e_efficiency:.4f}")
                res_col4.metric("Distance (r)", f"{r_distance:.4f} nm")

                # --- VISUALIZATION ---
                
                fig, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(clean_df[wl_col], clean_df['norm_Id'], color='#2563eb', label='Donor Emission')
                ax1.fill_between(clean_df[wl_col], clean_df['norm_Id'], alpha=0.2, color='#2563eb')
                ax1.set_ylabel('Normalized Intensity', color='#2563eb')
                
                ax2 = ax1.twinx()
                ax2.plot(clean_df[wl_col], clean_df[ea_col], color='#dc2626', linestyle='--', label='Acceptor Absorption')
                ax2.set_ylabel('Molar Absorptivity (εA)', color='#dc2626')
                
                plt.title("Spectral Overlap Analysis")
                st.pyplot(fig)

                st.info(f"Summary Results: E = {e_efficiency:.3f} | R₀ = {r0:.3f} nm | r = {r_distance:.3f} nm")

            except Exception as e:
                st.error(f"Calculation Error: {e}")

    except Exception as e:
        st.error(f"File Error: {e}")
else:
    st.info("👋 Upload spectral data to begin analysis.")