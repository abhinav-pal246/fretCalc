import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FRET Analysis Pro", layout="wide")

# FIX: Added explicit text colors for metrics to prevent white-on-white visibility issues
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    /* Style for the metric cards */
    [data-testid="stMetricValue"] {
        color: #1e3a8a !important; /* Dark blue for the actual numbers */
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        color: #4b5563 !important; /* Grey for the labels */
    }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #e0e0e0; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 { color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: PHYSICAL CONSTANTS ---
st.sidebar.header("🔬 Physical Constants")
k2 = st.sidebar.number_input("Orientation Factor (K²)", value=0.667, help="Standard value 2/3")
phi_d = st.sidebar.number_input("Donor Quantum Yield (ΦD)", value=0.118, format="%.3f")
n = st.sidebar.number_input("Refractive Index (n)", value=1.33)

st.sidebar.divider()
st.sidebar.header("📡 Experimental Inputs")
f0 = st.sidebar.number_input("Donor Intensity (F₀)", value=3787.66)
f_val = st.sidebar.number_input("Donor + Acceptor Intensity (F)", value=2278.21)

# --- MAIN INTERFACE ---
st.title("🧪 Advanced FRET Calculator")
st.write("Perform automated calculations for overlap integral, Förster distance, and energy transfer.")

uploaded_file = st.file_uploader("Upload Spectral Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")

    st.subheader("🔗 Column Mapping")
    col1, col2, col3 = st.columns(3)
    wl_col = col1.selectbox("Wavelength Column (λ)", df.columns)
    id_col = col2.selectbox("Donor Intensity Column (ID)", df.columns)
    ea_col = col3.selectbox("Acceptor Molar Absorptivity Column (εA)", df.columns)

    if st.button("🚀 Calculate FRET Parameters"):
        try:
            # Data Cleaning
            for col in [wl_col, id_col, ea_col]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            clean_df = df.dropna(subset=[wl_col, id_col, ea_col]).sort_values(by=wl_col)

            # 1. Calculation of Overlap Integral (J)
            area_id = simpson(y=clean_df[id_col], x=clean_df[wl_col])
            clean_df['norm_Id'] = clean_df[id_col] / area_id
            
            clean_df['J_integrand'] = clean_df['norm_Id'] * clean_df[ea_col] * (clean_df[wl_col]**4)
            j_value = simpson(y=clean_df['J_integrand'], x=clean_df[wl_col])
            
            # 2. Calculation of R0 (Förster Distance)
            r0 = 0.02108 * ((k2 * phi_d * (n**-4) * j_value) ** (1/6))
            
            # 3. Efficiency (E) and Distance (r)
            e_efficiency = 1 - (f_val / f0)
            r_distance = r0 * (((1 - e_efficiency) / e_efficiency) ** (1/6))

            # --- DISPLAY RESULTS (Metrics now have forced dark text color) ---
            st.divider()
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("Overlap Integral (J)", f"{j_value:.4e}")
            res_col2.metric("Förster Distance (R₀)", f"{r0:.4f} nm")
            res_col3.metric("Efficiency (E)", f"{e_efficiency:.4f}")
            res_col4.metric("Distance (r)", f"{r_distance:.4f} nm")

            st.info(f"Summary: E = {e_efficiency:.3f} | R₀ = {r0:.3f} nm | r = {r_distance:.3f} nm")

            # --- VISUALIZATION ---
            st.subheader("📊 Spectral Overlap Plot")
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(clean_df[wl_col], clean_df['norm_Id'], color='#2563eb', label='Donor Emission')
            ax1.fill_between(clean_df[wl_col], clean_df['norm_Id'], alpha=0.2, color='#2563eb')
            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Normalized Intensity', color='#2563eb')
            
            ax2 = ax1.twinx()
            ax2.plot(clean_df[wl_col], clean_df[ea_col], color='#dc2626', linestyle='--', label='Acceptor Absorption')
            ax2.set_ylabel('Molar Absorptivity (εA)', color='#dc2626')
            
            st.pyplot(fig)

            # Theory Validation
            st.subheader("✅ Distance Validation")
            if 0.5 * r0 <= r_distance <= 1.5 * r0:
                st.success(f"Valid result: Distance r ({r_distance:.2f} nm) is within 0.5R₀ to 1.5R₀.")
            else:
                st.warning("Distance r is outside the optimal FRET sensitivity range.")

        except Exception as e:
            st.error(f"Calculation error: {e}")
else:
    st.info("👋 Upload your spectral file to begin.")