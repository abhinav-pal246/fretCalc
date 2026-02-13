import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FRET Analysis Pro", layout="wide")

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

# --- SIDEBAR: PHYSICAL CONSTANTS ---
st.sidebar.header("🔬 Physical Constants")
# [cite_start]Constants derived from your research paper [cite: 702]
k2 = st.sidebar.number_input("Orientation Factor (K²)", value=0.667, help="Standard value 2/3")
phi_d = st.sidebar.number_input("Donor Quantum Yield (ΦD)", value=0.118, format="%.3f")
n = st.sidebar.number_input("Refractive Index (n)", value=1.33)

st.sidebar.divider()
st.sidebar.header("📡 Experimental Inputs")

# Using session state to allow the app to suggest the detected F0
if 'detected_f0' not in st.session_state:
    st.session_state.detected_f0 = 3787.66 # Default fallback

f0 = st.sidebar.number_input("Initial Donor Intensity (F₀)", 
                             value=float(st.session_state.detected_f0),
                             help="Auto-calculated from peak intensity if file is uploaded")

f_val = st.sidebar.number_input("Quenched Intensity (F)", value=2278.21)

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
        
        # Column Mapping
        st.subheader("🔗 Column Mapping")
        col1, col2, col3 = st.columns(3)
        wl_col = col1.selectbox("Wavelength Column (λ)", df.columns)
        id_col = col2.selectbox("Donor Intensity Column (ID)", df.columns)
        ea_col = col3.selectbox("Acceptor Molar Absorptivity Column (εA)", df.columns)

        # Logic to automatically find the peak for F0
        if st.button("🔍 Auto-detect F0 from Peak"):
            # Ensure column is numeric
            temp_id = pd.to_numeric(df[id_col], errors='coerce')
            peak_f0 = temp_id.max()
            if not np.isnan(peak_f0):
                st.session_state.detected_f0 = peak_f0
                st.rerun() # Refresh to update the sidebar input
            else:
                st.error("Could not find a valid peak in the selected column.")

        if st.button("🚀 Calculate FRET Parameters"):
            try:
                # Data Cleaning
                for col in [wl_col, id_col, ea_col]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                clean_df = df.dropna(subset=[wl_col, id_col, ea_col]).sort_values(by=wl_col)

                # [cite_start]1. Calculation of Overlap Integral (J) [cite: 693]
                area_id = simpson(y=clean_df[id_col], x=clean_df[wl_col])
                clean_df['norm_Id'] = clean_df[id_col] / area_id
                
                clean_df['J_integrand'] = clean_df['norm_Id'] * clean_df[ea_col] * (clean_df[wl_col]**4)
                j_value = simpson(y=clean_df['J_integrand'], x=clean_df[wl_col])
                
                # [cite_start]2. Calculation of R0 (Förster Distance) [cite: 700]
                r0 = 0.02108 * ((k2 * phi_d * (n**-4) * j_value) ** (1/6))
                
                # 3. Efficiency (E) and Distance (r)
                e_efficiency = 1 - (f_val / f0)
                r_distance = r0 * (((1 - e_efficiency) / e_efficiency) ** (1/6))

                # --- DISPLAY RESULTS ---
                st.divider()
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                formatted_j = "{:.3e}".format(j_value).replace("e+", " x 10^")
                
                res_col1.metric("Overlap Integral (J)", formatted_j)
                res_col2.metric("Förster Distance (R₀)", f"{r0:.4f} nm")
                res_col3.metric("Efficiency (E)", f"{e_efficiency:.4f}")
                res_col4.metric("Distance (r)", f"{r_distance:.4f} nm")

                st.info(f"Summary Results: E = {e_efficiency:.3f} | R₀ = {r0:.3f} nm | r = {r_distance:.3f} nm")

                # --- VISUALIZATION ---
                
                fig, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(clean_df[wl_col], clean_df['norm_Id'], color='#2563eb', label='Donor Emission (Normalized)')
                ax1.fill_between(clean_df[wl_col], clean_df['norm_Id'], alpha=0.2, color='#2563eb')
                ax1.set_xlabel('Wavelength (nm)')
                ax1.set_ylabel('Normalized Intensity', color='#2563eb')
                
                ax2 = ax1.twinx()
                ax2.plot(clean_df[wl_col], clean_df[ea_col], color='#dc2626', linestyle='--', label='Acceptor Absorption (εA)')
                ax2.set_ylabel('Molar Absorptivity (εA)', color='#dc2626')
                
                plt.title("FRET Overlap Region")
                st.pyplot(fig)

                # Validation
                st.subheader("✅ Technical Validation")
                if 0.5 * r0 <= r_distance <= 1.5 * r0:
                    st.success(f"Theoretical validation complete: r ({r_distance:.2f} nm) is within range (0.5R₀ to 1.5R₀).")
                else:
                    st.warning("The calculated distance is outside the optimal sensitivity range.")

            except Exception as e:
                st.error(f"Calculation Error: {e}")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👋 Upload data to begin.")