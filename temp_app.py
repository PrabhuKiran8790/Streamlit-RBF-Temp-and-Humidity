import streamlit as st
from predict import predict
import numpy as np
import joblib as jbl

def temperature():
    
    temp_data = jbl.load('models/all_temp_data.joblib')
    
    with st.sidebar:
        st.markdown("""
                    Inputs:  
                    - Temperature (T-1) (°C)  
                    - Temperature (T-2) (°C)  
                    - Humidity (T-1) (%)  
                    - Humidity (T-2) (%)  
                    - Season (Winter, Summer, Rainy)  
                    """)

    with st.form("Temperature Form"):
        col1, col2 = st.columns(2)
        with col1:
            T1 = st.number_input("Temperature (T-1) (°C)", min_value=0.0, max_value=100.0, value=32.0, step=0.1)
            H1 = st.number_input("Humidity (T-1) (%)", min_value=0.0, max_value=100.0, value=97.0, step=0.1)
            T1 = (T1 * 9/5) + 32
        with col2:
            T2 = st.number_input("Temperature (T-2) (°C)", min_value=0.0, max_value=50.0, value=33.0, step=0.1)
            T2 = (T2 * 9/5) + 32
            H2 = st.number_input("Humidity (T-2) (%)", min_value=0.0, max_value=100.0, value=98.0, step=0.1)
        season = st.selectbox("Season", ["Winter", "Summer", "Rainy"])
        if season == "Winter":
            season = 0
        elif season == "Summer":
            season = 1
        else:
            season = 2 # rainy
        if submit := st.form_submit_button("Predict"):
            inputs = np.array([T1, T2, H1, H2, season])
            temp = temp_data['temp_y_scalar'].inverse_transform(predict(inputs, "models/temperature_Metadata_N_12_P_11_bs_32.jbl", "models/temperature_RBF_ANN_model_bs_32_N_12_P_11.h5"))
            st.write(f"Predicted Temperature: {((temp[0][0] - 32)*5)/9:.1f} °C")