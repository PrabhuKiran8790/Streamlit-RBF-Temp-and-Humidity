import streamlit as st
import numpy as np
from pathlib import Path
import joblib as jbl
from predict import predict
from PIL import Image
import base64

st.set_page_config(page_title='RBF NN Humidity and Temperature', page_icon='⚡️', layout="wide", initial_sidebar_state="expanded", menu_items=None, )

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    return base64.b64encode(img_bytes).decode()

warangal_html = f'<img src="data:image/png;base64,{img_to_bytes("warangal.png")}" class="img-fluid" width="600" height="290">'

rbf_architecture = Image.open('rbf_architecture.jpeg')

def warangal(): return st.markdown(
    warangal_html, unsafe_allow_html=True,
)

def predict_form():
    with st.form("Temperature and Humidity Form"):
        st.write("#### Please enter the following inputs:")
        col1, col2 = st.columns(2)
        with col1:
            T1 = st.number_input("Temperature (T-1) (°C)", min_value=0.0, max_value=100.0, value=32.0, step=0.1)
            H1 = st.number_input("Humidity (T-1) (%)", min_value=0.0, max_value=100.0, value=97.0, step=0.1)
            T1 = (T1 * 9/5) + 32
        with col2:
            T2 = st.number_input("Temperature (T-24) (°C)", min_value=0.0, max_value=50.0, value=33.0, step=0.1)
            T2 = (T2 * 9/5) + 32
            H2 = st.number_input("Humidity (T-24) (%)", min_value=0.0, max_value=100.0, value=98.0, step=0.1)
        season = st.selectbox("Season", ["Winter", "Summer", "Rainy"])
        if season == "Winter":
            season = 0
        elif season == "Summer":
            season = 1
        else:
            season = 2 # rainy
        if submit := st.form_submit_button("Predict"):
            inputs = np.array([T1, T2, H1, H2, season])
            temp = predict(inputs, 'temp_optimal_info.jbl')
            humidity = predict(inputs, 'humidity_optimal_info.jbl')
            st.write(f"#### Predicted Temperature: {((temp[0][0] - 32)*5)/9:.1f} °C")
            st.write(f"#### Predicted Humidity: {humidity[0][0]:.1f} %")


choice = ['Predict']
param = st.sidebar.selectbox("Select any of the options below", choice)

title = st.write(
    "## RBF Neural Network Web App for Temperature and Humidity Prediction"
)

if param == 'Predict':
    with st.sidebar:
            st.markdown("""
                        ---
                        **Inputs:**
                        - Temperature (T-1) (°C)  
                        - Temperature (T-2) (°C)  
                        - Humidity (T-1) (%)  
                        - Humidity (T-2) (%)  
                        - Season (Winter, Summer, Rainy)  
                        ---
                        **Outputs:**
                        - Temperature (°C)
                        - Humidity (%)
                        ---
                        """)
    
    col1, col2 = st.columns([1.3, 2])
    with col1:
        warangal()
        st.caption("Warangal, Telangana, India")
    with col2:
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.write("#")
            st.image(rbf_architecture, caption="RBF Neural Network Architecture", use_column_width=False, width=600, output_format='PNG')
    predict_form()  



    
header_html = f'<img src="data:image/png;base64,{img_to_bytes("logos.png")}" class="img-fluid" width="300" height="170">'
st.sidebar.markdown(
    header_html, unsafe_allow_html=True,
)

st.sidebar.write("#\n#\n#")
st.sidebar.markdown("**Disclaimer:** This project is associated with the [Center for Artificial Intelligence and Deep Learning (CAIDL)](https://sru.edu.in/centers/caidl/) at [SR University](https://sru.edu.in).")