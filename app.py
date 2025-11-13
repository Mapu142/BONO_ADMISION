import os
import numpy as np
import streamlit as st
import joblib

# Configurar TensorFlow silencioso
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configuración básica
st.set_page_config(
    page_title="Predictor de Admisión",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Cargar modelo y scaler existentes"""
    if not os.path.exists('mejor_modelo_admision.h5'):
        st.error("❌ Archivo 'mejor_modelo_admision.h5' no encontrado")
        st.stop()
        
    if not os.path.exists('scaler_admision.pkl'):
        st.error("❌ Archivo 'scaler_admision.pkl' no encontrado")
        st.stop()
    
    try:
        from tensorflow import keras
        model = keras.models.load_model('mejor_modelo_admision.h5', compile=False)
        model.compile(optimizer='adam', loss='mse')
        scaler = joblib.load('scaler_admision.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        st.stop()

def predict(gre, toefl, rating, sop, lor, cgpa, research, model, scaler):
    """Hacer predicción"""
    data = np.array([[gre, toefl, rating, sop, lor, cgpa, research]])
    scaled = scaler.transform(data)
    pred = model.predict(scaled, verbose=0)
    return float(pred[0][0]) * 100

# Cargar modelo
model, scaler = load_model()

# Interfaz
st.title("🎓 Predictor de Admisión")

col1, col2 = st.columns(2)

with col1:
    gre = st.slider("GRE Score", 260, 340, 320)
    toefl = st.slider("TOEFL Score", 0, 120, 110)
    rating = st.slider("University Rating", 1, 5, 3)
    research = st.selectbox("Research", [0, 1], index=1, format_func=lambda x: "Sí" if x else "No")

with col2:
    sop = st.slider("SOP", 1.0, 5.0, 4.0, 0.5)
    lor = st.slider("LOR", 1.0, 5.0, 4.0, 0.5)
    cgpa = st.slider("CGPA", 6.8, 10.0, 8.5, 0.01)

# Predicción
prob = predict(gre, toefl, rating, sop, lor, cgpa, research, model, scaler)

# Resultado
if prob >= 80:
    color = "#28a745"
    status = "Muy Alta"
elif prob >= 60:
    color = "#ffc107"
    status = "Alta"
elif prob >= 40:
    color = "#fd7e14"
    status = "Media"
else:
    color = "#dc3545"
    status = "Baja"

st.markdown(f"""
<div style="text-align: center; padding: 20px; margin: 20px 0;">
    <h2 style="color: {color};">{prob:.1f}%</h2>
    <p style="color: {color};">Probabilidad {status}</p>
</div>
""", unsafe_allow_html=True)

st.progress(prob/100)