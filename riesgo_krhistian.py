import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image

# --- Custom Styling ---
st.markdown(
    """
    <style>
    body {
        font-family: sans-serif;
        background-color: #111827; /* Dark background */
        color: white;
        margin: 0; /* Remove default body margins */
    }
    .stApp {
        max-width: none !important; /* Allow app to take full width */
        margin: 0 !important; /* Remove app margins */
        padding: 2rem;
    }
    .st-container {
        background-color: #fff !important;
        color: #000 !important;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .st-header h1, .st-subheader {
        color: #007bff !important;
        text-align: center;
    }
    label {
        color: white !important;
        font-weight: bold;
        display: block; /* Display labels on their own line */
        margin-bottom: 0.5rem; /* Add some space below labels */
    }
    .st-selectbox div > div > div > div,
    .st-slider div > div > div,
    .st-number-input div > div > input {
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 0.75rem; /* Increase padding for better visual */
        color: #000;
        background-color: #f9f9f9; /* Light background for input fields */
        margin-bottom: 1rem; /* Add space below input fields */
    }
    .st-button > button {
        background-color: #007bff;
        color: white !important;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
        margin-top: 1rem; /* Add space above the button */
    }
    .st-button > button:hover {
        background-color: #0056b3;
    }
    .prediction-result {
        font-size: 1.2rem;
        margin-top: 1.5rem;
        text-align: center;
    }
    .high-risk {
        color: green; /* Changed to green */
        font-weight: bold;
    }
    .low-risk {
        color: green; /* Changed to green */
        font-weight: bold;
    }
    .model-info {
        margin-top: 1rem;
        font-style: italic;
        color: #999;
    }
    .st-slider > div > div > div > div[data-testid="stTrack"] {
        background-color: #007bff;
    }
    .st-slider > div > div > div > div[data-testid="stThumb"] {
        background-color: #007bff;
    }
    .st-sidebar h2, .st-sidebar label, .st-sidebar p, .st-sidebar div div div div {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load the Model ---
try:
    with open('modelo-clas-tree-knn-nn.pkl', 'rb') as file:
        model_Knn, model_Tree, model_NN, labelencoder, model_variables, min_max_scaler = pickle.load(file)
except FileNotFoundError:
    st.error("El archivo del modelo 'modelo-clas-tree-knn-nn.pkl' no se encontró. Asegúrate de que esté en la misma carpeta que este script.")
    st.stop()
except Exception as e:
    st.error(f"Ocurrió un error al cargar el modelo: {e}")
    st.stop()

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Ingrese los parámetros del auto")
    edad_vehiculo = st.number_input("Seleccione la edad del vehículo:", min_value=0, max_value=30, value=5, step=1)
    tipo_vehiculo = st.radio("Seleccione el tipo de vehículo:", ["combi", "family", "sport", "minivan"])
    modelo_prediccion = st.radio("Seleccione el modelo de predicción:", ["Red Neuronal", "KNN", "Árbol de Decisión"])
    st.write(f"Modelo Seleccionado: {modelo_prediccion}")
    predict_button = st.button("Realizar Predicción")

# --- Main Content ---
st.container()

st.header("Análisis de Riesgo")
st.subheader("Predicción para Aseguradora")

try:
    image = Image.open("auto.jpg")
    st.image(image, use_container_width=True)
except FileNotFoundError:
    st.info("No se encontró la imagen 'autos.jpg' en la misma carpeta que este script.")

if predict_button:
    # Mapping user-friendly names to model input names
    vehicle_map = {"combi": "combi", "family": "family", "sport": "sport", "minivan": "minivan"}
    model_map = {"Red Neuronal": "Nn", "KNN": "Knn", "Árbol de Decisión": "Dt"}

    selected_vehicle_input = tipo_vehiculo
    selected_model_input = model_map[modelo_prediccion]

    # Crear DataFrame con los datos del usuario
    user_data = pd.DataFrame({'age': [edad_vehiculo], 'cartype': [selected_vehicle_input]}) # Usar directamente

    # Preprocesamiento (One-Hot Encoding para el tipo de vehículo)
    user_data = pd.get_dummies(user_data, columns=['cartype'], drop_first=False)

    # Asegurarse de que las columnas estén en el mismo orden que durante el entrenamiento
    if 'model_variables' in locals():
        processed_data = pd.DataFrame(columns=model_variables)
        for col in user_data.columns:
            if col in processed_data.columns:
                processed_data[col] = user_data[col]
        processed_data = processed_data.fillna(0) # Llenar las columnas faltantes con 0
    else:
        st.error("Las variables del modelo no se cargaron correctamente.")
        st.stop()

    # Seleccionar el modelo basado en la elección del usuario
    if selected_model_input == "Knn":
        selected_model = model_Knn
    elif selected_model_input == "Dt":
        selected_model = model_Tree
    elif selected_model_input == "Nn":
        selected_model = model_NN
    else:
        st.error("Modelo no reconocido")
        st.stop()

    try:
        # Realizar la predicción
        prediction = selected_model.predict(processed_data)

        # Mostrar la predicción
        st.subheader("Resultado de la Predicción:")
        if prediction[0] == 0:  # 0 represents high risk
            st.markdown(f"<p class='prediction-result high-risk'>Alto Riesgo</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='prediction-result low-risk'>Bajo Riesgo</p>", unsafe_allow_html=True)

        st.markdown(f"<p class='model-info'>Modelo utilizado: {modelo_prediccion}</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")