import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Carregar o modelo

model = tf.keras.models.load_model("modeloFinal.h5")

# Classes das raças (atualiza para os teus rótulos)
CLASSES = ["Raça 1", "Raça 2", "Raça 3", "Raça 4", "Raça 5", "Raça 6", "Raça 7", "Raça 8", "Raça 9", "Raça 10"]

# Função para pré-processar a imagem
def preprocess_image(image):
    image = image.resize((128, 128))  # Ajusta a resolução esperada pelo modelo
    image_array = np.array(image) / 255.0  # Normaliza os valores
    return np.expand_dims(image_array, axis=0)

# Interface Streamlit
st.title("Classificador de Raças de Cães 🐶")
st.write("Carrega uma imagem e descobre a raça do cão!")

uploaded_file = st.file_uploader("Carrega uma imagem", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostrar a imagem carregada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem Carregada", use_column_width=True)

    # Fazer a inferência
    with st.spinner("A classificar..."):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = CLASSES[np.argmax(predictions)]

    # Mostrar o resultado
    st.success(f"A raça do cão é: **{predicted_class}**")
