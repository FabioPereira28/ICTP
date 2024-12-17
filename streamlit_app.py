import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import Image

# Definir uma camada personalizada que ignora o argumento 'groups'
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')  # Remove o argumento 'groups'
        super().__init__(**kwargs)

# Função para carregar o modelo
@st.cache_resource
def load_model():
    try:
        # Substituir DepthwiseConv2D pela versão customizada
        custom_objects = {"DepthwiseConv2D": CustomDepthwiseConv2D}
        return tf.keras.models.load_model("modeloFinal.h5", custom_objects=custom_objects, compile=False)
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Carregar o modelo treinado
model = load_model()

# Verificar se o modelo foi carregado
if model is None:
    st.stop()

# Classes das raças 
CLASSES = ["American Staffordshire Terrier", "Boxer", "Chihuahua", "Doberman", "Labrador", "Pastor Alemao", "Pinscher", "Rotweiller", "Saint Bernard", "Yorkshire Terrier"]

# Função para pré-processar a imagem
def preprocess_image(image):
    image = image.resize((128, 128))  # Ajusta o tamanho para o esperado pelo modelo
    image_array = np.array(image) / 255.0  # Normaliza os valores para [0, 1]
    return np.expand_dims(image_array, axis=0)

# Interface Streamlit
st.title("Classificador de Raças de Cães 🐶")
st.write("Carrega uma imagem e descobre a raça do cão!")

# Upload da imagem
uploaded_file = st.file_uploader("Carrega uma imagem", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostrar a imagem carregada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem Carregada", use_column_width=True)

    # Fazer a classificação
    with st.spinner("A classificar..."):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = CLASSES[np.argmax(predictions)]

    # Mostrar o resultado
    st.success(f"A raça do cão é: **{predicted_class}**")
