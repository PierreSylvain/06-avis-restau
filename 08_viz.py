import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import pandas as pd
from tensorflow.python.keras.applications.vgg16 import preprocess_input

st.set_page_config(
    page_title='Avis Restau',
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    cs_sidebar()
    cs_body()

    return None


def cs_sidebar():
    st.sidebar.header('Avis Restau')
    st.sidebar.markdown('''Améliorez le produit IA de votre start-up''', unsafe_allow_html=True)
    st.sidebar.subheader("Analyse des données textuelles")
    st.sidebar.markdown(" - Word cloud")
    st.sidebar.markdown(" - Visualisation des topics")
    st.sidebar.subheader("Analyse des images")
    st.sidebar.markdown("CNN et Transfer Learning")
    st.sidebar.markdown(" - Exemple de prédiction")
    st.sidebar.markdown(" - Faire une prédiction")

    return None


def cs_body():
    st.header('Analyse des données textuelles')
    image = Image.open('img/wordcloud.png')
    st.image(image)
    st.header('Visualisation des topics')
    html = open('lda.html', 'r', encoding='utf-8')
    source_code = html.read()
    components.html(source_code, height=1000)

    st.header('Traitement des images')
    st.subheader("Exemple de prédiction")
    # Prédiction
    img = '../data/'
    filename = 'interieur-restaurant.jpg'
    image = Image.open(filename)
    st.image(image)

    model = keras.models.load_model("cnn_vgg16_model")
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape(1, 224, 224, 3)
    image = preprocess_input(image)
    predict = model.predict(image)
    pred = """
    |Boisson|Nourriture|Intérieur|Menu|Extérieur|
    |:---:|:---:|:---:|:---:|:---:|
    """
    pred += "|" + predict[0][0].astype(int).astype(str)
    pred += "|" + predict[0][1].astype(int).astype(str)
    pred += "|" + predict[0][2].astype(int).astype(str)
    pred += "|" + predict[0][3].astype(int).astype(str)
    pred += "|" + predict[0][4].astype(int).astype(str) + "|"
    st.markdown(pred)

    st.subheader("Faire une prédiction")
    uploaded_file = st.file_uploader("Importer un fichier...", type="jpg")
    if uploaded_file is not None:
        # save file to disk
        filename = 'psa.jpg'
        with open(filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        image = load_img(filename, target_size=(224, 224))
        st.image(image)
        image = img_to_array(image)
        image = image.reshape(1, 224, 224, 3)
        image = preprocess_input(image)
        predict = model.predict(image)
        pred = """
        |Boisson|Nourriture|Intérieur|Menu|Extérieur|
        |:---:|:---:|:---:|:---:|:---:|
        """
        pred += "|" + predict[0][0].astype(int).astype(str)
        pred += "|" + predict[0][1].astype(int).astype(str)
        pred += "|" + predict[0][2].astype(int).astype(str)
        pred += "|" + predict[0][3].astype(int).astype(str)
        pred += "|" + predict[0][4].astype(int).astype(str) + "|"
        st.markdown(pred)


if __name__ == '__main__':
    main()
