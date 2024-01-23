import pandas as pd
import numpy as np
import streamlit as st
#import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
import joblib
#from sklearn.metrics import r2_score
import pickle

st.sidebar.title("Sommaire")

pages = ["Contexte du projet", "Modélisation"]

page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0]:
    st.write("## Contexte du projet")

    st.write(
        "Ce projet s'inscrit dans un contexte immobilier. L'objectif est de prédire le prix d'un logement à partir de ses caractéristiques, dans un cadre d'estimations financières.")

    st.write(
        "Nous avons à notre disposition le fichier housing.csv qui contient des données immobilières. Chaque observation en ligne correspond à un logement. Chaque variable en colonne est une caractéristique de logement.")

    st.write(
        "Dans un premier temps, nous explorerons ce dataset. Puis nous l'analyserons visuellement pour en extraire des informations selon certains axes d'étude. Finalement nous implémenterons des modèles de Machine Learning pour prédire le prix.")

    st.image("immobilier.jpg")
elif page == pages[1]:
    st.write("## Modélisation")
    # Page title
    st.write("### Prédiction de sentiment de tweets 🦜")
    def text():
        # Text input
        txt_input = st.text_area('Enter your text', '', height=200)
        # Il faudrait d'abord nettoyer le text
        #
        data = {
            'text_clean': txt_input
        }

        tweet = pd.DataFrame(data, index=[0])
        return tweet

    input_df = text()
    # submitted = st.form_submit_button('Submit')

    # if submitted

    #preprocess
    scaler = StandardScaler()
    num = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    X_train[num] = scaler.fit_transform(X_train[num])
    X_test[num] = scaler.transform(X_test[num])

    # modele
    reg = joblib.load("model_reg_line")
    rf = joblib.load("model_reg_rf")
    knn = joblib.load("model_reg_knn")

    # prediction
    y_pred_reg = reg.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_knn = knn.predict(X_test)

    model_choisi = st.selectbox(label="Modèle", options=['Regression Linéaire', 'Random Forest', 'KNN'])


    def train_model(model_choisi):
        if model_choisi == 'Regression Linéaire':
            y_pred = y_pred_reg
        elif model_choisi == 'Random Forest':
            y_pred = y_pred_rf
        elif model_choisi == 'KNN':
            y_pred = y_pred_knn
        r2 = r2_score(y_test, y_pred)
        return r2


    st.write("Coefficient de détermination", train_model(model_choisi))
















