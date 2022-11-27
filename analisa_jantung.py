# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


#memuat dataset
def get_data():
    return pd.read_csv('heart.csv')

#model latih
def train_model():
    x = test_size
    data = get_data()

    data = data.drop(['restecg','oldpeak','slope','ca','thal'],axis=1)
    X = data.drop(["target"],axis=1)
    y = data["target"]

    from sklearn.preprocessing import MinMaxScaler
    mm_age = MinMaxScaler()
    X[['age']] = mm_age.fit_transform(X[['age']])

    mm_trestbps = MinMaxScaler()
    X[['trestbps']] = mm_trestbps.fit_transform(X[['trestbps']])

    mm_chol = MinMaxScaler()
    X[['chol']] = mm_chol.fit_transform(X[['chol']])

    mm_thal = MinMaxScaler()
    X[['thalach']] = mm_thal.fit_transform(X[['thalach']])

    #memisahkan data train dan test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=x)

    #model train
    model = ['Naive Bayes']

    column_names = ["Model","Akurasi","Presisi","Recall","F1",
                        "Total Positif","Total Negatif", "False Positif", "False Negatif",
                        "Klasifikasi"]
    results = pd.DataFrame(columns = column_names)

    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()

    #train
    classifier.fit(X_train,y_train)

    #test
    y_pred = classifier.predict(X_test)

    from sklearn import metrics
    acc = metrics.accuracy_score(y_test, y_pred)*100
    prc = metrics.precision_score(y_test, y_pred)*100
    rec = metrics.recall_score(y_test, y_pred)*100
    f1 = metrics.f1_score(y_test, y_pred)*100

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,y_pred)
    tn, fp, fn, tp = cm.ravel()

    data = [[model[0],acc, prc, rec, f1, tp, tn, fp, fn, classifier]]
    column_names = ["Model","Akurasi","Presisi","Recall","F1",
                    "Total Positif","Total Negatif", "False Positif", "False Negatif",
                    "Klasifikasi"]
    model_results = pd.DataFrame(data = data, columns = column_names)
    results = results.append(model_results, ignore_index = True)

    return results


data = get_data()
st.sidebar.subheader("Atribut Analisis")

#membuat atribut dalam sidebar
In1 = st.sidebar.number_input('Age',min_value = 1, value = 50, step = 1)
In2 = st.sidebar.selectbox('Sex',('Female', 'Male'))
In3 = st.sidebar.selectbox('Chest Pain',( 'Typical Angina', 'Atypical Angina', 'Non Anginal', 'Asymptomatic'))
In4 = st.sidebar.number_input('Resting Blood Pressure', value = 150)
In5 = st.sidebar.number_input('Serum Cholestoral (mg/dl)', value = 250)
In6 = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl',('No','Yes'))
In7 = st.sidebar.number_input('Maximum Heart Rate Achieved', value = 150)
In8 = st.sidebar.radio('Exercise Induced Angina',('No','Yes'))

#ukuran basis uji
test_size = st.sidebar.slider  (label = 'Besaran Data Test (%):',
                            min_value=0,
                            max_value=100,
                            value=20,
                            step=1)

#menyimpan hasil pelatihan
results = train_model()

#menambahkan tombol
btn_predict = st.sidebar.button("PREDIKSI")

#Title
from PIL import Image
image = Image.open('heart2.png')
st.image(image, use_column_width=True)
st.title("Sistem Klasifikasi Prediksi Penyakit Jantung")

if btn_predict:

    values = [In1,In2,In3,In4,In5,In6,In7,In8]
    column_names = ['age','sex','cp','trestbps','chol', \
                    'fbs','thalach','exang']
    df = pd.DataFrame(values, column_names)

    if df[0][1] == 'Female':df [0][1] = 0
    elif df[0][1] == 'Male':df [0][1] = 1

    if df[0][2] == 'Typical Angina':df [0][2] = 0
    elif df[0][2] == 'Atypical Angina':df [0][2] = 1
    elif df[0][2] == 'Non Anginal':df [0][2] = 2
    elif df[0][2] == 'Asymptomatic':df [0][2] = 3

    if df[0][5] == 'Yes':df [0][5] = 1
    elif df[0][5] == 'No':df [0][5] = 0

    if df[0][7] == 'Yes':df [0][7] = 1
    elif df[0][7] == 'No':df [0][7] = 0

    df[0][0] = (df[0][0] - data['age'].min()) / (data['age'].max() - data['age'].min())
    df[0][3] = (df[0][3] - data['trestbps'].min()) / (data['trestbps'].max() - data['trestbps'].min())
    df[0][4] = (df[0][4] - data['chol'].min()) / (data['chol'].max() - data['chol'].min())
    df[0][6] = (df[0][6] - data['thalach'].min()) / (data['thalach'].max() - data['thalach'].min())
    pred = [list(df[0])]

    class_nb = results['Klasifikasi']
    classifier = class_nb[0]

    mod_nb= results['Model']
    mod = mod_nb[0]

    result = classifier.predict(pred)

    if result == 0:
        st.write("PREDIKSI: **NEGATIF**")
        st.info("Pasien diprediksi tidak menderita penyakit jantung")
    if result == 1:
        st.write("PREDIKSI: **POSITIF**")
        st.warning("Pasien diprediksi menderita penyakit jantung. Segera lakukan pemeriksaan lebih lanjut!")

    st.subheader("Test Set Result (%)")
    st.table(results[["Model","Recall","Akurasi","Presisi","F1"]])

    st.subheader("Confusion Matrix")
    st.table(results[["Model","Total Positif","Total Negatif", "False Positif", "False Negatif"]])

    st.subheader("Distribusi Target")
    freq = data['target'].value_counts()
    fig, ax = plt.subplots()
    ax = freq.plot  (kind='bar',
                    figsize = (10,5),
                    rot = 0,
                    grid = False)
    st.pyplot(fig)