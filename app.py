import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Modeli yeniden yüklemek
@st.cache_resource
def train_model():
    # Veriyi yükleme
    df = pd.read_csv('diyabet\\diabetes.csv')

    # Outlier temizleme
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Veri bölme
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Standardizasyon
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model eğitimi
    model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=2)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_test_scaled, y_test

# Streamlit arayüzü
st.title("Diyabet Tahmin Aracı")
st.write("Bu araç, diyabet olup olmadığınızı tahmin etmek için bir makine öğrenimi modelini kullanır.")

# Model ve scaler yükleme
model, scaler, X_test_scaled, y_test = train_model()

# Kullanıcı girişi
st.header("Yeni Veri Girişi")
preg = st.number_input("Gebelik Sayısı", min_value=0, max_value=20, value=1, step=1)
glucose = st.number_input("Glikoz Seviyesi", min_value=0.0, value=120.0)
bp = st.number_input("Kan Basıncı (mm Hg)", min_value=0.0, value=80.0)
skin = st.number_input("Cilt Kalınlığı (mm)", min_value=0.0, value=20.0)
insulin = st.number_input("İnsülin Seviyesi", min_value=0.0, value=85.0)
bmi = st.number_input("Vücut Kitle İndeksi (BMI)", min_value=0.0, value=25.0)
dpf = st.number_input("Genetik Risk Faktörü (DPF)", min_value=0.0, value=0.5)
age = st.number_input("Yaş", min_value=1, max_value=120, value=30, step=1)

# Tahmin
if st.button("Tahmin Et"):
    new_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    result = "Pozitif (Diyabet)" if prediction[0] == 1 else "Negatif (Diyabet Yok)"
    st.subheader(f"Model Tahmini: {result}")


