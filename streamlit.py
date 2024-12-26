import streamlit as st
import numpy as np
import pandas as pd

def calculate_mean_variance(X, y):
    classes = np.unique(y)
    stats = {}
    for c in classes:
        X_c = X[y == c]
        stats[c] = {
            "mean": np.mean(X_c, axis=0),
            "variance": np.var(X_c, axis=0)
        }
    return stats

def gaussian_probability(x, mean, variance):
    eps = 1e-6  # Mencegah pembagian oleh nol
    coefficient = 1 / np.sqrt(2 * np.pi * variance + eps)
    exponent = np.exp(-((x - mean) ** 2) / (2 * variance + eps))
    return coefficient * exponent

def predict_naive_bayes(stats, X_input):
    classes = stats.keys()
    probabilities = {}
    for c in classes:
        class_prob = 1
        for i in range(len(X_input)):
            mean = stats[c]["mean"][i]
            variance = stats[c]["variance"][i]
            prob = gaussian_probability(X_input[i], mean, variance)
            class_prob *= prob
        probabilities[c] = class_prob
    return max(probabilities, key=probabilities.get)

st.title("Naive Bayes Fraud Detection")
st.write("Masukkan nilai untuk setiap fitur untuk memprediksi apakah data adalah penipuan atau bukan.")

path = 'C:\\Users\\otnie\\Latihan\\.vscode\\creditcard.csv\\creditcard.csv'
df = pd.read_csv(path)

X = df.drop(columns=["Class"]).values
y = df["Class"].values

stats = calculate_mean_variance(X, y)

st.write("Masukkan nilai fitur:")
user_input = []
for col in df.columns[:-1]:  # Semua kolom kecuali "Class"
    value = st.number_input(f"{col}", step=0.01)
    user_input.append(value)

if st.button("Prediksi"):
    result = predict_naive_bayes(stats, np.array(user_input))
    if result == 1:
        st.success("Hasil prediksi: Data adalah PENIPUAN.")
    else:
        st.success("Hasil prediksi: Data BUKAN penipuan.")
