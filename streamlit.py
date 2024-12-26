# import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

path = 'C:\\Users\\otnie\\Latihan\\.vscode\\creditcard.csv\\creditcard.csv'
df = pd.read_csv(path)

# Data fitur dan target
X = df.drop(columns=['Class'])
y = df['Class']

# Membagi dataset menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

X_train = np.array(X_train, dtype=float)
X_test = np.array(X_test, dtype=float)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Fungsi untuk menghitung probabilitas prior
def calculate_prior(y):
    classes, counts = np.unique(y, return_counts=True)
    prior = {cls: count / len(y) for cls, count in zip(classes, counts)}
    return prior

# Fungsi untuk menghitung likelihood
def calculate_likelihood(X, y):
    likelihood = {}
    classes = np.unique(y)
    for cls in classes:
        # Ambil subset data berdasarkan kelas
        X_cls = X[y == cls]
        # Hitung rata-rata dan varians untuk setiap fitur
        mean = X_cls.mean(axis=0)
        variance = X_cls.var(axis=0) + 1e-6  # Tambahkan smoothing agar tidak ada pembagian dengan nol
        likelihood[cls] = {'mean': mean, 'variance': variance}
    return likelihood

# Fungsi untuk menghitung probabilitas posterior
def calculate_posterior(x, prior, likelihood):
    posteriors = {}
    for cls, params in likelihood.items():
        # Ambil mean dan varians
        mean, variance = params['mean'], params['variance']
        # Hitung probabilitas log Gaussian
        log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * variance))
        log_prob -= 0.5 * np.sum(((x - mean) ** 2) / variance)
        posteriors[cls] = np.log(prior[cls]) + log_prob
    return max(posteriors, key=posteriors.get)

# Fungsi prediksi untuk data baru
def predict(X, prior, likelihood):
    predictions = []
    for x in X:
        predictions.append(calculate_posterior(x, prior, likelihood))
    return np.array(predictions)

# Memisahkan data berdasarkan kelas
y_train = np.array(y_train)
X_train = np.array(X_train)

# Hitung prior dan likelihood
prior = calculate_prior(y_train)
likelihood = calculate_likelihood(X_train, y_train)

# Mengukur akurasi menggunakan data test
y_test = np.array(y_test)  # Data test harus diberikan
y_test_pred = predict(X_test, prior, likelihood)
accuracy = (y_test_pred == y_test).mean()
print(f"Akurasi model: {accuracy * 100:.2f}%")

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)
# st.title("Naive Bayes Fraud Detection")
# st.write("Masukkan nilai untuk setiap fitur untuk memprediksi apakah data adalah penipuan atau bukan.")

# # Load dataset (contoh dataset kredit)
# st.sidebar.title("Upload Dataset")
# uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# if uploaded_file is not None:
#         # Membaca dataset
#         df = pd.read_csv(uploaded_file)
#         st.write("Dataset yang diunggah:")
#         st.write(df.head())
        
#         # Pemisahan fitur dan target
#         X = df.drop(columns=["Class"]).values  # Pastikan kolom target bernama "Class"
#         y = df["Class"].values  # Pastikan kolom target bernama "Class"

#         # Hitung statistik untuk Naive Bayes
#         stats = calculate_mean_variance(X, y)
        
#         # Input data pengguna
#         st.write("Masukkan nilai fitur:")
#         user_input = []
#         for col in df.columns[:-1]:  # Semua kolom kecuali "Class"
#             value = st.number_input(f"{col}", step=0.01)
#             user_input.append(value)

        # Prediksi jika ada input
        if st.button("Prediksi"):
            result = predict_naive_bayes(stats, np.array(user_input))
            if result == 1:
                st.success("Hasil prediksi: Data adalah PENIPUAN.")
            else:
                st.success("Hasil prediksi: Data BUKAN penipuan.")
