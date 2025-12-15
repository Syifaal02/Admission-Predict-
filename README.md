# Prediksi Peluang Keterima Pascasarjana

Anggota Kelompok:  
- Siti Ayu (2406675) 
- Syifa Aleyda Nur Fauziyah (2406941)

---

## Deskripsi Proyek
Proyek ini bertujuan untuk menganalisis **peluang diterima mahasiswa pascasarjana**  
berdasarkan data akademik calon mahasiswa dan membangun **model Machine Learning**  
untuk memprediksi *Chance of Admit* pada program pascasarjana tertentu.

Proyek ini juga mengevaluasi faktor-faktor yang memengaruhi peluang diterima,  
seperti GRE Score, TOEFL Score, CGPA, SOP, LOR, Research, dan University Rating.  

Hasil analisis dan prediksi disajikan dalam bentuk **dashboard interaktif berbasis Streamlit**,  
yang memungkinkan pengguna:
- Memahami faktor akademik yang paling berpengaruh (*Feature Importance*).  
- Melakukan *What-If Simulation* untuk melihat perubahan peluang jika skor akademik meningkat.  
- Memperoleh prediksi peluang diterima dan rekomendasi strategis secara langsung.

---

## Dataset
Dataset yang digunakan adalah **Admission Predict Dataset** yang telah melalui  
proses *data cleaning* dan *preprocessing*.

Kolom utama:
- GRE Score  
- TOEFL Score  
- University Rating  
- SOP  
- LOR  
- CGPA  
- Research  
- Chance of Admit

---

## Tools & Teknologi
- Python  
- Pandas & NumPy  
- Scikit-learn  
- Matplotlib & Seaborn  
- Streamlit  
- Joblib

---

## Model Machine Learning
- Jenis Model:  
  - Linear Regression  
  - Ridge Regression  
  - Lasso Regression  
  - Random Forest Regressor  
  - Gradient Boosting Regressor  
- Target Prediksi: **Chance of Admit**  
- Model terbaik dipilih berdasarkan **Cross-Validation (R²)**  
- Model disimpan dalam format `.pkl` dan di-deploy ke **Streamlit**  

---

## Fitur Dashboard
- Input skor akademik: GRE, TOEFL, CGPA, SOP, LOR, Research, University Rating  
- Prediksi peluang diterima (*Chance of Admit*)  
- Rekomendasi kualitatif: Peluang sangat tinggi / tinggi / sedang / rendah  
- Visualisasi *Feature Importance* interaktif  
- Analisis error (*Residuals*) dan deteksi bias / outlier  
- *What-If Simulation* untuk mengubah skor dan melihat perubahan peluang

---

## Cara Menjalankan Aplikasi
1. Install dependensi:
```bash
pip install -r requirements.txt
