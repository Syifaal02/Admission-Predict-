import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Admission Predict",
    page_icon="🎓",
    layout="wide"
)

# ======================================================
# CSS — DASHBOARD PROFESIONAL + HOVER EFFECT
# ======================================================
st.markdown("""
<style>
.stApp { background-color: #F4F6FB; font-family: Inter, sans-serif; }
section[data-testid="stSidebar"] { background-color: #0B0F19; }
section[data-testid="stSidebar"] * { color: white !important; }

.topbar {
    background: #2563EB;
    padding: 14px 20px;
    border-radius: 14px;
    color: white;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 20px;
}

.card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 10px 24px rgba(15,23,42,.08);
    margin-bottom: 20px;
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 40px rgba(37,99,235,0.25);
}

.metric {
    font-size: 32px;
    font-weight: 800;
    color: #2563EB;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #0F172A;
}

.section-desc {
    font-size: 14px;
    color: #64748B;
    margin-bottom: 16px;
}

.progress {
    height: 8px;
    background: #E5E7EB;
    border-radius: 999px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: #2563EB;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD FILES
# ======================================================
model = joblib.load("best_model_admission.pkl")
metrics = pd.read_csv("metrics_admission.csv")

data = pd.read_csv("Admission_Predict.csv")
data.columns = data.columns.str.strip()
if "Serial No." in data.columns:
    data.drop(columns=["Serial No."], inplace=True)

TARGET = "Chance of Admit"
X = data.drop(columns=[TARGET])
y = data[TARGET]

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def get_user_input_df(X_df, key_prefix):
    user_input = {}
    for col in X_df.columns:
        user_input[col] = st.slider(
            col,
            float(X_df[col].min()),
            float(X_df[col].max()),
            float(X_df[col].median()),
            key=f"{key_prefix}_{col}"
        )
    return pd.DataFrame([user_input])

def predict_from_df(input_df):
    return float(model.predict(input_df)[0])

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.markdown("## 🎓 Admission Predict")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Evaluation", "Feature Importance", "Prediction", "Recommendation"]
)

# ======================================================
# TOP BAR
# ======================================================
st.markdown("<div class='topbar'>Admission Prediction Dashboard</div>", unsafe_allow_html=True)

# ======================================================
# DASHBOARD
# ======================================================
if page == "Dashboard":
    st.markdown("<div class='section-title'>Dashboard Overview</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label in zip([c1, c2, c3, c4], ["MAE", "MSE", "RMSE", "R2"]):
        col.markdown(f"""
        <div class="card">
            <div class="metric">{metrics.loc[0, label]:.4f}</div>
            <p>{label}</p>
        </div>
        """, unsafe_allow_html=True)

# ======================================================
# EVALUATION
# ======================================================
elif page == "Evaluation":
    st.markdown("<div class='section-title'>Model Evaluation</div>", unsafe_allow_html=True)
    st.dataframe(metrics, use_container_width=True)

# ======================================================
# FEATURE IMPORTANCE
# ======================================================
elif page == "Feature Importance":
    st.markdown("<div class='section-title'>Feature Importance</div>", unsafe_allow_html=True)

    perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": perm.importances_mean
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)

# ======================================================
# PREDICTION
# ======================================================
elif page == "Prediction":
    st.markdown("<div class='section-title'>Admission Prediction</div>", unsafe_allow_html=True)

    input_df = get_user_input_df(X, "pred")
    pred = predict_from_df(input_df)
    st.session_state["pred"] = pred

    st.markdown(f"""
    <div class="card">
        <b>Predicted Chance of Admit</b>
        <div class="metric">{pred*100:.1f}%</div>
        <div class="progress">
            <div class="progress-fill" style="width:{pred*100:.1f}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# RECOMMENDATION (FIXED + HOVER + BLUE FONT)
# ======================================================
elif page == "Recommendation":
    st.markdown("<div class='section-title'>Admission Recommendation</div>", unsafe_allow_html=True)

    if "pred" not in st.session_state:
        st.info("Silakan isi profil terlebih dahulu untuk mendapatkan rekomendasi.")
        input_df = get_user_input_df(X, "rec")
        pred = predict_from_df(input_df)
        st.session_state["pred"] = pred
    else:
        pred = st.session_state["pred"]

    st.markdown(f"""
    <div class="card">
        <b>Predicted Chance of Admit</b>
        <div class="metric">{pred*100:.1f}%</div>
        <div class="progress">
            <div class="progress-fill" style="width:{pred*100:.1f}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if pred >= 0.80:
        msg = "Peluang diterima sangat tinggi"
        advice = "Profil akademik dan non-akademik sangat kompetitif dan sesuai dengan universitas berdaya saing tinggi."
    elif pred >= 0.65:
        msg = "Peluang diterima cukup kuat"
        advice = "Profil sudah baik, dengan peluang peningkatan melalui optimasi CGPA, GRE/TOEFL, atau SOP."
    elif pred >= 0.50:
        msg = "Peluang diterima berada pada batas minimum"
        advice = "Strategi aplikasi sangat penting, disarankan memilih universitas dengan tingkat kompetisi menengah."
    else:
        msg = "Peluang diterima relatif rendah"
        advice = "Disarankan meningkatkan profil akademik dan pengalaman riset sebelum melakukan pendaftaran."

    st.markdown(f"""
    <div class="card">
        <h3 style="color:#2563EB;">Hasil Rekomendasi</h3>
        <p style="font-size:16px; font-weight:600; color:#2563EB;">{msg}</p>
        <p style="color:#2563EB; font-size:15px;">{advice}</p>
        <p style="color:#2563EB; font-size:13px;">
        Rekomendasi ini bersifat <i>decision support</i> berdasarkan output model.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.caption("© 2025 Admission Predict • Professional Decision Support Dashboard")
