import streamlit as st
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="🛡️",
    layout="wide"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load("model/spam_classifier.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_model()


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

.hero {
    background: linear-gradient(135deg, #111827, #1e3a8a);
    padding: 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    color: #dbeafe;
}

.result-box {
    padding: 25px;
    border-radius: 18px;
    margin-top: 20px;
    text-align: center;
}

.spam {
    background-color: #fee2e2;
    border: 2px solid #ef4444;
}

.ham {
    background-color: #dcfce7;
    border: 2px solid #22c55e;
}

.result-title {
    font-size: 32px;
    font-weight: bold;
}

.confidence {
    font-size: 20px;
    margin-top: 10px;
}

.info-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HERO SECTION
# ==========================================

st.markdown("""
<div class="hero">

<h1>🛡️ Spam Email Classifier</h1>

<p>
AI-powered email classification using Machine Learning,
TF-IDF and Logistic Regression.
</p>

</div>
""", unsafe_allow_html=True)


# ==========================================
# PROJECT INFORMATION
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Training Messages",
        "4,457"
    )

with col2:
    st.metric(
        "Test Messages",
        "1,115"
    )

with col3:
    st.metric(
        "Model Accuracy",
        "96.95%"
    )


st.divider()


# ==========================================
# EMAIL INPUT
# ==========================================

st.subheader("📩 Analyze an Email")

st.write(
    "Paste an email or message below and our machine learning "
    "model will predict whether it is Spam or Not Spam."
)


email_text = st.text_area(
    "Email / Message",
    height=220,
    placeholder=(
        "Example:\n\n"
        "Congratulations! You have won a $1000 prize. "
        "Click the link below to claim your reward."
    )
)


# ==========================================
# ANALYZE BUTTON
# ==========================================

if st.button(
    "🔍 Analyze Email",
    use_container_width=True
):

    if not email_text.strip():

        st.warning(
            "⚠️ Please enter an email or message first."
        )

    else:

        # Convert text into TF-IDF features
        email_vector = vectorizer.transform([email_text])

        # Prediction
        prediction = model.predict(email_vector)[0]

        # Probability
        probabilities = model.predict_proba(email_vector)[0]

        confidence = max(probabilities) * 100


        # ======================================
        # SPAM RESULT
        # ======================================

        if prediction == 1:

            st.markdown(
                f"""
                <div class="result-box spam">

                <div class="result-title">
                🚨 SPAM DETECTED
                </div>

                <div class="confidence">
                Confidence: <b>{confidence:.2f}%</b>
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.error(
                "This message has characteristics commonly "
                "associated with spam."
            )


        # ======================================
        # HAM RESULT
        # ======================================

        else:

            st.markdown(
                f"""
                <div class="result-box ham">

                <div class="result-title">
                ✅ NOT SPAM
                </div>

                <div class="confidence">
                Confidence: <b>{confidence:.2f}%</b>
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "This message appears to be legitimate."
            )


        # ======================================
        # PROBABILITY DETAILS
        # ======================================

        st.subheader("📊 Prediction Details")

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Not Spam Probability**")

            st.progress(
                float(probabilities[0])
            )

            st.write(
                f"{probabilities[0] * 100:.2f}%"
            )

        with col2:

            st.write("**Spam Probability**")

            st.progress(
                float(probabilities[1])
            )

            st.write(
                f"{probabilities[1] * 100:.2f}%"
            )


# ==========================================
# MODEL INFORMATION
# ==========================================

st.divider()

st.subheader("🤖 Machine Learning Model")

info1, info2, info3 = st.columns(3)

with info1:

    st.markdown("""
    <div class="info-card">

    <h4>🔤 TF-IDF</h4>

    Converts email text into numerical
    features that the ML model can understand.

    </div>
    """, unsafe_allow_html=True)


with info2:

    st.markdown("""
    <div class="info-card">

    <h4>🧠 Logistic Regression</h4>

    Classifies messages into Spam
    and Not Spam categories.

    </div>
    """, unsafe_allow_html=True)


with info3:

    st.markdown("""
    <div class="info-card">

    <h4>📈 Performance</h4>

    Test accuracy:
    <b>96.95%</b>

    </div>
    """, unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <br><br>
    <center>
    <small>
    Spam Email Classifier • AI Internship Project
    </small>
    </center>
    """,
    unsafe_allow_html=True
)