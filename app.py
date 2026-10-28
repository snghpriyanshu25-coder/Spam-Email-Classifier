import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from datetime import datetime


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SESSION STATE
# =========================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "model/spam_classifier.pkl"
    )

    vectorizer = joblib.load(
        "model/tfidf_vectorizer.pkl"
    )

    return model, vectorizer


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    data = pd.read_csv(
        "data/SMSSpamCollection",
        sep="\t",
        names=["label", "message"]
    )

    data["label"] = data["label"].map({
        "ham": 0,
        "spam": 1
    })

    return data


model, vectorizer = load_model()
data = load_dataset()


# =========================================================
# DATASET STATISTICS
# =========================================================

total_messages = len(data)

spam_messages = int(
    (data["label"] == 1).sum()
)

ham_messages = int(
    (data["label"] == 0).sum()
)

spam_percentage = (
    spam_messages / total_messages
) * 100

ham_percentage = (
    ham_messages / total_messages
) * 100


# =========================================================
# MODEL EVALUATION
# =========================================================

X = data["message"]
y = data["label"]

X_vectorized = vectorizer.transform(X)

predictions = model.predict(
    X_vectorized
)

accuracy = accuracy_score(
    y,
    predictions
)

precision = precision_score(
    y,
    predictions,
    zero_division=0
)

recall = recall_score(
    y,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y,
    predictions
)


# =========================================================
# THEME
# =========================================================

if st.session_state.theme == "Dark":

    background = "#0f172a"
    card = "#1e293b"
    text = "#f8fafc"
    muted = "#94a3b8"
    border = "#334155"
    sidebar = "#020617"

else:

    background = "#f8fafc"
    card = "#ffffff"
    text = "#0f172a"
    muted = "#64748b"
    border = "#e2e8f0"
    sidebar = "#ffffff"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    f"""
<style>

.stApp {{
    background: {background};
    color: {text};
}}

section[data-testid="stSidebar"] {{
    background: {sidebar};
    border-right: 1px solid {border};
}}

section[data-testid="stSidebar"] * {{
    color: {text};
}}

h1, h2, h3, h4, p, label {{
    color: {text} !important;
}}

.hero {{
    padding: 35px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        #020617,
        #1e293b,
        #334155
    );
    color: white;
    margin-bottom: 28px;
    border: 1px solid #475569;
    box-shadow: 0 12px 35px rgba(0,0,0,0.20);
}}

.hero h1 {{
    color: white !important;
    font-size: 44px;
    margin-bottom: 8px;
}}

.hero p {{
    color: #cbd5e1 !important;
    font-size: 18px;
}}

.card {{
    padding: 22px;
    border-radius: 16px;
    background: {card};
    border: 1px solid {border};
    box-shadow: 0 5px 18px rgba(0,0,0,0.08);
}}

.result-spam {{
    padding: 28px;
    border-radius: 18px;
    background: #450a0a;
    border: 2px solid #ef4444;
    text-align: center;
    margin: 22px 0;
}}

.result-safe {{
    padding: 28px;
    border-radius: 18px;
    background: #052e16;
    border: 2px solid #22c55e;
    text-align: center;
    margin: 22px 0;
}}

.result-spam h1,
.result-safe h1 {{
    color: white !important;
}}

.result-spam p,
.result-safe p {{
    color: #e2e8f0 !important;
}}

.footer {{
    text-align: center;
    padding: 35px;
    color: {muted};
}}

.small-text {{
    color: {muted};
    font-size: 13px;
}}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:12px 5px 8px 5px;
        ">

            <div style="font-size:55px;">
                🛡️
            </div>

            <h2>
                SpamGuard AI
            </h2>

            <p style="color:#94a3b8;">
                Intelligent Message Security
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # -----------------------------------------------------
    # THEME TOGGLE
    # -----------------------------------------------------

    st.markdown("### 🎨 Appearance")

    theme_choice = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=0 if st.session_state.theme == "Dark" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )

    if theme_choice != st.session_state.theme:

        st.session_state.theme = theme_choice

        st.rerun()

    st.divider()

    # -----------------------------------------------------
    # NAVIGATION
    # -----------------------------------------------------

    st.markdown("### 📌 Navigation")

    page = st.radio(
        "Select Section",
        [
            "🏠 Dashboard",
            "📧 Message Analyzer",
            "📊 Analytics",
            "🔢 Model Evaluation",
            "🕘 Prediction History",
            "⚙️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    # -----------------------------------------------------
    # MODEL STATUS
    # -----------------------------------------------------

    st.markdown("### 🤖 Model Status")

    st.success("● MODEL ONLINE")

    st.write(
        "**Algorithm:** Logistic Regression"
    )

    st.write(
        "**Features:** TF-IDF"
    )

    st.write(
        f"**Accuracy:** {accuracy * 100:.2f}%"
    )

    st.divider()

    # -----------------------------------------------------
    # DATASET
    # -----------------------------------------------------

    st.markdown("### 📚 Dataset")

    st.write(
        f"Messages: **{total_messages:,}**"
    )

    st.write(
        f"Spam: **{spam_messages:,}**"
    )

    st.write(
        f"Ham: **{ham_messages:,}**"
    )

    st.write(
        f"Spam Rate: **{spam_percentage:.2f}%**"
    )

    st.divider()

    # -----------------------------------------------------
    # QUICK STATUS
    # -----------------------------------------------------

    st.markdown("### ⚡ Quick Status")

    st.write(
        f"Predictions: **{len(st.session_state.prediction_history)}**"
    )

    st.write(
        "System: **Ready**"
    )

    st.divider()

    st.caption("AI Internship Project")
    st.caption("Python • ML • Streamlit")


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="hero">

            <h1>🛡️ SpamGuard AI</h1>

            <p>
            AI-powered spam detection system using
            Natural Language Processing and Machine Learning.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("🤖 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

    st.divider()

    st.subheader("📚 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Messages",
            f"{total_messages:,}"
        )

    with col2:

        st.metric(
            "Spam Messages",
            f"{spam_messages:,}"
        )

    with col3:

        st.metric(
            "Ham Messages",
            f"{ham_messages:,}"
        )

    with col4:

        st.metric(
            "Spam Rate",
            f"{spam_percentage:.2f}%"
        )

    st.subheader("📈 Message Distribution")

    chart_data = pd.DataFrame(
        {
            "Category": [
                "Ham",
                "Spam"
            ],
            "Messages": [
                ham_messages,
                spam_messages
            ]
        }
    )

    st.bar_chart(
        chart_data.set_index("Category")
    )

    st.divider()

    st.subheader("🚀 System Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            ### 🧠 AI Detection

            TF-IDF text features combined with
            Logistic Regression classification.
            """
        )

    with col2:

        st.markdown(
            """
            ### ⚡ Real-Time Analysis

            Analyze any email or SMS and receive
            an instant classification.
            """
        )

    with col3:

        st.markdown(
            """
            ### 📊 Smart Analytics

            Monitor dataset statistics,
            model metrics and predictions.
            """
        )


# =========================================================
# MESSAGE ANALYZER
# =========================================================

elif page == "📧 Message Analyzer":

    st.markdown(
        """
        <div class="hero">

            <h1>📧 Message Analyzer</h1>

            <p>
            Enter an email or SMS message and let
            SpamGuard AI analyze it.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    email_text = st.text_area(
        "Enter Email / Message",
        height=200,
        placeholder=(
            "Example: Congratulations! "
            "You have won a free prize. "
            "Click here to claim..."
        )
    )

    analyze_button = st.button(
        "🔍 Analyze Message",
        use_container_width=True
    )

    if analyze_button:

        if not email_text.strip():

            st.warning(
                "⚠️ Please enter a message first."
            )

        else:

            message_vector = vectorizer.transform(
                [email_text]
            )

            prediction = model.predict(
                message_vector
            )[0]

            probability = model.predict_proba(
                message_vector
            )[0]

            ham_probability = probability[0] * 100
            spam_probability = probability[1] * 100

            confidence = max(
                ham_probability,
                spam_probability
            )

            if prediction == 1:

                result = "SPAM"

                st.markdown(
                    """
                    <div class="result-spam">

                        <h1>🚨 SPAM DETECTED</h1>

                        <p>
                        This message has been classified
                        as spam by the AI model.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                result = "NOT SPAM"

                st.markdown(
                    """
                    <div class="result-safe">

                        <h1>✅ NOT SPAM</h1>

                        <p>
                        This message appears to be legitimate.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Prediction",
                    result
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with col3:

                st.metric(
                    "Spam Probability",
                    f"{spam_probability:.2f}%"
                )

            st.subheader(
                "🎯 Prediction Probability"
            )

            st.write(
                f"**Not Spam:** {ham_probability:.2f}%"
            )

            st.progress(
                min(ham_probability / 100, 1.0)
            )

            st.write(
                f"**Spam:** {spam_probability:.2f}%"
            )

            st.progress(
                min(spam_probability / 100, 1.0)
            )

            # -------------------------------------------------
            # SAVE PREDICTION HISTORY
            # -------------------------------------------------

            st.session_state.prediction_history.append(
                {
                    "Time":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "Message":
                        email_text[:100],

                    "Prediction":
                        result,

                    "Confidence":
                        f"{confidence:.2f}%",

                    "Spam Probability":
                        f"{spam_probability:.2f}%"
                }
            )

            st.success(
                "✅ Prediction saved to history."
            )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "📊 Analytics":

    st.markdown(
        """
        <div class="hero">

            <h1>📊 Analytics Dashboard</h1>

            <p>
            Explore message distribution and dataset statistics.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📈 Message Distribution")

    chart_data = pd.DataFrame(
        {
            "Category": [
                "Ham",
                "Spam"
            ],
            "Messages": [
                ham_messages,
                spam_messages
            ]
        }
    )

    st.bar_chart(
        chart_data.set_index("Category")
    )

    st.divider()

    st.subheader("📊 Message Percentage")

    percentage_data = pd.DataFrame(
        {
            "Category": [
                "Ham",
                "Spam"
            ],
            "Percentage": [
                ham_percentage,
                spam_percentage
            ]
        }
    )

    st.bar_chart(
        percentage_data.set_index("Category")
    )

    st.divider()

    st.subheader("📌 Dataset Statistics")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total Dataset",
            f"{total_messages:,}"
        )

        st.metric(
            "Spam Messages",
            f"{spam_messages:,}"
        )

    with col2:

        st.metric(
            "Ham Messages",
            f"{ham_messages:,}"
        )

        st.metric(
            "Spam Rate",
            f"{spam_percentage:.2f}%"
        )


# =========================================================
# MODEL EVALUATION
# =========================================================

elif page == "🔢 Model Evaluation":

    st.markdown(
        """
        <div class="hero">

            <h1>🔢 Model Evaluation</h1>

            <p>
            Detailed performance metrics of the
            trained spam classification model.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

    st.divider()

    st.subheader("🔢 Confusion Matrix")

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Ham",
            "Actual Spam"
        ],
        columns=[
            "Predicted Ham",
            "Predicted Spam"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("📖 Metric Explanation")

    st.markdown(
        """
        **Accuracy**  
        Percentage of all messages classified correctly.

        **Precision**  
        Percentage of messages predicted as spam
        that were actually spam.

        **Recall**  
        Percentage of actual spam messages
        detected by the model.

        **F1 Score**  
        Combined measure of precision and recall.
        """
    )


# =========================================================
# PREDICTION HISTORY
# =========================================================

elif page == "🕘 Prediction History":

    st.markdown(
        """
        <div class="hero">

            <h1>🕘 Prediction History</h1>

            <p>
            Review and export messages analyzed during
            this application session.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.prediction_history:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Prediction History",
            data=csv_data,
            file_name="spam_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.write("")

        if st.button(
            "🗑️ Clear Prediction History",
            use_container_width=True
        ):

            st.session_state.prediction_history = []

            st.rerun()

    else:

        st.info(
            "No predictions yet. "
            "Go to Message Analyzer and analyze a message."
        )


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "⚙️ About Project":

    st.markdown(
        """
        <div class="hero">

            <h1>⚙️ About SpamGuard AI</h1>

            <p>
            Machine Learning based spam message
            classification system.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("🧠 Technology Stack")

    tech_data = pd.DataFrame(
        {
            "Component": [
                "Programming Language",
                "Machine Learning Algorithm",
                "Feature Extraction",
                "Web Framework",
                "Dataset"
            ],

            "Technology": [
                "Python",
                "Logistic Regression",
                "TF-IDF",
                "Streamlit",
                "UCI SMS Spam Collection"
            ]
        }
    )

    st.dataframe(
        tech_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("📚 Project Workflow")

    st.markdown(
        """
        ### 1️⃣ Data Collection

        SMS messages are collected from the
        UCI SMS Spam Collection dataset.

        ### 2️⃣ Data Processing

        Text messages are converted into numerical
        features using TF-IDF.

        ### 3️⃣ Model Training

        Logistic Regression is trained to classify
        messages into Ham and Spam.

        ### 4️⃣ Model Evaluation

        Accuracy, Precision, Recall and F1 Score
        are calculated.

        ### 5️⃣ Real-Time Prediction

        Users can enter a new message and receive
        an instant prediction.

        ### 6️⃣ Analytics

        Dataset statistics, probabilities and
        prediction history are displayed.
        """
    )

    st.divider()

    st.subheader("📊 Current Model")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        st.write(
            "**Algorithm:** Logistic Regression"
        )

    with col2:

        st.metric(
            "Dataset Size",
            f"{total_messages:,}"
        )

        st.write(
            "**Features:** TF-IDF"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        <hr>

        🛡️ <b>SpamGuard AI</b>

        <br><br>

        AI Internship Project

        <br>

        Python • Machine Learning • Scikit-learn • Streamlit

        <br><br>

        TF-IDF + Logistic Regression

    </div>
    """,
    unsafe_allow_html=True
)
