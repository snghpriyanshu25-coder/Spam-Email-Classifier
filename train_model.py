import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

data = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Dataset loaded successfully!")
print("Total messages:", len(data))


# ==========================================
# 2. CONVERT LABELS
# ==========================================

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})


# ==========================================
# 3. SPLIT DATA
# ==========================================

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))


# ==========================================
# 4. TF-IDF VECTORIZATION
# ==========================================

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF completed!")


# ==========================================
# 5. TRAIN MACHINE LEARNING MODEL
# ==========================================

print("Training model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# ==========================================
# 6. TEST MODEL
# ==========================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print()
print("==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print()
print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]
))


# ==========================================
# 7. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "model/spam_classifier.pkl"
)

joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer.pkl"
)

print("==========================================")
print("MODEL SAVED SUCCESSFULLY!")
print("==========================================")

print("Model: model/spam_classifier.pkl")
print("Vectorizer: model/tfidf_vectorizer.pkl")