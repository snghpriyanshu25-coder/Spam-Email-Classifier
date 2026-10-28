import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


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

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})


# ==========================================
# 2. LOAD SAVED MODEL
# ==========================================

print("Loading trained model...")

model = joblib.load(
    "model/spam_classifier.pkl"
)

vectorizer = joblib.load(
    "model/tfidf_vectorizer.pkl"
)


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


# ==========================================
# 4. TRANSFORM TEST DATA
# ==========================================

X_test_tfidf = vectorizer.transform(X_test)


# ==========================================
# 5. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 6. CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

matrix = confusion_matrix(
    y_test,
    y_pred
)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print()
print("==========================================")
print("SPAM EMAIL CLASSIFIER - EVALUATION")
print("==========================================")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print()
print("Confusion Matrix:")
print(matrix)

print()
print("==========================================")
print("EVALUATION COMPLETED")
print("==========================================")