from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib
from pathlib import Path

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# 1. Load processed dataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
df = pd.read_csv(PROJECT_DIR / "dataset" / "processed_email_spam.csv")

print("Dataset loaded successfully!")
print("Total emails:", len(df))


# 2. Remove invalid rows

df = df.dropna(subset=["clean_text", "label"])

df["clean_text"] = df["clean_text"].astype(str)

print("Valid emails:", len(df))


# 3. Separate input and output

X_text = df["clean_text"]
y = df["label"]


# 4. Train-Test Split

X_train_text, X_test_text, y_train, y_test, train_index, test_index = train_test_split(
    X_text,
    y,
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training emails:", len(X_train_text))
print("Testing emails:", len(X_test_text))


# 5. TF-IDF Feature Extraction

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

print("TF-IDF conversion complete!")
print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# 6. Select Engineered Features

engineered_features = [
    "char_count",
    "word_count",
    "url_count",
    "email_count",
    "digit_count",
    "special_char_count",
    "exclamation_count",
    "uppercase_count"
]

X_train_engineered = df.loc[
    train_index, engineered_features
].values

X_test_engineered = df.loc[
    test_index, engineered_features
].values


# 7. Scale engineered features

scaler = MinMaxScaler()

X_train_engineered = scaler.fit_transform(
    X_train_engineered
)

X_test_engineered = scaler.transform(
    X_test_engineered
)


# Convert to sparse matrix

X_train_engineered = csr_matrix(X_train_engineered)
X_test_engineered = csr_matrix(X_test_engineered)


# 8. Combine TF-IDF + Engineered Features

X_train = hstack([
    X_train_tfidf,
    X_train_engineered
])

X_test = hstack([
    X_test_tfidf,
    X_test_engineered
])


print("Feature engineering complete!")
print("Combined training shape:", X_train.shape)
print("Combined testing shape:", X_test.shape)


# 9. Store model results

results = []


# 10. Naive Bayes

print("\nTraining: Naive Bayes")

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_pred)
nb_precision = precision_score(y_test, nb_pred, pos_label=0)
nb_recall = recall_score(y_test, nb_pred, pos_label=0)
nb_f1 = f1_score(y_test, nb_pred, pos_label=0)

print("Accuracy :", round(nb_accuracy, 4))
print("Precision:", round(nb_precision, 4))
print("Recall   :", round(nb_recall, 4))
print("F1 Score :", round(nb_f1, 4))

results.append([
    "Naive Bayes",
    nb_accuracy,
    nb_precision,
    nb_recall,
    nb_f1
])


# 11. Logistic Regression

print("\nTraining: Logistic Regression")

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred, pos_label=0)
lr_recall = recall_score(y_test, lr_pred, pos_label=0)
lr_f1 = f1_score(y_test, lr_pred, pos_label=0)

print("Accuracy :", round(lr_accuracy, 4))
print("Precision:", round(lr_precision, 4))
print("Recall   :", round(lr_recall, 4))
print("F1 Score :", round(lr_f1, 4))

results.append([
    "Logistic Regression",
    lr_accuracy,
    lr_precision,
    lr_recall,
    lr_f1
])


# 12. Linear SVM

print("\nTraining: SVM")

svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred, pos_label=0)
svm_recall = recall_score(y_test, svm_pred, pos_label=0)
svm_f1 = f1_score(y_test, svm_pred, pos_label=0)

print("Accuracy :", round(svm_accuracy, 4))
print("Precision:", round(svm_precision, 4))
print("Recall   :", round(svm_recall, 4))
print("F1 Score :", round(svm_f1, 4))

results.append([
    "SVM",
    svm_accuracy,
    svm_precision,
    svm_recall,
    svm_f1
])


# 13. Compare Models

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\nFinal Results:")
print(results_df)


# 14. Confusion Matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, svm_pred))


# 15. Classification Report

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        svm_pred,
        target_names=["Spam", "Ham"]
    )
)


# 16. Save Model and Vectorizer

joblib.dump(
    svm_model,
    MODELS_DIR / "spam_svm_model.pkl"
)

joblib.dump(
    vectorizer,
    MODELS_DIR / "tfidf_vectorizer.pkl"
)

print("\nModel and Vectorizer saved successfully!")