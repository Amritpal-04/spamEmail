import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. Load Dataset
df = pd.read_csv("../dataset/processed_email_spam.csv")

print("Dataset loaded successfully!")
print("Total emails:", len(df))

# 2. Select Input and Output
# Remove emails with missing text
df = df.dropna(subset=["clean_text", "label"])

# Make sure email text is string
df["clean_text"] = df["clean_text"].astype(str)

x = df["clean_text"]
y = df["label"]

print("\nInput (x): email text")
print("Output (y): spam./ham label")
print("Valid emails:", len(df))

# 3. Train / Test split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.20,
    random_state = 42,
    stratify = y
)

print("\nTraining emails:", len(x_train))
print("Testing emails:", len(x_test))

# 4. TF-IDF

vectorizer = TfidfVectorizer(
    stop_words = "english",
    max_features = 50000
)

x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

print("\nTF-TDF conversion complete!")

print("Training data shape:", x_train_tfidf.shape)
print("Testing data shape:", x_test_tfidf.shape)

# 5. Define models

models = {
    "Naive Bayes" : MultinomialNB(),
    "Logistic Regression" : LogisticRegression(
        max_iter = 1000
    ),
    "SVM": LinearSVC()
}

# 6. Train and Evaluate models

results = []

for name, model in models.items():
    print("\nTraining:", name)
    
    # train model
    model.fit(x_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(x_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    precision = precision_score(
        y_test, 
        y_pred, 
        pos_label = 0
    )
    
    recall = recall_score(y_test, y_pred, pos_label = 0)
    
    f1 = f1_score(y_test, y_pred, pos_label = 0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })
    
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    
# 7. Display Final results
    
results_df = pd.DataFrame(results)
    
print("\nFinal Results:")
print(results_df)

# 8. Final SVM Model

svm_model = LinearSVC()

svm_model.fit(x_train_tfidf, y_train)
y_pred = svm_model.predict(x_test_tfidf)

# 9. Confusion matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# 10. Classification report 
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names = ["Spam", "Ham"]))

# 11. Save model and Vectorizer 

joblib.dump(svm_model, "../models/spam_svm_model.pkl")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
print("\nModel and Vectorizer saved successfully!")