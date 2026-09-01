# 📧 Email Spam Detection using AI/ML

## 1. About the Project

This project is an AI/ML-based Email Spam Detection system.

The main goal of this project is to build a machine learning model that can
classify an email into two categories:

- Spam → unwanted or suspicious email
- Ham → legitimate/normal email

Our project will take an email as input and predict whether it is Spam or Ham.

---

# 2. Current Project Status

The project is currently in the Machine Learning development stage.

### Completed so far:

- Dataset collected
- Dataset analyzed
- Duplicate emails removed
- Email text preprocessing completed
- TF-IDF feature extraction completed
- Dataset divided into training and testing sets
- Three ML models trained
- Models evaluated
- Best model selected
- SVM model saved
- TF-IDF vectorizer saved

### Current best model:

**Linear SVM**

Accuracy: **98.19%**

---

# 3. Project Workflow

Our overall project workflow is:

Dataset
   ↓
Data Analysis
   ↓
Data Cleaning
   ↓
Text Preprocessing
   ↓
TF-IDF
   ↓
Train/Test Split
   ↓
Machine Learning Models
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Save Model
   ↓
Email Prediction
   ↓
Web/Application Interface

Some stages are still remaining and will be completed by the team.

---

# 4. Dataset

We are using the SpamAssassin email dataset.

The dataset contains email messages categorized as:

- Ham
- Spam

The dataset originally contained:

**10,749 emails**

After removing duplicate emails:

**6,100 unique emails**

During preprocessing, 11 emails with missing cleaned text/label values were removed.

Therefore, the final dataset used for training contains:

**6,089 emails**

---

# 5. Dataset Columns

The original dataset contains three important columns:

| Column | Meaning |
|---|---|
| `text` | Original email content |
| `group` | Dataset category |
| `label` | Target value |

Our labels are:

```text
0 = Spam
1 = Ham


[✓] Dataset collection
[✓] Dataset analysis
[✓] Duplicate removal
[✓] Text preprocessing
[✓] TF-IDF
[✓] Train/Test Split
[✓] Model training
[✓] Model comparison
[✓] Model evaluation
[✓] Save best model

[ ] Build predict.py
[ ] Test model with new emails
[ ] Improve/experiment with preprocessing
[ ] Create application interface
[ ] Connect model with application
[ ] Test complete application
[ ] Prepare project report
[ ] Prepare PPT
[ ] Final presentation