import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
df = pd.read_csv(PROJECT_DIR / "dataset" / "email_spam_dataset.csv")
print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

print("\nSpam vs ham:")
print(df['label'].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())