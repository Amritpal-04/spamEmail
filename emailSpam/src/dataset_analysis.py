import pandas as pd

df = pd.read_csv("../dataset/email_spam_dataset.csv")
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