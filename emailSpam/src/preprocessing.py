import pandas as pd
import re

df = pd.read_csv("../dataset/email_spam_dataset.csv")

print("Original dataset:", df.shape)

df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

def clean_text(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r"<.*?>", " ", text) # Remove HTML tags
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text) # Remove URLs
    text = re.sub(r"\S+@\S+", " ", text) # Remove email addresses
    text = text.replace("\\n", " ") # Remove newline characters
    text = re.sub(r"[^a-zA-Z\s']", " ", text) # Remove special characters
    text = re.sub(r"\s+", " ", text).strip() # Remove extra whitespace
    return text

df["clean_text"] = df["text"].apply(clean_text)

df.to_csv("../dataset/processed_email_spam.csv", index=False)

print("Processing complete!")
print("Processed dataset:", df.shape)

print("\nExample cleaned emails:")
print(df[["text", "clean_text", "label"]].head())

print("\nOriginal email:")
print(df["text"].iloc[0])

print("\nCleaned email:")
print(df["clean_text"].iloc[0])

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nGroup and label:")
print(pd.crosstab(df["group"], df["label"]))