import pandas as pd
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"

df = pd.read_csv(DATASET_DIR / "email_spam_dataset.csv")

print("Original dataset:", df.shape)

df = df.drop_duplicates()

print("After removing duplicates:", df.shape)


def clean_text(text):
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove literal \n
    text = text.replace("\\n", " ")

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Keep letters, spaces and apostrophes
    text = re.sub(r"[^a-zA-Z\s']", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_text"] = df["text"].apply(clean_text)

# Feature Engineering

# Number of characters in the original email
df["char_count"] = df["text"].astype(str).apply(len)

# Number of words in the original email
df["word_count"] = df["text"].astype(str).apply(
    lambda x: len(x.split())
)

# Number of URLs
df["url_count"] = df["text"].astype(str).apply(
    lambda x: len(re.findall(r"http\S+|www\S+", x))
)

# Number of email addresses
df["email_count"] = df["text"].astype(str).apply(
    lambda x: len(re.findall(r"\S+@\S+", x))
)

# Number of digits
df["digit_count"] = df["text"].astype(str).apply(
    lambda x: sum(c.isdigit() for c in x)
)

# Number of special characters
df["special_char_count"] = df["text"].astype(str).apply(
    lambda x: sum(not c.isalnum() and not c.isspace() for c in x)
)

# Number of exclamation marks
df["exclamation_count"] = df["text"].astype(str).apply(
    lambda x: x.count("!")
)

# Number of uppercase letters
df["uppercase_count"] = df["text"].astype(str).apply(
    lambda x: sum(c.isupper() for c in x)
)


df.to_csv(DATASET_DIR / "processed_email_spam.csv", index=False)


print("\nProcessing complete!")

print("Processed dataset:", df.shape)

print("\nExample processed emails:")
print(
    df[
        [
            "text",
            "clean_text",
            "char_count",
            "word_count",
            "url_count",
            "email_count",
            "digit_count",
            "special_char_count",
            "exclamation_count",
            "uppercase_count",
            "label"
        ]
    ].head()
)

print("\nFeature Engineering Summary:")
print(
    df[
        [
            "char_count",
            "word_count",
            "url_count",
            "email_count",
            "digit_count",
            "special_char_count",
            "exclamation_count",
            "uppercase_count"
        ]
    ].describe()
)

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nGroup and label:")
print(pd.crosstab(df["group"], df["label"]))