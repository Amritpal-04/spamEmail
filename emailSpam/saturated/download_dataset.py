import pandas as pd
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

url = "https://huggingface.co/datasets/talby/spamassassin/resolve/refs%2Fconvert%2Fparquet/text/train/0000.parquet"

df = pd.read_parquet(url)

print(df.head())
print("Total emails:", len(df))

df.to_csv(DATASET_DIR / "email_spam_dataset.csv", index=False)

print("CSV file created successfully!")