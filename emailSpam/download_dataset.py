import pandas as pd

url = "https://huggingface.co/datasets/talby/spamassassin/resolve/refs%2Fconvert%2Fparquet/text/train/0000.parquet"

df = pd.read_parquet(url)

print(df.head())
print("Total emails:", len(df))

df.to_csv("email_spam_dataset.csv", index=False)

print("CSV file created successfully!")