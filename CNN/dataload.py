import os
import pandas as pd
from sklearn . model_selection import train_test_split

BASE_DIR = os.path.join("CNN", "cats_and_dogs_filtered")

def make_df(animal):
    rows = []
    for label in ["cats", "dogs"]:
        folder = os.path.join(BASE_DIR, animal, label)
        for f in os.listdir(folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({"filepath": os.path.join(folder, f),
                             "label": label[:-1]})
    return pd.DataFrame(rows)

train_full_df = make_df("train")
val_df = make_df("validation")

print(train_full_df.head())
print(train_full_df.shape, val_df.shape)

counts = pd.DataFrame({
    "train": train_full_df["label"].value_counts(),
    "validation": val_df["label"].value_counts(),
})
print(counts)
counts.plot(kind="bar", title="Class distribution")

df = pd.concat([train_full_df, val_df])

X_train, X_val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

print(len(X_train), len(X_val))

