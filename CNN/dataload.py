import os
import pandas as pd

BASE_DIR = os.path.join("CNN", "cats_and_dogs_filtered")

def make_df(split):
    rows = []
    for label in ["cats", "dogs"]:
        folder = os.path.join(BASE_DIR, split, label)
        for f in os.listdir(folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({"filepath": os.path.join(folder, f),
                             "label": label[:-1]})   # "cat" / "dog"
    return pd.DataFrame(rows)

train_full_df = make_df("train")
val_df = make_df("validation")

print(train_full_df.head())
print(train_full_df.shape, val_df.shape)