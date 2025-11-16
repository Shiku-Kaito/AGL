import pandas as pd
from huggingface_hub import login
import os
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)
import datasets
import os

def main():
    local_dir = "./data/org/hest_data/"  # hest will be dowloaded to this folder
    os.makedirs(local_dir, exist_ok=True)

    df = pd.read_csv("./data/org/HEST_v1_1_0.csv")
    df.head()
    ids_to_query = df[
        df["id"].isin(["TENX65", "TENX89", "TENX152"])
    ].id.values

    list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
    dataset = datasets.load_dataset(
        "MahmoodLab/hest", cache_dir=local_dir, patterns=list_patterns
    )


if __name__ == '__main__':
    main()