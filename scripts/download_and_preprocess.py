# scripts/download_and_preprocess.py
import os
import pandas as pd
from pathlib import Path
from src.data.preprocess import preprocess_and_save

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def download_adult(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    urls = {
        "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "adult.names": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
        "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    }
    for name,url in urls.items():
        out = dest / name
        if not out.exists():
            print(f"Downloading {name} ...")
            import requests
            r = requests.get(url)
            r.raise_for_status()
            out.write_bytes(r.content)
    # load train portion
    colnames = ["age","workclass","fnlwgt","education","education-num","marital-status",
                "occupation","relationship","race","sex","capital-gain","capital-loss",
                "hours-per-week","native-country","income"]
    df_train = pd.read_csv(dest/"adult.data", header=None, names=colnames, skipinitialspace=True)
    df_test = pd.read_csv(dest/"adult.test", header=0, names=colnames, skiprows=1, skipinitialspace=True)
    df = pd.concat([df_train, df_test], ignore_index=True)
    return df

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = download_adult(DATA_DIR)
    # optional: drop rows with missing '?'
    df = df.replace("?", pd.NA)
    df = df.dropna().reset_index(drop=True)
    # Preprocess and save
    out = preprocess_and_save(df, out_dir=PROCESSED_DIR/"adult", target_col="income")
    print("Saved processed files:", out)
