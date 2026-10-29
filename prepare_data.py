"""Download and prepare the Steel Plates Faults dataset for ML Assignment 2.

Steps:
- Download the dataset from the UCI ML Repository (id 198) with ucimlrepo.
- Save the raw data to data/steel_plates_raw.csv.
- Check that every row has exactly one of the 7 fault columns set to 1.
- Combine the 7 fault columns into a single text target column, Fault_Type.
- Save the cleaned data to data/steel_plates_clean.csv.
- Make a stratified 80/20 train/test split and save data/train_data.csv
  and test_data.csv (repo root, used as the sample upload file for the app).

ucimlrepo is only needed here, so it is not listed in requirements.txt.
Install it with: pip install ucimlrepo
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

DATASET_ID = 198
TARGET = "Fault_Type"
FAULT_COLUMNS = [
    "Pastry",
    "Z_Scratch",
    "K_Scratch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]
TEST_SIZE = 0.2
RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "steel_plates_raw.csv"
CLEAN_PATH = DATA_DIR / "steel_plates_clean.csv"
TRAIN_PATH = DATA_DIR / "train_data.csv"
TEST_PATH = ROOT / "test_data.csv"


def download_raw() -> pd.DataFrame:
    dataset = fetch_ucirepo(id=DATASET_ID)
    # "original" keeps the UCI column order: 27 features followed by 7 faults.
    return dataset.data.original.copy()


def check_one_fault_per_row(df: pd.DataFrame) -> None:
    fault_counts = df[FAULT_COLUMNS].sum(axis=1)
    bad_rows = df.index[fault_counts != 1]
    if len(bad_rows) == 0:
        print("Check passed: every row has exactly one fault set to 1.")
        return
    print(f"Check FAILED: {len(bad_rows)} rows do not have exactly one fault set to 1.")
    print(df.loc[bad_rows, FAULT_COLUMNS].assign(n_faults=fault_counts[bad_rows]))
    raise SystemExit("Fix the fault labels before continuing.")


def combine_fault_columns(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.drop(columns=FAULT_COLUMNS)
    clean[TARGET] = df[FAULT_COLUMNS].idxmax(axis=1)
    return clean


def print_distribution(name: str, df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts()
    percents = (counts / len(df) * 100).round(1)
    table = pd.DataFrame({"count": counts, "percent": percents})
    print(f"\n{name} set: {len(df)} rows")
    print(table.to_string())


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    raw = download_raw()
    raw.to_csv(RAW_PATH, index=False)
    print(f"Saved raw data: {RAW_PATH.relative_to(ROOT)} {raw.shape}")

    missing_faults = set(FAULT_COLUMNS) - set(raw.columns)
    if missing_faults:
        raise SystemExit(f"Fault columns not found in raw data: {sorted(missing_faults)}")

    check_one_fault_per_row(raw)

    clean = combine_fault_columns(raw)
    n_features = clean.shape[1] - 1
    clean.to_csv(CLEAN_PATH, index=False)
    print(f"Saved clean data: {CLEAN_PATH.relative_to(ROOT)} {clean.shape} "
          f"({n_features} features + {TARGET})")

    train, test = train_test_split(
        clean,
        test_size=TEST_SIZE,
        stratify=clean[TARGET],
        random_state=RANDOM_STATE,
    )
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"Saved train data: {TRAIN_PATH.relative_to(ROOT)} {train.shape}")
    print(f"Saved test data: {TEST_PATH.relative_to(ROOT)} {test.shape}")

    print_distribution("Full", clean)
    print_distribution("Train", train)
    print_distribution("Test", test)


if __name__ == "__main__":
    main()
