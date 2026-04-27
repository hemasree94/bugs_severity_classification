import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
import argparse
import yaml
from database import init_db

params_file = yaml.safe_load(open("params.yaml"))

# ---------------- LOGGING ----------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "preprocess.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ---------------- FUNCTIONS ----------------

def remove_columns(df, columns_to_remove):
    return df.drop(columns=columns_to_remove)

def clean_text(df, text_column):
    df[text_column] = df[text_column].str.lower()
    df[text_column] = df[text_column].str.replace(r'[^\w\s]', '', regex=True)
    return df

def label_encode(df, column_to_encode, label_map):
    df[column_to_encode] = df[column_to_encode].map(label_map)
    return df

def split_data(df, target_column, test_size, val_size=0.1, random_state=42):
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column]
    )

    relative_val_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_df[target_column]
    )

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df

# ---------------- MAIN PIPELINE ----------------

def preprocess_data(columns_to_remove, text_column, column_to_encode, label_map, target_column, test_size, random_state, input_path, train_path, val_path, test_path):
    df = pd.read_csv(input_path)

    # params
    

    # processing
    df = remove_columns(df, columns_to_remove)
    df = clean_text(df, text_column)
    df = label_encode(df, column_to_encode, label_map)

    # split
    train_df, val_df, test_df = split_data(
        df,
        target_column=target_column,
        test_size=test_size,
        val_size=0.1,
        random_state=random_state
    )
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Saved processed datasets to files")

if __name__ == "__main__":
    columns_to_remove = params_file['prepare']['columns_to_remove']
    text_column = params_file['prepare']['text_column']
    column_to_encode = params_file['prepare']['column_to_encode']
    label_map = params_file['prepare']['label_map']
    target_column = params_file['prepare']['column_to_encode']
    test_size = params_file['prepare']['test_size']
    random_state = params_file['prepare']['random_state']
    parser = argparse.ArgumentParser()
    parser.add_argument("--columns-to-remove", default=columns_to_remove)
    parser.add_argument("--text-column", default=text_column)
    parser.add_argument("--column-to-encode", default=column_to_encode)
    parser.add_argument("--label-map", default=label_map)
    parser.add_argument("--target-column", default=target_column)
    parser.add_argument("--test-size", type=float, default=test_size)
    parser.add_argument("--random-state", type=int, default=random_state)
    parser.add_argument("--input", default="data/bugs.csv")
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    args = parser.parse_args()

    preprocess_data(
        columns_to_remove=columns_to_remove,
        text_column=text_column,
        column_to_encode=column_to_encode,
        label_map=label_map,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        input_path=args.input,
        train_path=args.train,
        val_path=args.val,
        test_path=args.test
    )