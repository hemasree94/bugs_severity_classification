"""
dag_data_engineering.py — Data Engineering Pipeline
Location: src/dag_data_engineering.py

Flow:
  1. load_data       — load raw bugs.csv
  2. remove_columns  — drop id, summary_len, summary_word_count
  3. clean_text      — lowercase, remove punctuation from summary
  4. encode_labels   — map severity strings to integers
  5. split_data      — train / val / test split
  6. generate_embeddings — sentence embeddings using all-MiniLM-L6-v2
  7. end
"""

import os
import sys
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

# --- Add project src to path so imports work ---
PROJECT_ROOT = "/home/hemasree/bugs_severity_classification"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from preprocess import remove_columns, clean_text, label_encode, split_data
from transform import generate_embeddings
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/bugs.csv")
CLEAN_PATH    = os.path.join(PROJECT_ROOT, "data/processed/clean.csv")
TRAIN_PATH    = os.path.join(PROJECT_ROOT, "data/processed/train.csv")
VAL_PATH      = os.path.join(PROJECT_ROOT, "data/processed/val.csv")
TEST_PATH     = os.path.join(PROJECT_ROOT, "data/processed/test.csv")
TRAIN_PARQUET = os.path.join(PROJECT_ROOT, "data/features/train.parquet")
VAL_PARQUET   = os.path.join(PROJECT_ROOT, "data/features/val.parquet")
TEST_PARQUET  = os.path.join(PROJECT_ROOT, "data/features/test.parquet")

# Correct label map matching your actual data
LABEL_MAP = {
    "trivial":  0,
    "minor":    1,
    "normal":   2,
    "major":    3,
    "critical": 4
}

# Columns to drop — not needed for ML
COLUMNS_TO_REMOVE = ["id", "summary_len", "summary_word_count"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# TASK 1: Load Data
# -------------------------------------------------------------------
def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    logger.info(f"Loaded {len(df)} rows from {RAW_DATA_PATH}")


# -------------------------------------------------------------------
# TASK 2: Remove Columns
# -------------------------------------------------------------------
def remove_cols():
    df = pd.read_csv(CLEAN_PATH)
    # Only drop columns that actually exist
    cols = [c for c in COLUMNS_TO_REMOVE if c in df.columns]
    df = remove_columns(df, cols)
    df.to_csv(CLEAN_PATH, index=False)
    logger.info(f"Removed columns: {cols}. Remaining: {list(df.columns)}")


# -------------------------------------------------------------------
# TASK 3: Clean Text
# -------------------------------------------------------------------
def clean_text_task():
    df = pd.read_csv(CLEAN_PATH)
    df = clean_text(df, "summary")
    df.to_csv(CLEAN_PATH, index=False)
    logger.info("Text cleaning done")


# -------------------------------------------------------------------
# TASK 4: Encode Labels
# -------------------------------------------------------------------
def encode_labels():
    df = pd.read_csv(CLEAN_PATH)
    df = label_encode(df, "severity", LABEL_MAP)
    # Drop rows where severity could not be mapped (NaN)
    before = len(df)
    df = df.dropna(subset=["severity"])
    df["severity"] = df["severity"].astype(int)
    after = len(df)
    if before != after:
        logger.warning(f"Dropped {before - after} rows with unmapped severity labels")
    df.to_csv(CLEAN_PATH, index=False)
    logger.info(f"Label encoding done. Unique values: {df['severity'].unique().tolist()}")


# -------------------------------------------------------------------
# TASK 5: Split Data
# -------------------------------------------------------------------
def split_task():
    df = pd.read_csv(CLEAN_PATH)
    train_df, val_df, test_df = split_data(
        df,
        target_column="severity",
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH,   index=False)
    test_df.to_csv(TEST_PATH,  index=False)
    logger.info(f"Split done — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


# -------------------------------------------------------------------
# TASK 6: Generate Embeddings (all 3 splits)
# -------------------------------------------------------------------
def embedding_task():
    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    os.makedirs(os.path.dirname(TRAIN_PARQUET), exist_ok=True)

    for path, out_path, split_name in [
        (TRAIN_PATH, TRAIN_PARQUET, "train"),
        (VAL_PATH,   VAL_PARQUET,   "val"),
        (TEST_PATH,  TEST_PARQUET,  "test"),
    ]:
        df = pd.read_csv(path)
        embeddings = generate_embeddings(model, df["summary"].fillna("").tolist())
        df["embedding"] = [x.tolist() for x in embeddings]
        df.to_parquet(out_path, index=False)
        logger.info(f"{split_name} embeddings saved to {out_path} ({len(df)} rows)")

    logger.info("All embeddings generated successfully")


# -------------------------------------------------------------------
# DAG DEFINITION
# -------------------------------------------------------------------
default_args = {
    "owner":       "hemasree",
    "retries":     1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="bugs_data_engineering_detailed",
    description="Data engineering pipeline: load → clean → encode → split → embed",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 4, 27, tzinfo=timezone.utc),
    catchup=False,
    tags=["bugs", "data-engineering"],
) as dag:

    t_load   = PythonOperator(task_id="load_data",            python_callable=load_data)
    t_remove = PythonOperator(task_id="remove_columns",       python_callable=remove_cols)
    t_clean  = PythonOperator(task_id="clean_text",           python_callable=clean_text_task)
    t_encode = PythonOperator(task_id="label_encode",         python_callable=encode_labels)
    t_split  = PythonOperator(task_id="split_data",           python_callable=split_task)
    t_embed  = PythonOperator(task_id="generate_embeddings",  python_callable=embedding_task)
    t_end    = EmptyOperator(task_id="end")

    t_load >> t_remove >> t_clean >> t_encode >> t_split >> t_embed >> t_end