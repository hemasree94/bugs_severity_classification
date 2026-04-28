# dag.py — Data Engineering DAG (expanded steps)

import os
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

# import your functions
from preprocess import remove_columns, clean_text, label_encode, split_data
from transform import generate_embeddings
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
PROJECT_ROOT = "/home/hemasree/bugs_severity_classification"

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/bugs.csv")
CLEAN_PATH = os.path.join(PROJECT_ROOT, "data/processed/clean.csv")

TRAIN_PATH = os.path.join(PROJECT_ROOT, "data/processed/train.csv")
VAL_PATH = os.path.join(PROJECT_ROOT, "data/processed/val.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "data/processed/test.csv")

FEATURE_PATH = os.path.join(PROJECT_ROOT, "data/features/features.parquet")

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# GLOBAL SHARED DATA (simple approach using files)
# -------------------------------------------------------------------

# TASK 1
def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    df.to_csv(CLEAN_PATH, index=False)
    logger.info(f"Loaded {len(df)} rows")


# TASK 2
def remove_cols():
    df = pd.read_csv(CLEAN_PATH)
    df = remove_columns(df, ["unnecessary_column"])  # adjust
    df.to_csv(CLEAN_PATH, index=False)


# TASK 3
def clean_text_task():
    df = pd.read_csv(CLEAN_PATH)
    df = clean_text(df, "summary")
    df.to_csv(CLEAN_PATH, index=False)


# TASK 4
def encode_labels():
    df = pd.read_csv(CLEAN_PATH)

    label_map = {"low": 0, "medium": 1, "high": 2}
    df = label_encode(df, "severity", label_map)

    df.to_csv(CLEAN_PATH, index=False)


# TASK 5
def split_task():
    df = pd.read_csv(CLEAN_PATH)

    train_df, val_df, test_df = split_data(
        df,
        target_column="severity",
        test_size=0.2
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)


# TASK 6
def embedding_task():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    df = pd.read_csv(TRAIN_PATH)

    embeddings = generate_embeddings(
        model,
        df["summary"].fillna("").tolist()
    )

    df["embedding"] = [x.tolist() for x in embeddings]

    df.to_parquet(FEATURE_PATH, index=False)

    logger.info("Embeddings generated")


# -------------------------------------------------------------------
# DAG
# -------------------------------------------------------------------
default_args = {
    "owner": "hemasree",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="bugs_data_engineering_detailed",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 4, 27, tzinfo=timezone.utc),
    catchup=False,
) as dag:

    t_load = PythonOperator(task_id="load_data", python_callable=load_data)

    t_remove = PythonOperator(task_id="remove_columns", python_callable=remove_cols)

    t_clean = PythonOperator(task_id="clean_text", python_callable=clean_text_task)

    t_encode = PythonOperator(task_id="label_encode", python_callable=encode_labels)

    t_split = PythonOperator(task_id="split_data", python_callable=split_task)

    t_embed = PythonOperator(task_id="generate_embeddings", python_callable=embedding_task)

    t_end = EmptyOperator(task_id="end")

    # FLOW
    t_load >> t_remove >> t_clean >> t_encode >> t_split >> t_embed >> t_end