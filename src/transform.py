import pandas as pd
import numpy as np
import logging
import os
import argparse
from sentence_transformers import SentenceTransformer
from database import init_db
from sqlalchemy import text

# ---------------- LOGGING ----------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "transform.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ---------------- DB INSERT ----------------
def insert_embeddings(engine, df, table_name):
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text(f"""
                    INSERT INTO {table_name} (id, summary, severity, embedding)
                    VALUES (:id, :summary, :severity, :embedding)
                    ON CONFLICT (id) DO NOTHING
                """),
                {
                    "id": int(row["id"]),
                    "summary": row["summary"],
                    "severity": int(row["severity"]),
                    "embedding": "[" + ",".join(map(str, row["embedding"])) + "]"
                }
            )

# ---------------- EMBEDDINGS ----------------
def generate_embeddings(model, texts):
    return model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

# ---------------- MAIN ----------------
def run_transform(train_path, val_path, test_path,out_train, out_val, out_test):

    logger.info("Loading data from files")

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    text_column = "summary"
    target_column = "severity"

    # Load model
    logger.info("Loading SentenceTransformer model")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings
    logger.info("Generating train embeddings")
    X_train = generate_embeddings(model, train_df[text_column].fillna("").tolist())

    logger.info("Generating val embeddings")
    X_val = generate_embeddings(model, val_df[text_column].fillna("").tolist())

    logger.info("Generating test embeddings")
    X_test = generate_embeddings(model, test_df[text_column].fillna("").tolist())

    # Attach embeddings
    train_df["embedding"] = [x.tolist() for x in X_train]
    val_df["embedding"]   = [x.tolist() for x in X_val]
    test_df["embedding"]  = [x.tolist() for x in X_test]

    # ✅ ensure output directory exists
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    os.makedirs(os.path.dirname(out_val), exist_ok=True)
    os.makedirs(os.path.dirname(out_test), exist_ok=True)

    # Save to files (DVC tracking)
    train_df.to_parquet(out_train, index=False)
    val_df.to_parquet(out_val, index=False)
    test_df.to_parquet(out_test, index=False)

    logger.info("Saved embeddings to files")

    # ✅ OPTIONAL: save to DB

    logger.info("Transform complete")

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--out_train", default="data/features/train.parquet")
    parser.add_argument("--out_val", default="data/features/val.parquet")
    parser.add_argument("--out_test", default="data/features/test.parquet")

    args = parser.parse_args()

    run_transform(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        out_train=args.out_train,
        out_val=args.out_val,
        out_test=args.out_test
    )
    