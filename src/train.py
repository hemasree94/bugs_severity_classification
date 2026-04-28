import ast
import os, json, pickle, logging, subprocess, argparse
import numpy as np
import pandas as pd
import yaml, mlflow, mlflow.sklearn
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss,
    classification_report, confusion_matrix
)
from database import init_db


params_file = yaml.safe_load(open("params.yaml"))
LABEL_NAMES = ["trivial", "minor", "normal", "major", "critical"]

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "train.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)





def load_embeddings_from_files(train_path, val_path):
    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)

    logger.info(f"Train embeddings: {len(train_df)} | Val embeddings: {len(val_df)}")

    X_train = np.array(train_df["embedding"].tolist())
    X_val   = np.array(val_df["embedding"].tolist())

    y_train = train_df["severity"].values
    y_val   = val_df["severity"].values

    return X_train, X_val, y_train, y_val


def train_logistic_regression(X_train, y_train, X_val, y_val, params):
    model = LogisticRegression(
        max_iter=1,
        warm_start=True,
        class_weight="balanced",
        random_state=params["random_state"],
        solver="saga"
    )
    logger.info("Training logistic regression — iterative tracking")
    for i in range(1, 101):
        model.max_iter = i
        model.fit(X_train, y_train)

        train_loss = log_loss(y_train, model.predict_proba(X_train))
        val_loss   = log_loss(y_val,   model.predict_proba(X_val))
        val_f1     = f1_score(y_val, model.predict(X_val), average="weighted", zero_division=0)
        val_acc    = accuracy_score(y_val, model.predict(X_val))

        mlflow.log_metric("train_loss", train_loss, step=i)
        mlflow.log_metric("val_loss",   val_loss,   step=i)
        mlflow.log_metric("val_f1",     val_f1,     step=i)
        mlflow.log_metric("val_acc",    val_acc,    step=i)

        if i % 10 == 0:
            logger.info(f"  iter={i} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

    return model


def train_random_forest(X_train, y_train, X_val, y_val, params):
    model = RandomForestClassifier(
        n_estimators=10,
        class_weight="balanced",
        random_state=params["random_state"],
        warm_start=True,
        n_jobs=-1,
    )
    logger.info("Training random forest — tracking per 10 trees")
    for n_trees in range(10, params["n_estimators"] + 10, 10):
        model.n_estimators = n_trees
        model.fit(X_train, y_train)

        val_f1  = f1_score(y_val, model.predict(X_val), average="weighted", zero_division=0)
        val_acc = accuracy_score(y_val, model.predict(X_val))

        mlflow.log_metric("val_f1",  val_f1,  step=n_trees)
        mlflow.log_metric("val_acc", val_acc, step=n_trees)

        if n_trees % 50 == 0:
            logger.info(f"  n_trees={n_trees} | val_f1={val_f1:.4f} val_acc={val_acc:.4f}")

    return model


def log_final_metrics(model, X_val, y_val):
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1w = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    f1m = f1_score(y_val, y_pred, average="macro",    zero_division=0)
    per_class = f1_score(y_val, y_pred, average=None, zero_division=0,
                         labels=LABEL_NAMES)

    mlflow.log_metric("final_accuracy",    acc)
    mlflow.log_metric("final_f1_weighted", f1w)
    mlflow.log_metric("final_f1_macro",    f1m)

    for i, label in enumerate(LABEL_NAMES):
        if i < len(per_class):
            mlflow.log_metric(f"f1_{label}", float(per_class[i]))

        report = classification_report(
                y_val,
                y_pred,
                target_names=LABEL_NAMES,
                zero_division=0
            )
    logger.info(f"\n{report}")

    os.makedirs("metrics", exist_ok=True)

    with open("metrics/classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("metrics/classification_report.txt")

    cm = confusion_matrix(y_val, y_pred)
    with open("metrics/confusion_matrix.json", "w") as f:
        json.dump({"labels": LABEL_NAMES, "matrix": cm.tolist()}, f, indent=2)
    mlflow.log_artifact("metrics/confusion_matrix.json")

    mlflow.log_artifact("params.yaml")

    return acc, f1w



def run_training(train_path, val_path, model_output_path, metrics_path):
    p = params_file["train"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment("bug_severity_classification")

    X_train, X_val, y_train, y_val = load_embeddings_from_files(train_path, val_path)
    git_commit = subprocess.getoutput("git rev-parse --short HEAD")

    best_model, best_f1, best_run_id = None, 0, None

    for model_name in p["models"]:
        logger.info(f"\n{'='*40}")
        logger.info(f"Starting run: {model_name}")
        logger.info(f"{'='*40}")

        with mlflow.start_run(run_name=model_name) as run:
            mlflow.set_tag("model_name",      model_name)
            mlflow.set_tag("git_commit",      git_commit)
            mlflow.set_tag("dataset_version", "v1")

            mlflow.log_params(p)
            mlflow.log_param("train_size",  len(X_train))
            mlflow.log_param("val_size",    len(X_val))
            mlflow.log_param("num_classes", len(LABEL_NAMES))

            if model_name == "logistic_regression":
                model = train_logistic_regression(X_train, y_train, X_val, y_val, p)
            elif model_name == "random_forest":
                model = train_random_forest(X_train, y_train, X_val, y_val, p)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            acc, f1 = log_final_metrics(model, X_val, y_val)

            mlflow.sklearn.log_model(
                model,
                name=f"bug_severity_{model_name}",
                registered_model_name=f"bug_severity_{model_name}",
            )

            logger.info(f"{model_name}: acc={acc:.4f} f1={f1:.4f}")

            if f1 > best_f1:
                best_f1     = f1
                best_model  = model
                best_run_id = run.info.run_id

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, "wb") as f:
        pickle.dump({"model": best_model}, f)
    logger.info(f"Best model saved to {model_output_path} (f1={best_f1:.4f})")
    # Save training label distribution for drift detection
    from collections import Counter
    train_label_counts = Counter(y_train.tolist())
    train_dist = {label: train_label_counts.get(label, 0) / len(y_train) for label in LABEL_NAMES}
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/train_distribution.json", "w") as f:
        json.dump(train_dist, f, indent=2)
    logger.info(f"Training distribution saved to metrics/train_distribution.json")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"f1_weighted": round(best_f1, 4), "run_id": best_run_id}, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="Train bug severity classifier")
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--val-features", required=True)
    parser.add_argument("--model-output-path", default="models/best_model.pkl")
    parser.add_argument("--metrics-path", default="metrics/best_model_metrics.json")    
    args = parser.parse_args()
    
    run_training(
        train_path=args.train_features,
        val_path=args.val_features,
        model_output_path=args.model_output_path,
        metrics_path=args.metrics_path,
    )