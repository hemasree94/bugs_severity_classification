import os, json, pickle, logging, argparse
import numpy as np
import pandas as pd
import mlflow
import yaml
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

params_file = yaml.safe_load(open("params.yaml"))
LABEL_NAMES = ["trivial", "minor", "normal", "major", "critical"]

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "evaluate.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_model(model_path):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    logger.info(f"Loaded model from {model_path}")
    return bundle["model"]


def load_test_embeddings(test_features_path):
    """
    Load test embeddings from parquet file.
    Expects columns: 'embedding' and 'severity'
    """
    test_df = pd.read_parquet(test_features_path)
    
    X_test = np.array(test_df["embedding"].tolist())
    y_test = test_df["severity"].values
    
    logger.info(f"Test embeddings: {X_test.shape} | labels: {len(y_test)}")
    return X_test, y_test



def run_evaluation(test_features_path, model_path, metrics_path):
    model = load_model(model_path)
    
    X_test, y_test = load_test_embeddings(test_features_path)
    
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1m = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    report = classification_report(
        y_test, y_pred,
        target_names=LABEL_NAMES,
        zero_division=0
    )
    logger.info(f"\n{report}")

    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy":      round(float(acc), 4),
        "f1_weighted":   round(float(f1w), 4),
        "f1_macro":      round(float(f1m), 4),
        "per_class_f1": {
            label: round(float(per_class[i]), 4)
            for i, label in enumerate(LABEL_NAMES)
            if i < len(per_class)
        },
        "confusion_matrix": cm,
        "n_test_samples": int(len(y_test)),
    }

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation metrics saved to {metrics_path}")

    # Log to MLflow under same experiment
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment("bug_severity_classification")
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metric("test_accuracy",    acc)
        mlflow.log_metric("test_f1_weighted", f1w)
        mlflow.log_metric("test_f1_macro",    f1m)
        for i, label in enumerate(LABEL_NAMES):
            if i < len(per_class):
                mlflow.log_metric(f"test_f1_{label}", float(per_class[i]))
        mlflow.log_artifact(metrics_path)
        logger.info("Logged evaluation metrics to MLflow")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate bug severity classifier")
    parser.add_argument("--test-features", required=True, help="Path to test embeddings parquet file")
    parser.add_argument("--model-path",    required=True, help="Path to trained model pickle file")
    parser.add_argument("--metrics-path",  default="metrics/test_metrics.json", help="Path to save metrics")
    args = parser.parse_args()

    run_evaluation(
        test_features_path=args.test_features,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
    )