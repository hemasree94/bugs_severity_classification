"""
drift_detection.py — Drift Detection Module
Location: src/drift_detection.py

Two checks:
  1. Feedback Accuracy Drop  → model performance drift
  2. KS Test                 → data/distribution drift (production vs training)

Returns a DriftResult with:
  - drift_detected (bool)
  - reasons (list of strings)
  - details (dict with all numbers)
"""

import json
import logging
import os

import numpy as np
import psycopg2
from scipy import stats

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DB_CONFIG = {
    "dbname":   "bugs_db",
    "user":     "hema",
    "password": "hemasree123",
    "host":     "localhost",
    "port":     5432,
    "options":  "-c search_path=public"
}

LABEL_NAMES           = ["trivial", "minor", "normal", "major", "critical"]
LABEL_TO_INT          = {l: i for i, l in enumerate(LABEL_NAMES)}
TRAIN_DIST_PATH       = "metrics/train_distribution.json"

ACCURACY_THRESHOLD    = 0.90   # below 90% feedback accuracy = drift
KS_PVALUE_THRESHOLD   = 0.05   # below 0.05 p-value = significant distribution shift
MIN_FEEDBACK_COUNT    = 10     # minimum feedbacks needed to check accuracy drift
MIN_PRODUCTION_COUNT  = 20     # minimum production predictions needed for KS test

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# RESULT OBJECT
# -------------------------------------------------------------------
class DriftResult:
    def __init__(self, drift_detected: bool, reasons: list, details: dict):
        self.drift_detected = drift_detected
        self.reasons        = reasons
        self.details        = details

    def __str__(self):
        status = "DRIFT DETECTED" if self.drift_detected else "NO DRIFT"
        return (
            f"\n{'='*50}\n"
            f"Drift Status : {status}\n"
            f"Reasons      : {', '.join(self.reasons) if self.reasons else 'None'}\n"
            f"Details      : {json.dumps(self.details, indent=2)}\n"
            f"{'='*50}"
        )


# -------------------------------------------------------------------
# HELPER: Fetch data from DB
# -------------------------------------------------------------------
def _fetch_from_db():
    """Fetch recent predictions and feedback from PostgreSQL."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    # Fetch last 200 predictions (production data)
    cur.execute("""
        SELECT predicted, feedback_correct
        FROM predictions
        ORDER BY created_at DESC
        LIMIT 200
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    predicted_labels = [row[0] for row in rows]
    feedback_rows    = [(row[0], row[1]) for row in rows if row[1] is not None]

    return predicted_labels, feedback_rows


# -------------------------------------------------------------------
# CHECK 1: Feedback Accuracy Drop
# -------------------------------------------------------------------
def check_accuracy_drift(feedback_rows: list) -> tuple:
    """
    Check if model feedback accuracy has dropped below threshold.
    Returns (drift: bool, accuracy: float, reason: str)
    """
    total = len(feedback_rows)

    if total < MIN_FEEDBACK_COUNT:
        logger.info(f"Not enough feedback for accuracy check ({total}/{MIN_FEEDBACK_COUNT}). Skipping.")
        return False, None, f"insufficient_feedback ({total} rows)"

    correct  = sum(1 for _, fb in feedback_rows if fb == 1)
    accuracy = correct / total

    logger.info(f"Feedback accuracy: {accuracy:.2%} ({correct}/{total})")

    if accuracy < ACCURACY_THRESHOLD:
        reason = f"accuracy dropped to {accuracy:.2%} (threshold={ACCURACY_THRESHOLD:.0%})"
        return True, accuracy, reason

    return False, accuracy, None


# -------------------------------------------------------------------
# CHECK 2: KS Test — Production vs Training Distribution
# -------------------------------------------------------------------
def check_distribution_drift(predicted_labels: list) -> tuple:
    """
    Compare production prediction distribution vs training distribution using KS test.
    Returns (drift: bool, ks_stat: float, p_value: float, reason: str)
    """
    total = len(predicted_labels)

    if total < MIN_PRODUCTION_COUNT:
        logger.info(f"Not enough production data for KS test ({total}/{MIN_PRODUCTION_COUNT}). Skipping.")
        return False, None, None, f"insufficient_production_data ({total} rows)"

    if not os.path.exists(TRAIN_DIST_PATH):
        logger.warning(f"Training distribution file not found at {TRAIN_DIST_PATH}. Skipping KS test.")
        return False, None, None, "train_distribution_missing"

    # Load training distribution
    with open(TRAIN_DIST_PATH) as f:
        train_dist = json.load(f)

    # Convert production labels to integer array
    prod_ints   = [LABEL_TO_INT.get(l, 2) for l in predicted_labels]

    # Build training distribution as integer array (weighted sample)
    train_ints  = []
    for label, proportion in train_dist.items():
        count = int(proportion * total)
        train_ints.extend([LABEL_TO_INT[label]] * count)

    # Pad or trim to same length
    train_ints  = train_ints[:total]
    if len(train_ints) < total:
        train_ints += [2] * (total - len(train_ints))  # pad with "normal"

    ks_stat, p_value = stats.ks_2samp(prod_ints, train_ints)
    logger.info(f"KS test — stat={ks_stat:.4f}, p-value={p_value:.4f}")

    # Log distributions for visibility
    prod_dist = {l: round(prod_ints.count(LABEL_TO_INT[l]) / total, 3) for l in LABEL_NAMES}
    logger.info(f"Production distribution : {prod_dist}")
    logger.info(f"Training distribution   : {train_dist}")

    if p_value < KS_PVALUE_THRESHOLD:
        reason = f"KS test p-value={p_value:.4f} < threshold={KS_PVALUE_THRESHOLD} (stat={ks_stat:.4f})"
        return True, ks_stat, p_value, reason

    return False, ks_stat, p_value, None


# -------------------------------------------------------------------
# MAIN: Run Both Checks
# -------------------------------------------------------------------
def run_drift_detection() -> DriftResult:
    """
    Run both drift checks and return a DriftResult.
    Drift is flagged if EITHER check fails.
    """
    logger.info("Starting drift detection...")

    predicted_labels, feedback_rows = _fetch_from_db()
    logger.info(f"Fetched {len(predicted_labels)} predictions, {len(feedback_rows)} with feedback")

    reasons = []
    details = {}

    # --- Check 1: Accuracy
    acc_drift, accuracy, acc_reason = check_accuracy_drift(feedback_rows)
    details["feedback_total"]    = len(feedback_rows)
    details["feedback_accuracy"] = round(accuracy, 4) if accuracy is not None else None
    details["accuracy_drift"]    = acc_drift
    if acc_drift:
        reasons.append(acc_reason)

    # --- Check 2: KS Test
    ks_drift, ks_stat, p_value, ks_reason = check_distribution_drift(predicted_labels)
    details["production_total"]  = len(predicted_labels)
    details["ks_stat"]           = round(ks_stat, 4) if ks_stat is not None else None
    details["ks_pvalue"]         = round(p_value, 4) if p_value is not None else None
    details["ks_drift"]          = ks_drift
    if ks_drift:
        reasons.append(ks_reason)

    drift_detected = acc_drift or ks_drift

    result = DriftResult(drift_detected=drift_detected, reasons=reasons, details=details)
    logger.info(str(result))

    return result


# -------------------------------------------------------------------
# Run standalone for testing
# -------------------------------------------------------------------
if __name__ == "__main__":
    result = run_drift_detection()
    print(result)