import requests
import pandas as pd
import logging
import time
import os
from datetime import datetime, UTC

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "scraper.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# -----------------------------
# Config
# -----------------------------
BUGZILLA_BASE = "https://bugzilla.mozilla.org/rest"

SEVERITY_LABELS = ["critical", "major", "normal", "minor", "trivial"]

PRODUCTS = [
    "Firefox",
    "Core",
    "Firefox for Android",
    "DevTools",
    "Toolkit",
]

SEVERITY_REMAP = {
    "critical": "critical",
    "major": "high",
    "normal": "medium",
    "minor": "low",
    "trivial": "low",
}

# -----------------------------
# Fetch bugs
# -----------------------------
def fetch_bugs_page(product, severity, offset, limit=100):
    url = f"{BUGZILLA_BASE}/bug"

    params = {
        "product": product,
        "severity": severity,
        "bug_type": "defect",
        "limit": limit,
        "offset": offset,
        "include_fields": (
            "id,summary,severity,priority,"
            "product,component,status,resolution,"
            "creation_time,comment_count"
        ),
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json().get("bugs", [])

    except requests.exceptions.HTTPError:
        if response.status_code == 429:
            logger.warning("Rate limit hit — sleeping 60s")
            time.sleep(60)
            return fetch_bugs_page(product, severity, offset, limit)

        logger.error(f"HTTP error: {response.text}")
        return []

    except Exception as e:
        logger.error(f"Error fetching bugs: {e}")
        return []


# -----------------------------
# Scrape one product
# -----------------------------
def scrape_product(product, max_per_severity=400):
    all_bugs = []

    for severity in SEVERITY_LABELS:
        logger.info(f"Scraping {product} | severity={severity}")

        offset = 0
        collected = []

        while len(collected) < max_per_severity:
            batch = fetch_bugs_page(product, severity, offset)

            if not batch:
                break

            for bug in batch:
                raw_severity = bug.get("severity", "").lower()
                mapped = SEVERITY_REMAP.get(raw_severity)

                if not mapped:
                    continue

                summary = bug.get("summary", "") or ""

                collected.append({
                    "id": bug.get("id"),
                    "summary": summary,
                    "raw_severity": raw_severity,
                    "severity": mapped,
                    "priority": bug.get("priority", ""),
                    "product": bug.get("product", ""),
                    "component": bug.get("component", ""),
                    "status": bug.get("status", ""),
                    "resolution": bug.get("resolution", ""),
                    "creation_time": bug.get("creation_time", ""),
                    "comment_count": bug.get("comment_count", 0),
                    "summary_len": len(summary),
                    "summary_word_count": len(summary.split()),
                })

            logger.info(f"Offset {offset} → fetched {len(batch)}")

            offset += len(batch)
            time.sleep(0.3)

            if len(batch) < 100:
                break

        logger.info(f"{product}/{severity} → {len(collected[:max_per_severity])} collected")
        all_bugs.extend(collected[:max_per_severity])

    return all_bugs


# -----------------------------
# Main pipeline
# -----------------------------
def run_scraper(output_path="data/raw/bugs.csv", max_per_severity=200):
    logger.info("Starting Bugzilla scraper")

    all_bugs = []

    for product in PRODUCTS:
        logger.info(f"Processing product: {product}")

        bugs = scrape_product(product, max_per_severity)
        all_bugs.extend(bugs)

        logger.info(f"{product}: {len(bugs)} bugs collected")

    if not all_bugs:
        raise ValueError("No bugs scraped")

    df = pd.DataFrame(all_bugs)

    # Remove duplicates
    df.drop_duplicates(subset="id", inplace=True)

    # Add timestamp (fixed version)
    df["scraped_at"] = datetime.now(UTC).isoformat()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # Log stats
    logger.info(f"Saved {len(df)} bugs to {output_path}")
    logger.info(f"Severity distribution:\n{df['severity'].value_counts()}")
    logger.info(f"Raw severity distribution:\n{df['raw_severity'].value_counts()}")

    return output_path


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    run_scraper()