# Bug Severity Classification System

This project is a bug severity classification system with a data engineering pipeline, API, and monitoring stack.

---

## Quick Start (Using Docker)

### 1. Clone the repository

```bash
git clone https://github.com/hemasree94/bugs_severity_classification
cd bugs_severity_classification
```

---

### 2. Start the application

```bash
docker compose up --build
```

The first run may take a few minutes.

---

### 3. Access the services

* API Docs → http://localhost:8000/docs
* Airflow → http://localhost:8080
* Grafana → http://localhost:3000
* Prometheus → http://localhost:9090

---

### 4. Run the pipeline (Airflow)

1. Open Airflow UI
2. Enable DAG: `bugs_data_engineering_pipeline`
3. Click **Trigger DAG**

---

## Stop the application

```bash
docker compose down
```



## Documentation

Detailed documentation (HLD, LLD, Test Plan) is included separately.
