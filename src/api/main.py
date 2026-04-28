import datetime
import time
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# --- DRIFT DETECTION ---
from drift_detection import run_drift_detection

# --- PROMETHEUS ---
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    generate_latest, CONTENT_TYPE_LATEST
)

# --- COUNTERS (total occurrences) ---
PREDICTIONS_TOTAL = Counter(
    "bugs_predictions_total",
    "Total number of bug severity predictions made",
    ["severity", "user_id"]  # custom labels
)
LOGINS_TOTAL = Counter(
    "bugs_logins_total",
    "Total login attempts",
    ["status"]  # 'success' or 'failure'
)
SIGNUPS_TOTAL = Counter(
    "bugs_signups_total",
    "Total signup attempts",
    ["status"]
)
FEEDBACK_TOTAL = Counter(
    "bugs_feedback_total",
    "Total feedback submissions",
    ["correct"]  # 'true' or 'false'
)
HTTP_ERRORS_TOTAL = Counter(
    "bugs_http_errors_total",
    "Total HTTP errors",
    ["endpoint", "status_code"]
)

# --- GAUGES (current state) ---
ACTIVE_REQUESTS = Gauge(
    "bugs_active_requests",
    "Number of currently active requests"
)
TOTAL_USERS_IN_DB = Gauge(
    "bugs_total_users_in_db",
    "Total registered users in database"
)
TOTAL_PREDICTIONS_IN_DB = Gauge(
    "bugs_total_predictions_in_db",
    "Total predictions stored in database"
)

# --- HISTOGRAMS (latency distributions) ---
PREDICTION_LATENCY = Histogram(
    "bugs_prediction_latency_seconds",
    "Time taken for a prediction request",
    ["user_id"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)
FEEDBACK_LATENCY = Histogram(
    "bugs_feedback_latency_seconds",
    "Time taken for a feedback request",
    buckets=[0.005, 0.01, 0.05, 0.1, 0.5]
)
LOGIN_LATENCY = Histogram(
    "bugs_login_latency_seconds",
    "Time taken for login",
    buckets=[0.01, 0.05, 0.1, 0.5]
)

CONFIDENCE_SUMMARY = Summary(
    "bugs_prediction_confidence",
    "Summary of confidence scores for predictions",
    ["severity"]
)

# --- DRIFT DETECTION METRICS ---
DRIFT_DETECTED = Gauge(
    "bugs_drift_detected",
    "Boolean flag indicating if any drift has been detected (0=no drift, 1=drift detected)"
)
ACCURACY_DRIFT = Gauge(
    "bugs_accuracy_drift",
    "Boolean flag for accuracy drift (0=no drift, 1=accuracy below threshold)"
)
DISTRIBUTION_DRIFT = Gauge(
    "bugs_distribution_drift",
    "Boolean flag for distribution drift via KS test (0=no drift, 1=distribution shifted)"
)
FEEDBACK_ACCURACY = Gauge(
    "bugs_feedback_accuracy_percent",
    "Percentage of predictions marked correct by users (%)"
)
KS_STAT = Gauge(
    "bugs_ks_test_statistic",
    "KS test statistic from distribution drift check"
)
KS_PVALUE = Gauge(
    "bugs_ks_test_pvalue",
    "P-value from KS test (lower = more significant drift)"
)
DRIFT_CHECK_TIMESTAMP = Gauge(
    "bugs_drift_check_timestamp_seconds",
    "Unix timestamp of the last drift detection run"
)

# --- DATABASE CONFIG ---
DB_URL = "postgresql+psycopg2://hema:hemasree123@localhost:5432/bugs_db?options=-csearch_path=public"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- AUTH CONFIG ---
SECRET_KEY = "your_super_secret_key"
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# --- DB MODELS ---
class DBUser(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class DBPrediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    summary = Column(String, nullable=False)
    predicted = Column(String, nullable=False)
    confidence = Column(Float)
    probabilities = Column(JSON)
    feedback_correct = Column(Integer, nullable=True)
    ground_truth = Column(String, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- APP SETUP ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- MIDDLEWARE: track active requests globally ---
@app.middleware("http")
async def track_active_requests(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    try:
        response = await call_next(request)
        return response
    finally:
        ACTIVE_REQUESTS.dec()

# --- DEPENDENCIES ---
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = db.query(DBUser).filter(DBUser.username == payload.get("sub")).first()
        if not user: raise HTTPException(status_code=401)
        return user
    except: raise HTTPException(status_code=401)

# --- METRICS ENDPOINT (Prometheus scrapes this) ---
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# --- AUTH ROUTES ---
@app.post("/auth/signup")
def signup(user: dict, db: Session = Depends(get_db)):
    if db.query(DBUser).filter(DBUser.username == user['username']).first():
        SIGNUPS_TOTAL.labels(status="failure").inc()
        HTTP_ERRORS_TOTAL.labels(endpoint="/auth/signup", status_code="400").inc()
        raise HTTPException(status_code=400, detail="Username taken")

    hashed = pwd_context.hash(user['password'])
    db_user = DBUser(username=user['username'], email=user['email'], password=hashed, role=user['role'])
    db.add(db_user)
    db.commit()

    SIGNUPS_TOTAL.labels(status="success").inc()
    TOTAL_USERS_IN_DB.set(db.query(DBUser).count())  # update gauge

    return {"msg": "Success"}

@app.post("/auth/login")
def login(user: dict, db: Session = Depends(get_db)):
    start = time.time()

    db_user = db.query(DBUser).filter(DBUser.username == user['username']).first()
    if not db_user or not pwd_context.verify(user['password'], db_user.password):
        LOGINS_TOTAL.labels(status="failure").inc()
        HTTP_ERRORS_TOTAL.labels(endpoint="/auth/login", status_code="400").inc()
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = jwt.encode({"sub": db_user.username}, SECRET_KEY, algorithm=ALGORITHM)

    LOGINS_TOTAL.labels(status="success").inc()
    LOGIN_LATENCY.observe(time.time() - start)

    return {"token": token, "username": db_user.username, "role": db_user.role}

# --- PREDICTION ROUTES ---
@app.post("/predict")
def predict(req: dict, db: Session = Depends(get_db), user: DBUser = Depends(get_current_user)):
    start = time.time()

    summary_lower = req['summary'].lower()
    severity = "normal"
    if "crash" in summary_lower or "critical" in summary_lower: severity = "critical"
    elif "ui" in summary_lower or "typo" in summary_lower: severity = "trivial"

    probs = {"trivial": 0.1, "minor": 0.1, "normal": 0.2, "major": 0.2, "critical": 0.4}
    confidence = 0.88

    new_pred = DBPrediction(
        summary=req['summary'], predicted=severity, confidence=confidence,
        probabilities=probs, user_id=user.id
    )
    db.add(new_pred)
    db.commit()
    db.refresh(new_pred)

    # --- Record metrics ---
    PREDICTIONS_TOTAL.labels(severity=severity, user_id=str(user.id)).inc()
    PREDICTION_LATENCY.labels(user_id=str(user.id)).observe(time.time() - start)
    CONFIDENCE_SUMMARY.labels(severity=severity).observe(confidence)
    TOTAL_PREDICTIONS_IN_DB.set(db.query(DBPrediction).count())  # update gauge

    return {
        "prediction_id": new_pred.id, "severity": severity, "confidence": confidence,
        "probabilities": probs, "drift_warning": False
    }

@app.post("/feedback")
def feedback(req: dict, db: Session = Depends(get_db)):
    start = time.time()

    pred = db.query(DBPrediction).filter(DBPrediction.id == req['prediction_id']).first()
    if pred:
        pred.feedback_correct = 1 if req['correct'] else 0
        pred.ground_truth = req.get('ground_truth')
        db.commit()

    correct_label = "true" if req['correct'] else "false"
    FEEDBACK_TOTAL.labels(correct=correct_label).inc()
    FEEDBACK_LATENCY.observe(time.time() - start)

    return {"status": "ok"}

@app.get("/history")
def history(db: Session = Depends(get_db), user: DBUser = Depends(get_current_user)):
    results = db.query(DBPrediction, DBUser.username).join(DBUser).order_by(DBPrediction.id.desc()).all()
    return [{
        "id": p.id, "summary": p.summary, "predicted": p.predicted,
        "confidence": p.confidence, "feedback_correct": p.feedback_correct,
        "ground_truth": p.ground_truth, "username": uname, "created_at": p.created_at
    } for p, uname in results]

# --- STATS ROUTES ---
@app.get("/feedback/stats")
def feedback_stats(db: Session = Depends(get_db)):
    fbs = db.query(DBPrediction).filter(DBPrediction.feedback_correct != None).all()
    total = len(fbs)
    correct = sum(1 for f in fbs if f.feedback_correct == 1)
    acc = (correct / total * 100) if total > 0 else 0
    return {"total": total, "correct": correct, "accuracy_pct": round(acc, 1)}

@app.get("/drift-check")
def drift_check():
    """
    Run drift detection and update Prometheus metrics.
    Returns detailed drift information.
    """
    try:
        result = run_drift_detection()
        
        # Update Prometheus gauges
        DRIFT_DETECTED.set(1 if result.drift_detected else 0)
        ACCURACY_DRIFT.set(1 if result.details.get("accuracy_drift") else 0)
        DISTRIBUTION_DRIFT.set(1 if result.details.get("ks_drift") else 0)
        
        accuracy = result.details.get("feedback_accuracy")
        if accuracy is not None:
            FEEDBACK_ACCURACY.set(accuracy * 100)
        
        ks_stat = result.details.get("ks_stat")
        if ks_stat is not None:
            KS_STAT.set(ks_stat)
        
        ks_pvalue = result.details.get("ks_pvalue")
        if ks_pvalue is not None:
            KS_PVALUE.set(ks_pvalue)
        
        DRIFT_CHECK_TIMESTAMP.set(time.time())
        
        return {
            "drift_detected": result.drift_detected,
            "reasons": result.reasons,
            "details": result.details
        }
    except Exception as e:
        return {
            "error": str(e),
            "drift_detected": False,
            "reasons": ["drift_check_failed"],
            "details": {}
        }

@app.get("/admin/stats")
def admin_stats(db: Session = Depends(get_db)):
    feedback_rows = db.query(DBPrediction).filter(DBPrediction.feedback_correct != None).all()
    total_feedback = len(feedback_rows)
    correct_feedback = sum(1 for f in feedback_rows if f.feedback_correct == 1)
    accuracy = (correct_feedback / total_feedback * 100) if total_feedback > 0 else 0
    
    return {
        "total_predictions": db.query(DBPrediction).count(),
        "total_users": db.query(DBUser).count(),
        "total_feedback": total_feedback,
        "feedback_accuracy": round(accuracy, 1),
        "drift_alerts": int(DRIFT_DETECTED._value.get()),
        "model_ready": True,
        "mlflow_url": "http://localhost:5000"
    }

@app.get("/admin/users")
def admin_users(db: Session = Depends(get_db)):
    return db.query(DBUser).all()

# --- FRONTEND ROUTES ---
@app.get("/")
@app.get("/login.html")
def read_login():
    return FileResponse("src/frontend/login.html")

@app.get("/index.html")
def read_index():
    return FileResponse("src/frontend/index.html")

@app.get("/history.html")
def read_history():
    return FileResponse("src/frontend/history.html")

@app.get("/admin.html")
def read_admin():
    return FileResponse("src/frontend/admin.html")