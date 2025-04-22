# Simlane MVP v0 (self‑installing)
# ---------------------------------------------------------------
# Minimal FastAPI service that ingests static CSV uploads, trains a very simple
# propensity‑to‑buy model, and exposes an API for predictions.
# The script will attempt to auto‑install missing Python packages so it runs
# out‑of‑the‑box with just `python simlane_mvp_v0.py`.
# ---------------------------------------------------------------

import sys
import subprocess

# -----------------------------------------------------------------------------
# Best‑effort auto‑installer for required third‑party packages
# -----------------------------------------------------------------------------
REQUIRED_PKGS = [
    "fastapi",           # web framework
    "uvicorn[standard]", # ASGI server runtime
    "sqlmodel",          # ORM / SQL
    "pandas",            # data wrangling
    "scikit-learn",      # ML baseline
    "python-multipart"   # for file uploads
]

# Install any missing packages at runtime (helpful for bare environments)
for pkg in REQUIRED_PKGS:
    try:
        __import__(pkg.split("[")[0])
    except ModuleNotFoundError:
        print(f"[Simlane] Installing missing dependency: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# -----------------------------------------------------------------------------
# Now that dependencies are guaranteed, import them
# -----------------------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, HTTPException
from sqlmodel import Field, SQLModel, create_engine, Session, select
from typing import Optional
import pandas as pd
from sklearn.linear_model import LogisticRegression
import uvicorn
import io
import pickle

# -----------------------------------------------------------------------------
# Database setup (SQLite for simplicity)
# -----------------------------------------------------------------------------
DATABASE_URL = "sqlite:///simlane.db"
engine = create_engine(DATABASE_URL, echo=False)

# -----------------------------------------------------------------------------
# ORM table definitions matching the MVP raw data feeds
# -----------------------------------------------------------------------------

class Transaction(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    transaction_id: str = Field(primary_key=True)
    customer_id: str
    product_id: str
    date: Optional[str]
    quantity: Optional[int]
    revenue: Optional[float]

class PricingLog(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    pricing_id: str = Field(primary_key=True)
    date: Optional[str]
    product_id: str
    list_price: Optional[float]
    discount: Optional[float]
    final_price: Optional[float]

class CompetitorPrice(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    competitor_id: str = Field(primary_key=True)
    competitor_name: str
    product_id: str
    price: Optional[float]
    date: Optional[str]

class Opportunity(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    opp_id: str = Field(primary_key=True)
    customer_id: str
    stage_entered_at: Optional[str]
    stage_exited_at: Optional[str]
    amount: Optional[float]
    outcome: Optional[str]  # expected values e.g., "WON", "LOST"
    industry: Optional[str]

# -----------------------------------------------------------------------------
# Helper to initialize the database schema
# -----------------------------------------------------------------------------

def init_db():
    SQLModel.metadata.create_all(engine)

# -----------------------------------------------------------------------------
# FastAPI app instance
# -----------------------------------------------------------------------------
app = FastAPI(title="Simlane MVP v0")

MODEL_PATH = "propensity.pkl"
model: Optional[LogisticRegression] = None  # loaded at startup if present

# -----------------------------------------------------------------------------
# Event hooks
# -----------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    """Create tables and load an existing model if it was previously trained."""
    init_db()
    global model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        model = None

# -----------------------------------------------------------------------------
# Internal helper to bulk‑insert DataFrame rows into a SQLModel table
# -----------------------------------------------------------------------------

def bulk_insert_dataframe(df: pd.DataFrame, model_cls):
    with Session(engine) as session:
        objs = [model_cls(**row) for _, row in df.iterrows()]
        session.bulk_save_objects(objs)
        session.commit()

# -----------------------------------------------------------------------------
# REST endpoints
# -----------------------------------------------------------------------------

@app.post("/upload/{table_name}")
async def upload_csv(table_name: str, file: UploadFile = File(...)):
    """Upload a CSV file to populate any of the raw tables.

    Allowed table_name values: transactions | pricing_logs | competitors | opportunities
    """
    table_map = {
        "transactions": Transaction,
        "pricing_logs": PricingLog,
        "competitors": CompetitorPrice,
        "opportunities": Opportunity,
    }
    if table_name not in table_map:
        raise HTTPException(status_code=400, detail="Unknown table name")

    # Read file into pandas DataFrame (auto‑detect encoding)
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    bulk_insert_dataframe(df, table_map[table_name])
    return {"status": "ok", "rows_ingested": len(df)}

@app.post("/train/propensity")
def train_propensity():
    """Train a naive propensity‑to‑buy model (LogisticRegression on amount)."""
    with Session(engine) as session:
        ops = session.exec(select(Opportunity)).all()

    if not ops:
        raise HTTPException(status_code=400, detail="No opportunity data to train on.")

    df = pd.DataFrame([o.dict() for o in ops])
    df = df.dropna(subset=["outcome", "amount"])
    X = df[["amount"]]
    y = df["outcome"].apply(lambda x: 1 if x.upper() == "WON" else 0)

    clf = LogisticRegression()
    clf.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    global model
    model = clf

    return {"status": "trained", "samples": len(df)}

@app.get("/predict/propensity")
def predict_propensity(amount: float):
    """Predict win probability for a single deal amount."""
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet.")

    prob = float(model.predict_proba([[amount]])[0][1])
    return {"propensity_to_buy": prob}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# -----------------------------------------------------------------------------
# Entrypoint for local execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("simlane_mvp_v0:app", host="0.0.0.0", port=8000, reload=True)
