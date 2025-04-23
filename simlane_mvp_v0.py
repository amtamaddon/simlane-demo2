# ------------------------------------------------------------
# Simlane MVP – v1.3  (robust enrich_opportunities + guard)
# ------------------------------------------------------------
#  ▶ Run:  streamlit run simlane_app.py
# ------------------------------------------------------------

# ---------- 1.  Page config ----------
import streamlit as st
st.set_page_config(page_title="Simlane Sales Prediction",
                   page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

# ---------- 2.  Core libraries ----------
import os, io, pickle, datetime
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    import shap
except Exception:
    shap = None         # SHAP is optional

from sqlmodel import Field, SQLModel, create_engine, Session, select

# ---------- 3.  Database ----------
DB_PATH      = "simlane.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL, echo=False,
                       connect_args={"check_same_thread": False})

# ---------- 4.  ORM models ----------
class Transaction(SQLModel, table=True):
    transaction_id: str = Field(primary_key=True)
    customer_id: str
    product_id: str
    date: Optional[str]
    quantity: Optional[int]
    revenue: Optional[float]

class PricingLog(SQLModel, table=True):
    pricing_id: str = Field(primary_key=True)
    date: Optional[str]
    product_id: str
    list_price: Optional[float]
    discount: Optional[float]
    final_price: Optional[float]

class CompetitorPrice(SQLModel, table=True):
    competitor_id: str = Field(primary_key=True)
    competitor_name: str
    product_id: str
    price: Optional[float]
    date: Optional[str]

class Opportunity(SQLModel, table=True):
    opp_id: str = Field(primary_key=True)
    customer_id: str
    stage_entered_at: Optional[str]
    stage_exited_at: Optional[str]
    amount: Optional[float]
    discount_pct: Optional[float] = 0.0
    list_price: Optional[float]
    cost_price: Optional[float]
    outcome: Optional[str]
    industry: Optional[str]
    competitor_name: Optional[str]
    competitor_price: Optional[float]

def init_db() -> None:
    SQLModel.metadata.create_all(engine)

# ---------- 5.  Helpers ----------
def _safe(series: pd.Series) -> dict:
    """replace NaN with None before ORM insert"""
    return {k: (None if pd.isna(v) else v) for k, v in series.items()}

def bulk_insert(df: pd.DataFrame, model):
    with Session(engine) as s:
        for _, row in df.iterrows():
            s.merge(model(**_safe(row)))
        s.commit()

# ---------- 6.  Demo CSV data (tiny) ----------
TX_CSV = """
transaction_id,customer_id,product_id,date,quantity,revenue
T1001,C101,P201,2024-01-15,2,4000
T1002,C102,P202,2024-02-02,1,1500
T1003,C101,P203,2024-03-01,3,4500
"""
PR_CSV = """
pricing_id,date,product_id,list_price,discount,final_price
PL1001,2024-01-01,P201,2200,10,1980
PL1002,2024-02-01,P202,1600,5,1520
PL1003,2024-03-01,P203,1700,15,1445
"""
CP_CSV = """
competitor_id,competitor_name,product_id,price,date
CP1001,CompetitorA,P201,2100,2024-01-01
CP1002,CompetitorB,P202,1550,2024-02-01
CP1003,CompetitorC,P203,1500,2024-03-01
"""
OPP_CSV = """
opp_id,customer_id,stage_entered_at,stage_exited_at,amount,discount_pct,list_price,cost_price,outcome,industry,competitor_name,competitor_price
OPP1001,C101,2024-01-01,2024-01-15,12000,5,1000,600,WON,Technology,CompetitorA,950
OPP1002,C102,2024-02-15,2024-03-03,8000,10,1600,900,LOST,Healthcare,CompetitorB,1550
OPP1003,C101,2024-03-10,2024-03-25,10000,0,1700,1000,WON,Technology,CompetitorC,1500
"""

def load_sample_data():
    with Session(engine) as s:
        if s.exec(select(Opportunity).limit(1)).first():
            return           # already loaded
    bulk_insert(pd.read_csv(io.StringIO(TX_CSV.strip())),  Transaction)
    bulk_insert(pd.read_csv(io.StringIO(PR_CSV.strip())),  PricingLog)
    bulk_insert(pd.read_csv(io.StringIO(CP_CSV.strip())),  CompetitorPrice)
    bulk_insert(pd.read_csv(io.StringIO(OPP_CSV.strip())), Opportunity)
    st.success("Sample data inserted.")

# ---------- 7.  Feature engineering ----------
def customer_metrics(cid):
    with Session(engine) as s:
        tx = s.exec(select(Transaction).where(Transaction.customer_id == cid)).all()
    total = sum(t.revenue or 0 for t in tx)
    count = len(tx)
    last  = max([t.date for t in tx if t.date] or ["1970-01-01"])
    delta = (datetime.datetime.utcnow() -
             datetime.datetime.strptime(last, "%Y-%m-%d")).days
    return {
        "total_spent": total,
        "avg_order_value": total / count if count else 0,
        "order_count": count,
        "days_since_last_purchase": delta,
    }

def price_gap(lst_price, comp_price):
    if not comp_price or comp_price == 0:
        return 0.0
    return (lst_price - comp_price) / comp_price * 100

def sales_cycle(start, end):
    try:
        return (datetime.datetime.strptime(end, "%Y-%m-%d") -
                datetime.datetime.strptime(start, "%Y-%m-%d")).days
    except Exception:
        return 30

def enrich_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for _, r in df.iterrows():
        if pd.isna(r.customer_id):
            continue
        feat = {
            "opp_id": r.opp_id,
            "amount": r.amount,
            "discount_pct": r.discount_pct or 0.0,
            "industry": r.industry or "other",
            "outcome": 1 if str(r.outcome).upper() == "WON" else 0,
            "sales_cycle_days": sales_cycle(r.stage_entered_at, r.stage_exited_at),
        }
        feat.update(customer_metrics(r.customer_id))
        final_price = r.list_price * (1 - feat["discount_pct"] / 100)
        feat["price_gap_pct"] = price_gap(final_price, r.competitor_price)
        rows.append(feat)

    # --- IMPORTANT: always return a DataFrame (even if empty) ---
    return pd.DataFrame(rows)

# ---------- 8.  Model I/O ----------
MODEL_F = "simlane_model.pkl"
MODEL   = None

def load_model() -> bool:
    global MODEL
    if Path(MODEL_F).exists():
        MODEL = pickle.load(open(MODEL_F, "rb"))
        return True
    return False

def train_model():
    with Session(engine) as s:
        rows = s.exec(select(Opportunity)).all()
    df_raw = pd.DataFrame([o.dict() for o in rows])
    df     = enrich_opportunities(df_raw)
    if df.empty:
        return False, "No data."
    X = df.drop(columns=["opp_id", "outcome"])
    y = df["outcome"]

    num = [c for c in X if X[c].dtype in [np.int64, np.float64]]
    cat = ["industry"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="unk")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LGBMClassifier())])
    pipe.fit(X, y)
    pickle.dump(pipe, open(MODEL_F, "wb"))
    global MODEL
    MODEL = pipe
    return True, "Model trained."

# ---------- 9.  Prediction helper ----------
def get_scored_opps() -> pd.DataFrame:
    with Session(engine) as s:
        opps = s.exec(select(Opportunity)).all()
    raw   = pd.DataFrame([o.dict() for o in opps])
    feats = enrich_opportunities(raw)

    # defensive guard: wrong type OR empty
    if not isinstance(feats, pd.DataFrame) or feats.empty:
        return pd.DataFrame()

    X      = feats.drop(columns=["opp_id", "outcome"])
    feats  = feats.copy()
    feats["prob_win"] = MODEL.predict_proba(X)[:, 1]
    feats.rename(columns={"outcome": "actual"}, inplace=True)
    return feats

# ---------- 10.  Streamlit UI ----------
def main():
    # dev‑reset
    if st.sidebar.button("🗑️ Reset DB & model"):
        for p in (DB_PATH, MODEL_F):
            if Path(p).exists():
                Path(p).unlink()
        init_db()
        st.experimental_rerun()

    # bootstrap
    init_db()
    load_sample_data()
    mdl_loaded = load_model()

    st.title("Simlane Sales Prediction System")
    st.info(f"Model loaded from disk:  {mdl_loaded}")

    if not mdl_loaded:
        if st.button("⚡ Train model now"):
            ok, msg = train_model()
            (st.success if ok else st.error)(msg)
            if ok:
                st.experimental_rerun()
        return

    # ---- Prediction UI ----
    st.header("Opportunity win‑probabilities")
    df_pred = get_scored_opps()
    if df_pred.empty:
        st.warning("No opportunity data available.")
        return

    thresh = st.slider("Probability threshold for WIN", 0.0, 1.0, 0.50, 0.05)
    df_pred["predicted"] = np.where(df_pred["prob_win"] >= thresh, "WON", "LOST")

    st.dataframe(df_pred[["opp_id", "prob_win", "predicted", "actual",
                          "amount", "industry", "sales_cycle_days"]]
                 .style.format({"prob_win": "{:.1%}"}))

    # ---- optional SHAP ----
    st.subheader("Explain a single prediction")
    if shap is None:
        st.info("Install `shap` to enable explanations.")
    else:
        sel = st.selectbox("Select opportunity", df_pred["opp_id"])
        if sel:
            row = df_pred[df_pred["opp_id"] == sel]
            Xr  = row.drop(columns=["opp_id", "actual", "prob_win", "predicted"])
            explainer = shap.Explainer(MODEL["clf"])
            sv = explainer(MODEL["pre"].transform(Xr))
            st.write(f"Predicted win‑probability: {row['prob_win'].iat[0]:.1%}")
            shap.plots.waterfall(sv[0], show=False)
            st.pyplot(bbox_inches="tight")

# ---------- 11.  run ----------
if __name__ == "__main__":
    main()
