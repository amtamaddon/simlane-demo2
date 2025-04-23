# ------------------------------------------------------------
# Simlane MVP – v1.1  (complete working example)
# ------------------------------------------------------------
#  – Streamlit       1.29+
#  – SQLModel        0.0.8
#  – LightGBM        4.3.0
#  – pandas / numpy  latest
# ------------------------------------------------------------

# 1) Streamlit page config
import streamlit as st
st.set_page_config(
    page_title="Simlane Sales Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2) Standard libraries
import os, io, pickle, datetime
from pathlib import Path
from typing import Optional, List

# 3) Data & ML stack
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 4) Plotting (optional – remove if not needed)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 5) SHAP can be heavy; import lazily / defensively
try:
    import shap
except Exception:
    shap = None

# 6) SQLModel / SQLAlchemy
from sqlmodel import Field, SQLModel, create_engine, Session, select

# ------------------------------------------------------------
# Database setup
# ------------------------------------------------------------
DB_PATH = "simlane.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},  # streamlit = multi‑thread
)

# ------------------------------------------------------------
# ORM models
# ------------------------------------------------------------
class Transaction(SQLModel, table=True):
    __tablename__ = "transaction"
    __table_args__ = {"extend_existing": True}

    transaction_id: str = Field(primary_key=True)
    customer_id: str
    product_id: str
    date: Optional[str]
    quantity: Optional[int]
    revenue: Optional[float]


class PricingLog(SQLModel, table=True):
    __tablename__ = "pricing_log"
    __table_args__ = {"extend_existing": True}

    pricing_id: str = Field(primary_key=True)
    date: Optional[str]
    product_id: str
    list_price: Optional[float]
    discount: Optional[float]
    final_price: Optional[float]


class CompetitorPrice(SQLModel, table=True):
    __tablename__ = "competitor_price"
    __table_args__ = {"extend_existing": True}

    competitor_id: str = Field(primary_key=True)
    competitor_name: str
    product_id: str
    price: Optional[float]
    date: Optional[str]


class Opportunity(SQLModel, table=True):
    __tablename__ = "opportunity"
    __table_args__ = {"extend_existing": True}

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
    """Create tables if they don’t exist"""
    SQLModel.metadata.create_all(engine)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _safe_dict(series: pd.Series) -> dict:
    """Replace pandas NaN/NaT with real Python None before ORM insert"""
    return {k: (None if pd.isna(v) else v) for k, v in series.items()}


def bulk_insert_dataframe(df: pd.DataFrame, model):
    """
    Upsert dataframe rows into the table mapped by `model`
    using SQLModel.merge() to avoid PK collisions.
    """
    from sqlalchemy.exc import SQLAlchemyError

    with Session(engine) as session:
        for _, row in df.iterrows():
            obj = model(**_safe_dict(row))
            session.merge(obj)  # ← upsert
        try:
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            st.warning(f"Ignoring data‑load error: {e}")


# ------------------------------------------------------------
# Demo data
# ------------------------------------------------------------
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


def load_sample_data() -> None:
    """Insert demo data only if the tables are empty"""
    with Session(engine) as s:
        tables_filled = any(
            [
                s.exec(select(Transaction).limit(1)).first(),
                s.exec(select(PricingLog).limit(1)).first(),
                s.exec(select(CompetitorPrice).limit(1)).first(),
                s.exec(select(Opportunity).limit(1)).first(),
            ]
        )
    if tables_filled:
        return

    bulk_insert_dataframe(pd.read_csv(io.StringIO(TX_CSV.strip())), Transaction)
    bulk_insert_dataframe(pd.read_csv(io.StringIO(PR_CSV.strip())), PricingLog)
    bulk_insert_dataframe(pd.read_csv(io.StringIO(CP_CSV.strip())), CompetitorPrice)
    bulk_insert_dataframe(pd.read_csv(io.StringIO(OPP_CSV.strip())), Opportunity)

    st.success("Sample data loaded into the fresh database.")


# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------
def customer_metrics(cid: str) -> dict:
    """Basic RFM‑like features"""
    with Session(engine) as s:
        tx = s.exec(select(Transaction).where(Transaction.customer_id == cid)).all()

    total = sum(t.revenue or 0 for t in tx)
    order_cnt = len(tx)
    last_dt = (
        max([t.date for t in tx if t.date] or ["1970-01-01"])
        if order_cnt
        else "1970-01-01"
    )
    delta_days = (datetime.datetime.utcnow() - datetime.datetime.strptime(last_dt, "%Y-%m-%d")).days

    return {
        "total_spent": total,
        "avg_order_value": total / order_cnt if order_cnt else 0,
        "order_count": order_cnt,
        "days_since_last_purchase": delta_days,
    }


def price_gap(list_price: float, comp_price: Optional[float]) -> float:
    if not comp_price or comp_price == 0:
        return 0.0
    return (list_price - comp_price) / comp_price * 100


def sales_cycle(start: Optional[str], end: Optional[str]) -> int:
    try:
        d1 = datetime.datetime.strptime(str(start), "%Y-%m-%d")
        d2 = datetime.datetime.strptime(str(end), "%Y-%m-%d")
        return (d2 - d1).days
    except Exception:
        return 30  # default


def enrich_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    """Return ML‑ready dataframe"""
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

        final_price = r.list_price * (1 - feat["discount_pct"] / 100.0)
        feat["price_gap_pct"] = price_gap(final_price, r.competitor_price)

        rows.append(feat)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Model I/O
# ------------------------------------------------------------
MODEL_FNAME = "simlane_model.pkl"
MODEL = None  # populated by load_model()


def load_model() -> bool:
    global MODEL
    if Path(MODEL_FNAME).exists():
        MODEL = pickle.load(open(MODEL_FNAME, "rb"))
        return True
    return False


def train_model() -> tuple[bool, str]:
    with Session(engine) as s:
        opps = s.exec(select(Opportunity)).all()

    df_raw = pd.DataFrame([o.dict() for o in opps])
    df = enrich_opportunities(df_raw)

    if df.empty:
        return False, "No opportunity data available."

    X = df.drop(columns=["opp_id", "outcome"])
    y = df["outcome"]

    num_cols = [c for c in X if X[c].dtype in [np.int64, np.float64]]
    cat_cols = ["industry"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="unk")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", LGBMClassifier())])
    pipe.fit(X, y)

    pickle.dump(pipe, open(MODEL_FNAME, "wb"))
    global MODEL
    MODEL = pipe
    return True, "Model trained and saved."


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
def main():
    # Developer utility: wipe DB + model
    if st.sidebar.button("🗑️  Reset DB & Model (dev)"):
        if Path(DB_PATH).exists():
            Path(DB_PATH).unlink()
        if Path(MODEL_FNAME).exists():
            Path(MODEL_FNAME).unlink()

        init_db()
        st.success("Database and model files deleted. Reloading app …")

        # Compatible rerun across Streamlit versions
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()

    # -- bootstrap
    init_db()
    load_sample_data()
    model_loaded = load_model()

    # -- main area
    st.title("Simlane Sales Prediction System")
    st.markdown(
        """
A very small demo showing: SQLModel → LightGBM → Streamlit  
(The data is fake & minimal; replace it with your own.)
"""
    )

    st.info(f"Model loaded from disk:  **{model_loaded}**")

    if not model_loaded:
        if st.button("⚡ Train Model"):
            ok, msg = train_model()
            if ok:
                st.success(msg)
                st.experimental_rerun()
            else:
                st.error(msg)
    else:
        st.success("Model is ready – plug your own prediction UI here.")


if __name__ == "__main__":
    main()
