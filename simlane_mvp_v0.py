# ------------------------------------------------------------
# Simlane MVP – v1.2  (adds prediction UI + SHAP explain)
# ------------------------------------------------------------
# 1) Page config & imports –– identical to v1.1  ⬆️
# ------------------------------------------------------------
import streamlit as st
st.set_page_config(page_title="Simlane Sales Prediction",
                   page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

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
    shap = None

from sqlmodel import Field, SQLModel, create_engine, Session, select
# ------------------------------------------------------------
# 2) database + ORM models  –– identical to v1.1  ⬆️
# ------------------------------------------------------------
DB_PATH = "simlane.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL, echo=False,
                       connect_args={"check_same_thread": False})

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
    SQLModel.metadata.create_all(engine)

# ------------------------------------------------------------
# 3) helpers / sample‑data / feature engineering
#     (unchanged from v1.1 – collapse for brevity)
# ------------------------------------------------------------
def _safe_dict(series): ...
def bulk_insert_dataframe(df, model): ...
TX_CSV = """..."""
PR_CSV = """..."""
CP_CSV = """..."""
OPP_CSV = """..."""
def load_sample_data(): ...
def customer_metrics(cid): ...
def price_gap(list_price, comp): ...
def sales_cycle(start, end): ...
def enrich_opportunities(df): ...

# ------------------------------------------------------------
# 4) model I/O (unchanged)
# ------------------------------------------------------------
MODEL_FNAME = "simlane_model.pkl"
MODEL = None
def load_model():
    global MODEL
    if Path(MODEL_FNAME).exists():
        MODEL = pickle.load(open(MODEL_FNAME, "rb"))
        return True
    return False
def train_model():
    ...

# ------------------------------------------------------------
# 5) NEW  – prediction helper
# ------------------------------------------------------------
def get_scored_opps() -> pd.DataFrame:
    """
    Returns a dataframe with:
        opp_id, prob_win, predicted, actual, amount, industry, ...
    """
    with Session(engine) as s:
        opps = s.exec(select(Opportunity)).all()

    raw = pd.DataFrame([o.dict() for o in opps])
    feats = enrich_opportunities(raw)
    if feats.empty:
        return pd.DataFrame()

    X = feats.drop(columns=["opp_id", "outcome"])
    proba = MODEL.predict_proba(X)[:, 1]

    feats = feats.copy()
    feats["prob_win"] = proba
    feats.rename(columns={"outcome": "actual"}, inplace=True)
    return feats


# ------------------------------------------------------------
# 6) Streamlit UI
# ------------------------------------------------------------
def main():
    # -- dev reset button (unchanged)
    if st.sidebar.button("🗑️ Reset DB & model"):
        if Path(DB_PATH).exists():
            Path(DB_PATH).unlink()
        if Path(MODEL_FNAME).exists():
            Path(MODEL_FNAME).unlink()
        init_db()
        st.success("Wiped DB & model; reloading …")
        st.experimental_rerun()

    # -- bootstrap
    init_db()
    load_sample_data()
    model_loaded = load_model()

    # -- header
    st.title("Simlane Sales Prediction System")
    st.info(f"Model loaded from disk:  {model_loaded}")

    # -- training if needed
    if not model_loaded:
        if st.button("⚡ Train model now"):
            ok, msg = train_model()
            if ok:
                st.success(msg)
                st.experimental_rerun()
            else:
                st.error(msg)
        return  # stop – no model, nothing else to show

    # ----------------------------------------------------
    #  P R E D I C T I O N   U I
    # ----------------------------------------------------
    st.header("Opportunity win‑probabilities")

    df_pred = get_scored_opps()
    if df_pred.empty:
        st.warning("No opportunity data available.")
        return

    # threshold slider
    thresh = st.slider("Probability threshold for WIN", 0.0, 1.0, 0.50, 0.05)
    df_pred["predicted"] = np.where(df_pred["prob_win"] >= thresh, "WON", "LOST")

    # nice formatting
    st.dataframe(
        df_pred[
            [
                "opp_id",
                "prob_win",
                "predicted",
                "actual",
                "amount",
                "industry",
                "sales_cycle_days",
            ]
        ].style.format({"prob_win": "{:.1%}"})
    )

    # ----------------------------------------------------
    # (Optional) SHAP explanation for one opportunity
    # ----------------------------------------------------
    st.subheader("Explain a single prediction")
    if shap is None:
        st.info("shap package not installed – skip explanation.")
    else:
        sel = st.selectbox("Select opportunity", df_pred["opp_id"])
        if sel:
            row = df_pred[df_pred["opp_id"] == sel]
            X_row = row.drop(columns=["opp_id", "actual", "prob_win", "predicted"])
            # We need the same preprocessing as the pipeline performs:
            shap_explainer = shap.Explainer(MODEL["clf"], feature_names=None)
            shap_values = shap_explainer(MODEL["pre"].transform(X_row))

            st.write(f"Predicted win‑probability: {row['prob_win'].iat[0]:.1%}")
            st.caption("Positive SHAP value → pushes probability up (towards WIN)")
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(bbox_inches="tight")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
