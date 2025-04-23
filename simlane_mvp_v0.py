# Simlane MVP – v1 with OperationalError & InvalidRequestError fixes
# --------------------------------------------------------
# Key changes vs. previous version
# 1) Engine uses `check_same_thread=False` to avoid SQLite lock issues.
# 2) `load_sample_data()` queries use LIMIT 1 instead of loading full tables.
# 3) Optional sidebar button to hard-reset the database during dev so the
#    schema always matches the SQLModel definitions (handy after column adds).
# 4) Each SQLModel class now declares `__tablename__` + `__table_args__ = {"extend_existing": True}`
#    to avoid `InvalidRequestError` when redeploying against an existing schema.
# --------------------------------------------------------

# Set page configuration – must be first Streamlit command
import streamlit as st
st.set_page_config(
    page_title="Simlane Sales Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Built-ins / stdlib
import os, io, pickle, datetime, uuid, sys
from datetime import timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

# Third-party libs
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sqlmodel import Field, SQLModel, create_engine, Session, select

# --------------------------------------------------------
# Database setup (SQLite)
# --------------------------------------------------------
DATABASE_URL = "sqlite:///simlane.db"
# **IMPORTANT** – allow multi-threaded access which Streamlit uses
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}
)

MODEL_PATH = "simlane_model.pkl"
SHAP_EXPLAINER_PATH = "simlane_explainer.pkl"

# --------------------------------------------------------
# ORM table definitions
# --------------------------------------------------------
class Transaction(SQLModel, table=True):
    __tablename__ = "transaction"
    __table_args__ = {"extend_existing": True}
    transaction_id: str = Field(primary_key=True, nullable=False)
    customer_id: str
    product_id: str
    date: Optional[str]
    quantity: Optional[int]
    revenue: Optional[float]

class PricingLog(SQLModel, table=True):
    __tablename__ = "pricing_log"
    __table_args__ = {"extend_existing": True}
    pricing_id: str = Field(primary_key=True, nullable=False)
    date: Optional[str]
    product_id: str
    list_price: Optional[float]
    discount: Optional[float]
    final_price: Optional[float]

class CompetitorPrice(SQLModel, table=True):
    __tablename__ = "competitor_price"
    __table_args__ = {"extend_existing": True}
    competitor_id: str = Field(primary_key=True, nullable=False)
    competitor_name: str
    product_id: str
    price: Optional[float]
    date: Optional[str]

class Opportunity(SQLModel, table=True):
    __tablename__ = "opportunity"
    __table_args__ = {"extend_existing": True}
    opp_id: str = Field(primary_key=True, nullable=False)
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

# --------------------------------------------------------
# Helpers – DB initialisation & bulk insert
# --------------------------------------------------------

def init_db():
    """Create tables if they do not yet exist."""
    SQLModel.metadata.create_all(engine)


def bulk_insert_dataframe(df: pd.DataFrame, model_cls):
    with Session(engine) as session:
        session.bulk_save_objects([model_cls(**row) for _, row in df.iterrows()])
        session.commit()

# --------------------------------------------------------
# Sample-data loader (unchanged except LIMIT 1 optimisation)
# --------------------------------------------------------

def load_sample_data():
    with Session(engine) as session:
        exists = any([
            session.exec(select(Transaction).limit(1)).first(),
            session.exec(select(PricingLog).limit(1)).first(),
            session.exec(select(CompetitorPrice).limit(1)).first(),
            session.exec(select(Opportunity).limit(1)).first(),
        ])
        if exists:
            return False  # sample already loaded

    # … Original CSV-to-DataFrame loading blocks go here …
    # (Use your previous code for transactions, pricing, competitor, opportunity CSV inserts)
    st.success("Sample data loaded successfully!")
    return True

# --------------------------------------------------------
# *** The rest of your original functions are unchanged ***
# (feature engineering, model functions, Streamlit UI etc.)
# Paste them exactly as before or as needed.
# --------------------------------------------------------

# --------------------------------------------------------
# Main – plus sidebar reset button
# --------------------------------------------------------

def main():
    # Quick dev helper – nuke & recreate DB so schema matches code
    if st.sidebar.button("⚠️ Reset DB (dev only)"):
        if Path("simlane.db").exists():
            Path("simlane.db").unlink()
        init_db()
        st.success("Database reset – reloading app …")
        try:
            st.experimental_rerun()
        except AttributeError:
            st.rerun()

    # Original CSS, titles, and tab UI remain the same
    # Paste the remainder of your main() as-is …

if __name__ == "__main__":
    init_db()
    main()
