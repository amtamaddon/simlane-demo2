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
        # ---- Main App UI & Logic ----
    # Custom CSS for styling
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .big-number {
        font-size: 32px;
        font-weight: bold;
        color: #1f77b4;
    }
    .card-title {
        font-size: 14px;
        color: #666;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 0px 16px;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Simlane Sales Prediction System")

    # Initialize and load data/model
    init_db()
    data_loaded = load_sample_data()
    model_loaded = load_model()
    model_status = "✅ Model loaded" if model_loaded else "❌ No trained model found"

    if data_loaded and not model_loaded:
        with st.spinner("Training initial model with sample data..."):
            success, message = train_lightgbm_model()
            if success:
                st.success(message)
                model_loaded = True

    # Setup tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧑‍💼 Account Executive",
        "👨‍💼 Sales Manager",
        "📈 RevOps / CRO",
        "⚙️ Setup & Configuration"
    ])

    # -- Tab 1: Account Executive --
    with tab1:
        st.header("Deal Assessment")
        st.write("Analyze a specific opportunity to determine win probability and optimal pricing.")
        # ... include the entire input form, analysis logic, and charts from your original code here ...

    # -- Tab 2: Sales Manager --
    with tab2:
        st.header("Deal Desk Queue")
        st.write("Focus your coaching on high-value deals with significant margin opportunity.")
        # ... include your prioritization table and scenario analysis UI ...

    # -- Tab 3: RevOps / CRO --
    with tab3:
        st.header("Performance Analytics")
        st.write("Measure the impact of the Simlane system on your sales performance.")
        # ... include performance charts, metrics, lift report, export options ...

    # -- Tab 4: Setup & Configuration --
    with tab4:
        st.header("Setup & Configuration")
        st.write("Manage your data tables, uploads, and model training settings.")
        # ... include your data display tabs and upload/train buttons ...


if __name__ == "__main__":
    init_db()
    main()
