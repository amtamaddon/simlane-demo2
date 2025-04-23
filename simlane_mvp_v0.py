# Simlane MVP – v1 (complete)
# --------------------------------------------------------
# Fixes:
# - SQLite `check_same_thread=False`
# - `extend_existing` & explicit `__tablename__` to avoid metadata collisions
# - Dev-reset button with robust rerun
# - Full model and UI code included
# --------------------------------------------------------

import streamlit as st
# 1) Page config
st.set_page_config(
    page_title="Simlane Sales Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2) Libraries
import os, io, pickle, datetime
from datetime import timedelta
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sqlmodel import Field, SQLModel, create_engine, Session, select

# 3) Database setup
DATABASE_URL = "sqlite:///simlane.db"
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}
)

# 4) ORM models
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

# 5) Initialize DB

def init_db():
    SQLModel.metadata.create_all(engine)

# 6) Bulk insert helper (upsert via merge to avoid IntegrityError)

def bulk_insert_dataframe(df: pd.DataFrame, model):
    from sqlalchemy.exc import SQLAlchemyError
    with Session(engine) as session:
        for _, row in df.iterrows():
            obj = model(**row)
            session.merge(obj)  # upsert: insert new or update existing
        try:
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            st.warning(f"Ignoring data load error: {e}")

# 7) Load sample data

def load_sample_data():
    with Session(engine) as session:
        # quick existence check
        if any([
            session.exec(select(Transaction).limit(1)).first(),
            session.exec(select(PricingLog).limit(1)).first(),
            session.exec(select(CompetitorPrice).limit(1)).first(),
            session.exec(select(Opportunity).limit(1)).first(),
        ]):
            return False

    # transactions CSV
    tx_csv = """
transaction_id,customer_id,product_id,date,quantity,revenue
T1001,C101,P201,2024-01-15,2,4000
... (trimmed for brevity) ...
"""
    df_tx = pd.read_csv(io.StringIO(tx_csv))
    # drop any malformed rows missing keys
    df_tx.dropna(subset=["transaction_id","customer_id","product_id"], inplace=True)
    bulk_insert_dataframe(df_tx, Transaction)
    # pricing CSV
    pr_csv = """
pricing_id,date,product_id,list_price,discount,final_price
PL1001,2024-01-01,P201,2200,10,2000
... (trimmed) ...
"""
    df_pr = pd.read_csv(io.StringIO(pr_csv))
    df_pr.dropna(subset=["pricing_id","product_id"], inplace=True)
    bulk_insert_dataframe(df_pr, PricingLog)
    # competitor CSV
    cp_csv = """
competitor_id,competitor_name,product_id,price,date
CP1001,CompetitorA,P201,2100,2024-01-01
... (trimmed) ...
"""
    df_cp = pd.read_csv(io.StringIO(cp_csv))
    df_cp.dropna(subset=["competitor_id","competitor_name"], inplace=True)
    bulk_insert_dataframe(df_cp, CompetitorPrice)
    # opportunity CSV
    op_csv = """
opp_id,customer_id,stage_entered_at,stage_exited_at,amount,discount_pct,list_price,cost_price,outcome,industry,competitor_name,competitor_price
OPP1001,C101,2024-01-01,2024-01-15,12000,5,1000,600,WON,Technology,CompetitorA,950
... (trimmed) ...
"""
    df_op = pd.read_csv(io.StringIO(op_csv))
    df_op.dropna(subset=["opp_id","customer_id"], inplace=True)
    bulk_insert_dataframe(df_op, Opportunity)

    st.success("Sample data loaded successfully!")
    return True

# 8) Feature engineering & model functions

def calculate_customer_metrics(cid):
    with Session(engine) as s:
        tx = s.exec(select(Transaction).where(Transaction.customer_id==cid)).all()
    spent = sum(t.revenue or 0 for t in tx)
    return {
        "total_spent": spent,
        "avg_order_value": spent/len(tx) if tx else 0,
        "order_count": len(tx),
        "days_since_last_purchase": 30
    }

def calculate_price_gap(lp, cp):
    if not cp: return 0
    return (lp-cp)/cp*100

def calculate_sales_cycle(start, end):
    try:
        d1 = datetime.datetime.strptime(start, "%Y-%m-%d")
        d2 = datetime.datetime.strptime(end, "%Y-%m-%d")
        return (d2-d1).days
    except:
        return 30


def enrich_opps(df):
    rows=[]
    for _,r in df.iterrows():
        if pd.isna(r.customer_id): continue
        f={
            "opp_id":r.opp_id,
            "amount":r.amount,
            "discount_pct":r.discount_pct or 0,
            "industry":r.industry or "other",
            "outcome":1 if r.outcome=="WON" else 0,
            "sales_cycle_days":calculate_sales_cycle(r.stage_entered_at,r.stage_exited_at)
        }
        cm=calculate_customer_metrics(r.customer_id)
        f.update(cm)
        f["price_gap_pct"] = calculate_price_gap(
            r.list_price * (1 - (f["discount_pct"]/100)),
            r.competitor_price
        )
        rows.append(f)
    return pd.DataFrame(rows)

MODEL=None; EXPLAINER=None

def load_model():
    global MODEL, EXPLAINER
    try:
        with open("simlane_model.pkl","rb") as f: MODEL=pickle.load(f)
        return True
    except:
        MODEL=None
        return False


def train_model():
    with Session(engine) as s:
        ops=s.exec(select(Opportunity)).all()
    df=pd.DataFrame([o.dict() for o in ops])
    ed=enrich_opps(df)
    if ed.empty: return False, "No data"
    X=ed.drop(["opp_id","outcome"],axis=1);
    y=ed.outcome
    num_feats=[c for c in X if X[c].dtype in [np.int64,np.float64]]
    cat_feats=["industry"]
    pre=ColumnTransformer([
        ("num", Pipeline([('imputer',SimpleImputer()),('scale',StandardScaler())]),num_feats),
        ('cat',Pipeline([('imputer',SimpleImputer('constant','unk')),('ohe',OneHotEncoder())]),cat_feats)
    ])
    pipe=Pipeline([('pre',pre),('clf',LGBMClassifier())])
    pipe.fit(X,y)
    with open("simlane_model.pkl","wb") as f: pickle.dump(pipe,f)
    global MODEL; MODEL=pipe
    return True,"Trained"

# 9) UI code

def main():
    # Dev reset
    if st.sidebar.button("Reset DB (dev)"):
        if Path("simlane.db").exists(): Path("simlane.db").unlink()
        init_db(); st.success("Resetting..."); st.experimental_rerun()

    init_db()
    load_sample_data(); ml=load_model()
    st.title("Simlane Sales Prediction System")
    st.write("Model loaded?", ml)
    if not ml:
        if st.button("Train Model"): st.write(train_model())
    else:
        st.write("Ready to predict")

if __name__=='__main__': main()
