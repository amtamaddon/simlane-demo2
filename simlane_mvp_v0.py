import sys
import os
import subprocess
import streamlit as st

# Check if running in Streamlit environment
is_streamlit = 'STREAMLIT_SHARING_MODE' in os.environ or sys.argv[0].endswith('streamlit')

# -----------------------------------------------------------------------------
# Skip auto-installation when running in Streamlit
# -----------------------------------------------------------------------------
REQUIRED_PKGS = [
    "fastapi",           # web framework
    "uvicorn[standard]", # ASGI server runtime
    "sqlmodel",          # ORM / SQL
    "pandas",            # data wrangling
    "scikit-learn",      # ML baseline
    "python-multipart"   # for file uploads
]

# Only attempt to install packages if NOT running in Streamlit
if not is_streamlit:
    for pkg in REQUIRED_PKGS:
        try:
            __import__(pkg.split("[")[0])
        except ModuleNotFoundError:
            print(f"[Simlane] Installing missing dependency: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# -----------------------------------------------------------------------------
# Now that dependencies are assumed to be available, import them
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
model = None  # loaded at startup if present

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
# Streamlit UI
# -----------------------------------------------------------------------------

def streamlit_app():
    st.title("Simlane MVP Dashboard")
    
    # Initialize database
    init_db()
    
    # Load model if exists
    global model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        model_status = "✅ Model loaded"
    except FileNotFoundError:
        model_status = "❌ No trained model found"
    
    st.sidebar.header("Model Status")
    st.sidebar.write(model_status)
    
    # Create tabs for different operations
    tab1, tab2, tab3, tab4 = st.tabs(["Upload Data", "Train Model", "Make Predictions", "View Database"])
    
    # Tab 1: Upload Data
    with tab1:
        st.header("Upload CSV Data")
        table_name = st.selectbox(
            "Select table to upload to:",
            ["transactions", "pricing_logs", "competitors", "opportunities"]
        )
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"Preview of {uploaded_file.name}:")
            st.dataframe(df.head())
            
            if st.button("Upload to Database"):
                table_map = {
                    "transactions": Transaction,
                    "pricing_logs": PricingLog,
                    "competitors": CompetitorPrice,
                    "opportunities": Opportunity,
                }
                bulk_insert_dataframe(df, table_map[table_name])
                st.success(f"Successfully uploaded {len(df)} rows to {table_name} table!")
    
    # Tab 2: Train Model
    with tab2:
        st.header("Train Propensity Model")
        st.write("This will train a model to predict the probability of winning an opportunity based on its amount.")
        
        if st.button("Train Model"):
            with Session(engine) as session:
                ops = session.exec(select(Opportunity)).all()
            
            if not ops:
                st.error("No opportunity data to train on. Please upload opportunity data first.")
            else:
                df = pd.DataFrame([o.dict() for o in ops])
                df = df.dropna(subset=["outcome", "amount"])
                
                if len(df) == 0:
                    st.error("No valid opportunity data with both outcome and amount. Please check your data.")
                else:
                    X = df[["amount"]]
                    y = df["outcome"].apply(lambda x: 1 if x.upper() == "WON" else 0)
                    
                    clf = LogisticRegression()
                    clf.fit(X, y)
                    
                    with open(MODEL_PATH, "wb") as f:
                        pickle.dump(clf, f)
                    
                    model = clf
                    
                    st.success(f"Model trained successfully with {len(df)} samples!")
    
    # Tab 3: Make Predictions
    with tab3:
        st.header("Predict Win Probability")
        
        amount = st.number_input("Enter opportunity amount:", min_value=0.0, value=10000.0)
        
        if st.button("Predict"):
            if model is None:
                st.error("Model not trained yet. Please train the model first.")
            else:
                prob = float(model.predict_proba([[amount]])[0][1])
                st.success(f"Propensity to buy: {prob:.2%}")
                
                # Visualization
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.barh(["Probability"], [prob])
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability of Winning")
                st.pyplot(fig)
    
    # Tab 4: View Database
    with tab4:
        st.header("View Database Tables")
        table_to_view = st.selectbox(
            "Select table to view:",
            ["transactions", "pricing_logs", "competitors", "opportunities"]
        )
        
        if st.button("View Data"):
            with Session(engine) as session:
                if table_to_view == "transactions":
                    data = session.exec(select(Transaction)).all()
                elif table_to_view == "pricing_logs":
                    data = session.exec(select(PricingLog)).all()
                elif table_to_view == "competitors":
                    data = session.exec(select(CompetitorPrice)).all()
                else:  # opportunities
                    data = session.exec(select(Opportunity)).all()
                
                if not data:
                    st.info(f"No data in {table_to_view} table.")
                else:
                    df = pd.DataFrame([item.dict() for item in data])
                    st.dataframe(df)
                    st.write(f"Total rows: {len(df)}")

# -----------------------------------------------------------------------------
# Entrypoint for local execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if is_streamlit:
        streamlit_app()
    else:
        uvicorn.run("simlane_mvp_v0:app", host="0.0.0.0", port=8000, reload=True)
