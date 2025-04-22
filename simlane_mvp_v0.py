import sys
import os
import streamlit as st
from sqlmodel import Field, SQLModel, create_engine, Session, select
from typing import Optional
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
# Internal helper to bulk‑insert DataFrame rows into a SQLModel table
# -----------------------------------------------------------------------------

def bulk_insert_dataframe(df: pd.DataFrame, model_cls):
    with Session(engine) as session:
        objs = [model_cls(**row) for _, row in df.iterrows()]
        session.bulk_save_objects(objs)
        session.commit()

# -----------------------------------------------------------------------------
# Model management
# -----------------------------------------------------------------------------

MODEL_PATH = "propensity.pkl"
model = None

def load_model():
    global model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return True
    except FileNotFoundError:
        return False

def train_model():
    global model
    with Session(engine) as session:
        ops = session.exec(select(Opportunity)).all()
    
    if not ops:
        return False, "No opportunity data to train on."
    
    df = pd.DataFrame([o.dict() for o in ops])
    df = df.dropna(subset=["outcome", "amount"])
    
    if len(df) == 0:
        return False, "No valid opportunity data with both outcome and amount."
    
    X = df[["amount"]]
    y = df["outcome"].apply(lambda x: 1 if x.upper() == "WON" else 0)
    
    clf = LogisticRegression()
    clf.fit(X, y)
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    
    model = clf
    return True, f"Model trained successfully with {len(df)} samples!"

def predict(amount):
    global model
    if model is None:
        load_model()
        if model is None:
            return None
    
    prob = float(model.predict_proba([[amount]])[0][1])
    return prob

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def main():
    st.title("Simlane MVP Dashboard")
    
    # Initialize database
    init_db()
    
    # Load model if exists
    model_loaded = load_model()
    model_status = "✅ Model loaded" if model_loaded else "❌ No trained model found"
    
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
            success, message = train_model()
            if success:
                st.success(message)
            else:
                st.error(message)
    
    # Tab 3: Make Predictions
    with tab3:
        st.header("Predict Win Probability")
        
        amount = st.number_input("Enter opportunity amount:", min_value=0.0, value=10000.0)
        
        if st.button("Predict"):
            prob = predict(amount)
            if prob is None:
                st.error("Model not trained yet. Please train the model first.")
            else:
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

if __name__ == "__main__":
    main()
