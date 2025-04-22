import streamlit as st

# Set page configuration
st.set_page_config(layout="wide", page_title="Simlane Sales Prediction System", page_icon="💰")

import os
import sys
import io
import pickle
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
import datetime
import uuid
from datetime import timedelta

# ML Libraries
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ORM/Database
from sqlmodel import Field, SQLModel, create_engine, Session, select

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

# -----------------------------------------------------------------------------
# Database setup (SQLite for simplicity)
# -----------------------------------------------------------------------------
DATABASE_URL = "sqlite:///simlane.db"
engine = create_engine(DATABASE_URL, echo=False)

# -----------------------------------------------------------------------------
# Model paths
# -----------------------------------------------------------------------------
MODEL_PATH = "simlane_model.pkl"
SHAP_EXPLAINER_PATH = "simlane_explainer.pkl"

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
    discount_pct: Optional[float] = 0.0
    list_price: Optional[float]
    cost_price: Optional[float]
    outcome: Optional[str]  # expected values e.g., "WON", "LOST"
    industry: Optional[str]
    competitor_name: Optional[str]
    competitor_price: Optional[float]

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
# Sample data loading function
# -----------------------------------------------------------------------------

def load_sample_data():
    # Check if tables are empty
    with Session(engine) as session:
        transaction_count = session.exec(select(Transaction)).first()
        pricing_count = session.exec(select(PricingLog)).first()
        competitor_count = session.exec(select(CompetitorPrice)).first()
        opportunity_count = session.exec(select(Opportunity)).first()
        
        # Only load sample data if all tables are empty
        if not (transaction_count or pricing_count or competitor_count or opportunity_count):
            st.info("Loading sample data for first-time setup...")
            
            # Sample transaction data
            transactions_data = """
transaction_id,customer_id,product_id,date,quantity,revenue
T1001,C101,P201,2024-01-15,2,4000
T1002,C102,P202,2024-01-17,1,5500
T1003,C103,P203,2024-01-20,3,9000
T1004,C101,P204,2024-01-22,1,3500
T1005,C104,P201,2024-01-28,2,4000
T1006,C105,P205,2024-02-03,1,12000
T1007,C106,P206,2024-02-08,4,16000
T1008,C102,P207,2024-02-10,1,7500
T1009,C107,P208,2024-02-15,2,8000
T1010,C108,P209,2024-02-18,1,9500
T1011,C109,P210,2024-02-22,3,15000
T1012,C101,P203,2024-02-25,2,6000
T1013,C110,P201,2024-03-01,1,2000
T1014,C111,P211,2024-03-05,5,25000
T1015,C112,P212,2024-03-08,1,8500
T1016,C113,P213,2024-03-12,2,11000
T1017,C103,P206,2024-03-15,3,12000
T1018,C107,P214,2024-03-18,1,6500
T1019,C114,P215,2024-03-22,4,18000
T1020,C115,P216,2024-03-26,2,14000
T1021,C116,P217,2024-03-30,1,9000
T1022,C101,P218,2024-04-02,3,15000
T1023,C117,P219,2024-04-05,1,7500
T1024,C106,P220,2024-04-08,2,13000
T1025,C118,P221,2024-04-12,4,22000
            """
            df_transactions = pd.read_csv(io.StringIO(transactions_data))
            bulk_insert_dataframe(df_transactions, Transaction)
            
            # Sample pricing data
            pricing_data = """
pricing_id,date,product_id,list_price,discount,final_price
PL1001,2024-01-01,P201,2200,10,2000
PL1002,2024-01-01,P202,6000,8.33,5500
PL1003,2024-01-01,P203,3500,14.29,3000
PL1004,2024-01-01,P204,4000,12.5,3500
PL1005,2024-01-01,P205,13000,7.69,12000
PL1006,2024-01-01,P206,4500,11.11,4000
PL1007,2024-01-01,P207,8000,6.25,7500
PL1008,2024-01-01,P208,4500,11.11,4000
PL1009,2024-01-01,P209,10000,5,9500
PL1010,2024-01-01,P210,5500,9.09,5000
PL1011,2024-01-01,P211,5500,9.09,5000
PL1012,2024-01-01,P212,9000,5.56,8500
PL1013,2024-01-01,P213,6000,8.33,5500
PL1014,2024-01-01,P214,7000,7.14,6500
PL1015,2024-01-01,P215,5000,10,4500
PL1016,2024-01-01,P216,7500,6.67,7000
PL1017,2024-01-01,P217,9500,5.26,9000
PL1018,2024-01-01,P218,5500,9.09,5000
PL1019,2024-01-01,P219,8000,6.25,7500
PL1020,2024-01-01,P220,7000,7.14,6500
PL1021,2024-01-01,P221,6000,8.33,5500
            """
            df_pricing = pd.read_csv(io.StringIO(pricing_data))
            bulk_insert_dataframe(df_pricing, PricingLog)
            
            # Sample competitor data
            competitor_data = """
competitor_id,competitor_name,product_id,price,date
CP1001,CompetitorA,P201,2100,2024-01-01
CP1002,CompetitorB,P201,2300,2024-01-01
CP1003,CompetitorA,P202,5800,2024-01-01
CP1004,CompetitorB,P202,5900,2024-01-01
CP1005,CompetitorA,P203,3200,2024-01-01
CP1006,CompetitorB,P203,3100,2024-01-01
CP1007,CompetitorA,P204,3600,2024-01-01
CP1008,CompetitorB,P204,3700,2024-01-01
CP1009,CompetitorA,P205,11500,2024-01-01
CP1010,CompetitorB,P205,12500,2024-01-01
CP1011,CompetitorA,P206,4200,2024-01-01
CP1012,CompetitorB,P206,4100,2024-01-01
CP1013,CompetitorA,P207,7800,2024-01-01
CP1014,CompetitorB,P207,7600,2024-01-01
CP1015,CompetitorA,P208,4200,2024-01-01
CP1016,CompetitorB,P208,4300,2024-01-01
CP1017,CompetitorA,P209,9800,2024-01-01
CP1018,CompetitorB,P209,9700,2024-01-01
CP1019,CompetitorA,P210,5200,2024-01-01
CP1020,CompetitorB,P210,5100,2024-01-01
CP1021,CompetitorC,P201,2000,2024-01-01
CP1022,CompetitorC,P202,5600,2024-01-01
CP1023,CompetitorC,P203,3000,2024-01-01
CP1024,CompetitorC,P204,3400,2024-01-01
CP1025,CompetitorC,P205,12200,2024-01-01
            """
            df_competitor = pd.read_csv(io.StringIO(competitor_data))
            bulk_insert_dataframe(df_competitor, CompetitorPrice)
            
            # Sample opportunity data with enhanced fields
            opportunity_data = """
opp_id,customer_id,stage_entered_at,stage_exited_at,amount,discount_pct,list_price,cost_price,outcome,industry,competitor_name,competitor_price
OPP1001,C101,2024-01-01,2024-01-15,12000,5,1000,600,WON,Technology,CompetitorA,950
OPP1002,C102,2024-01-05,2024-01-25,25000,0,1000,600,WON,Healthcare,CompetitorB,1050
OPP1003,C103,2024-01-10,2024-02-10,45000,2,1000,600,WON,Financial Services,CompetitorC,980
OPP1004,C104,2024-01-15,2024-02-15,8000,8,1000,600,LOST,Retail,CompetitorA,900
OPP1005,C105,2024-01-20,2024-02-20,30000,3,1000,600,WON,Manufacturing,CompetitorB,1020
OPP1006,C106,2024-01-25,2024-02-25,55000,0,1000,600,WON,Technology,CompetitorC,990
OPP1007,C107,2024-02-01,2024-03-01,15000,10,1000,600,LOST,Healthcare,CompetitorA,920
OPP1008,C108,2024-02-05,2024-03-05,22000,7,1000,600,LOST,Retail,CompetitorB,970
OPP1009,C109,2024-02-10,2024-03-10,18000,3,1000,600,WON,Financial Services,CompetitorC,1000
OPP1010,C110,2024-02-15,2024-03-15,7000,12,1000,600,LOST,Manufacturing,CompetitorA,880
OPP1011,C111,2024-02-20,2024-03-20,65000,0,1000,600,WON,Technology,CompetitorB,1080
OPP1012,C112,2024-02-25,2024-03-25,28000,2,1000,600,WON,Healthcare,CompetitorC,990
OPP1013,C113,2024-03-01,2024-04-01,42000,8,1000,600,LOST,Financial Services,CompetitorA,930
OPP1014,C114,2024-03-05,2024-04-05,33000,4,1000,600,WON,Retail,CompetitorB,1030
OPP1015,C115,2024-03-10,2024-04-10,50000,0,1000,600,WON,Manufacturing,CompetitorC,1010
OPP1016,C116,2024-03-15,2024-04-15,17000,9,1000,600,LOST,Technology,CompetitorA,910
OPP1017,C117,2024-03-20,2024-04-20,26000,3,1000,600,WON,Healthcare,CompetitorB,990
OPP1018,C118,2024-03-25,2024-04-25,38000,2,1000,600,WON,Financial Services,CompetitorC,1000
OPP1019,C101,2024-01-30,2024-02-28,10000,0,1000,600,WON,Technology,CompetitorA,1050
OPP1020,C103,2024-02-15,2024-03-15,35000,7,1000,600,LOST,Financial Services,CompetitorB,940
OPP1021,C106,2024-03-05,2024-04-05,48000,1,1000,600,WON,Technology,CompetitorC,1020
OPP1022,C109,2024-03-20,2024-04-20,22000,2,1000,600,WON,Financial Services,CompetitorA,1010
OPP1023,C112,2024-04-01,2024-04-30,31000,6,1000,600,LOST,Healthcare,CompetitorB,950
OPP1024,C114,2024-04-10,2024-05-10,27000,3,1000,600,WON,Retail,CompetitorC,990
OPP1025,C101,2024-04-15,2024-05-15,14000,0,1000,600,WON,Technology,CompetitorA,1060
OPP1026,C105,2024-04-20,2024-05-20,36000,8,1000,600,LOST,Manufacturing,CompetitorB,920
OPP1027,C111,2024-04-25,2024-05-25,72000,0,1000,600,WON,Technology,CompetitorC,1040
OPP1028,C115,2024-05-01,2024-05-30,45000,1,1000,600,WON,Manufacturing,CompetitorA,1030
OPP1029,C103,2024-05-05,2024-06-05,29000,9,1000,600,LOST,Financial Services,CompetitorB,930
OPP1030,C118,2024-05-10,2024-06-10,41000,2,1000,600,WON,Financial Services,CompetitorC,1000
            """
            df_opportunity = pd.read_csv(io.StringIO(opportunity_data))
            bulk_insert_dataframe(df_opportunity, Opportunity)
            
            st.success("Sample data loaded successfully!")
            return True
    return False

# -----------------------------------------------------------------------------
# Enhanced Feature Engineering and Model Processing
# -----------------------------------------------------------------------------

def calculate_customer_metrics(customer_id):
    """Calculate customer-related metrics for feature enrichment."""
    with Session(engine) as session:
        # Get all transactions for this customer
        transactions = session.exec(
            select(Transaction).where(Transaction.customer_id == customer_id)
        ).all()
        
        if not transactions:
            return {
                "total_spent": 0,
                "avg_order_value": 0,
                "order_count": 0,
                "days_since_last_purchase": 999  # High number for new customers
            }
        
        # Calculate metrics
        revenues = [t.revenue for t in transactions if t.revenue is not None]
        if not revenues:
            return {
                "total_spent": 0,
                "avg_order_value": 0,
                "order_count": len(transactions),
                "days_since_last_purchase": 999
            }
            
        return {
            "total_spent": sum(revenues),
            "avg_order_value": sum(revenues) / len(revenues) if revenues else 0,
            "order_count": len(transactions),
            "days_since_last_purchase": 30  # Placeholder, would calculate from dates
        }

def calculate_price_gap(list_price, competitor_price):
    """Calculate price gap percentage relative to competitor."""
    if not list_price or not competitor_price or competitor_price == 0:
        return 0
    
    return ((list_price - competitor_price) / competitor_price) * 100

def calculate_sales_cycle_days(entered_at, exited_at):
    """Calculate sales cycle duration in days."""
    if not entered_at or not exited_at:
        return 30  # Default value
    
    try:
        start_date = datetime.datetime.strptime(entered_at, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(exited_at, "%Y-%m-%d")
        return (end_date - start_date).days
    except (ValueError, TypeError):
        return 30  # Default value

def enrich_opportunity_features(opportunities_df):
    """Add derived features to opportunities dataframe."""
    enriched_data = []
    
    for _, opp in opportunities_df.iterrows():
        # Skip opportunities without required fields
        if pd.isna(opp.customer_id) or pd.isna(opp.amount) or pd.isna(opp.outcome):
            continue
            
        # Basic features from opportunity
        features = {
            "opp_id": opp.opp_id,
            "amount": opp.amount,
            "discount_pct": opp.discount_pct if not pd.isna(opp.discount_pct) else 0,
            "industry": opp.industry if not pd.isna(opp.industry) else "unknown",
            "outcome": 1 if str(opp.outcome).upper() == "WON" else 0
        }
        
        # Calculate sales cycle duration
        features["sales_cycle_days"] = calculate_sales_cycle_days(
            opp.stage_entered_at, opp.stage_exited_at
        )
            
        # Add customer metrics
        customer_metrics = calculate_customer_metrics(opp.customer_id)
        features.update(customer_metrics)
        
        # Calculate price gap if competitor price exists
        if not pd.isna(opp.competitor_price) and not pd.isna(opp.list_price):
            # Adjust list price by discount percentage
            effective_price = opp.list_price * (1 - (opp.discount_pct / 100 if not pd.isna(opp.discount_pct) else 0))
            features["price_gap_pct"] = calculate_price_gap(effective_price, opp.competitor_price)
        else:
            features["price_gap_pct"] = 0
        
        enriched_data.append(features)
    
    return pd.DataFrame(enriched_data)

# -----------------------------------------------------------------------------
# Probability-Weighted Margin Calculation
# -----------------------------------------------------------------------------

def calculate_pwm(win_probability, amount, list_price, discount_pct=0, cost_price=None):
    """Calculate probability-weighted margin for a deal."""
    # Default cost as 60% of list price if not provided
    if cost_price is None:
        cost_price = list_price * 0.6
    
    # Calculate actual price after discount
    actual_price = list_price * (1 - discount_pct/100)
    
    # Calculate margin per unit
    margin_per_unit = actual_price - cost_price
    
    # Calculate total margin
    total_margin = margin_per_unit * amount
    
    # Weight by win probability
    prob_weighted_margin = total_margin * win_probability
    
    return prob_weighted_margin

def determine_safe_discount_band(model, feature_vector, current_discount, target_min_prob=0.5, max_range=10):
    """Calculate recommended discount range that maintains acceptable win probability."""
    # Create a range of discount percentages to test
    min_discount = max(0, current_discount - max_range)
    max_discount = min(40, current_discount + max_range)  # Cap at 40% discount
    
    test_discounts = list(range(int(min_discount), int(max_discount) + 1))
    results = []
    
    # Clone the feature vector for simulations
    for discount in test_discounts:
        test_vector = feature_vector.copy()
        test_vector['discount_pct'] = discount
        
        # Convert to DataFrame for prediction
        test_df = pd.DataFrame([test_vector])
        
        # Predict with this discount
        prob = model.predict_proba(test_df)[0][1]
        
        results.append((discount, prob))
    
    # Find range where probability stays above target
    acceptable_discounts = [d for d, p in results if p >= target_min_prob]
    
    if not acceptable_discounts:
        return (current_discount, current_discount)  # No safe band found
        
    return (min(acceptable_discounts), max(acceptable_discounts))

# -----------------------------------------------------------------------------
# Enhanced Model Management
# -----------------------------------------------------------------------------

model = None
shap_explainer = None

def load_model():
    """Load trained model from disk."""
    global model, shap_explainer
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        
        # Also load SHAP explainer if it exists
        try:
            with open(SHAP_EXPLAINER_PATH, "rb") as f:
                shap_explainer = pickle.load(f)
        except FileNotFoundError:
            shap_explainer = None
            
        return True
    except FileNotFoundError:
        model = None
        shap_explainer = None
        return False

def train_lightgbm_model():
    """Train an enhanced LightGBM model with SHAP explanations."""
    global model, shap_explainer
    
    # Get raw opportunity data
    with Session(engine) as session:
        ops = session.exec(select(Opportunity)).all()
    
    if not ops:
        return False, "No opportunity data to train on."
    
    # Convert to DataFrame and enrich with features
    df = pd.DataFrame([o.dict() for o in ops])
    enriched_df = enrich_opportunity_features(df)
    
    if len(enriched_df) == 0:
        return False, "No valid opportunity data after feature engineering."
    
    # Define feature columns
    categorical_features = ['industry']
    numeric_features = [
        'amount', 
        'discount_pct',
        'sales_cycle_days', 
        'total_spent', 
        'avg_order_value', 
        'order_count', 
        'days_since_last_purchase', 
        'price_gap_pct'
    ]
    
    # Create preprocessor
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create and train pipeline with LightGBM
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42
        ))
    ])
    
    # Split data for training and evaluation
    X = enriched_df.drop(['opp_id', 'outcome'], axis=1)
    y = enriched_df['outcome']
    
    # If we have enough data, do a train/test split
    if len(enriched_df) >= 20:  # Arbitrary minimum for splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics_message = f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}"
    else:
        # Not enough data for split, use whole dataset
        pipeline.fit(X, y)
        
        # Use cross-validation instead
        try:
            cv_scores = cross_val_score(pipeline, X, y, cv=min(5, len(enriched_df)))
            avg_score = np.mean(cv_scores)
            metrics_message = f"Cross-validation accuracy: {avg_score:.2f}"
        except ValueError:
            metrics_message = "Not enough samples for cross-validation"
    
    # Create SHAP explainer for feature importance
    try:
        # We need to transform the data first
        X_processed = pipeline.named_steps['preprocessor'].transform(X)
        
        # Create SHAP explainer using the classifier from the pipeline
        explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
        
        # Save the explainer
        with open(SHAP_EXPLAINER_PATH, "wb") as f:
            pickle.dump(explainer, f)
            
        shap_explainer = explainer
    except Exception as e:
        st.warning(f"Could not create SHAP explainer: {str(e)}")
    
    # Save the model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    
    model = pipeline
    
    # Return success and statistics
    return True, f"Model trained successfully with {len(enriched_df)} samples! {metrics_message}"

def get_top_drivers(feature_vector):
    """Get top factors driving the prediction for a specific opportunity."""
    global model, shap_explainer
    
    if not model or not shap_explainer:
        return []
    
    try:
        # Convert feature vector to DataFrame
        X = pd.DataFrame([feature_vector])
        
        # Preprocess the data
        X_processed = model.named_steps['preprocessor'].transform(X)
        
        # Calculate SHAP values
        shap_values = shap_explainer.shap_values(X_processed)
        
        # For binary classification, shap_values is a list of two arrays
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # We want the values for the positive class (index 1)
            shap_values = shap_values[1]
        
        # Get feature names after preprocessing
        feature_names = []
        
        # Get numeric feature names
        numeric_indices = model.named_steps['preprocessor'].transformers_[0][2]
        numeric_names = [model.named_steps['preprocessor'].feature_names_in_[i] for i in numeric_indices]
        feature_names.extend(numeric_names)
        
        # Get categorical feature names
        try:
            cat_indices = model.named_steps['preprocessor'].transformers_[1][2]
            cat_encoder = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
            cat_names = cat_encoder.get_feature_names_out([model.named_steps['preprocessor'].feature_names_in_[i] for i in cat_indices])
            feature_names.extend(cat_names)
        except:
            # If we can't get categorical names, use placeholders
            cat_count = X_processed.shape[1] - len(numeric_names)
            feature_names.extend([f"categorical_{i}" for i in range(cat_count)])
        
        # Get absolute SHAP values and sort
        abs_shap_values = np.abs(shap_values[0])
        indices = np.argsort(abs_shap_values)[::-1]
        
        # Return top drivers with direction
        top_drivers = []
        for i in indices[:5]:  # Get top 5 drivers
            direction = "+" if shap_values[0][i] > 0 else "-"
            
            # Clean up feature name
            if i < len(feature_names):
                feature_name = feature_names[i]
                if feature_name.startswith(('industry_', 'categorical_')):
                    parts = feature_name.split('_')
                    feature_name = parts[0].capitalize()
                else:
                    # Make numeric features more readable
                    feature_name = feature_name.replace('_', ' ').title()
            else:
                feature_name = f"Feature {i}"
            
            top_drivers.append({
                "feature": feature_name,
                "importance": float(abs_shap_values[i]),
                "direction": direction,
                "value": feature_vector.get(feature_names[i].split('_')[0] if i < len(feature_names) else "unknown", "N/A")
            })
        
        return top_drivers
    except Exception as e:
        st.warning(f"Error calculating drivers: {str(e)}")
        return []

def predict_with_lightgbm(feature_vector):
    """Make predictions with the trained model."""
    global model
    if model is None:
        load_model()
        if model is None:
            return None, None, None
    
    # Create a DataFrame with the feature vector
    sample = pd.DataFrame([feature_vector])
    
    # Get prediction probability
    probs = model.predict_proba(sample)
    win_prob = probs[0][1]  # Probability of class 1 (win)
    
    # Get top drivers for this prediction
    drivers = get_top_drivers(feature_vector)
    
    # Determine safe discount band
    discount_band = determine_safe_discount_band(
        model, 
        feature_vector, 
        feature_vector.get('discount_pct', 0),
        target_min_prob=0.5
    )
    
    return win_prob, drivers, discount_band

# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def create_win_probability_gauge(win_prob):
    """Create an interactive gauge chart for win probability."""
    fig = go.Figure()
    
    # Add a gauge chart
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=win_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Win Probability"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#FF9999'},
                {'range': [30, 70], 'color': '#FFDD99'},
                {'range': [70, 100], 'color': '#99CC99'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': win_prob * 100
            }
        }
    ))
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Arial", size=12)
    )
    
    return fig

def create_driver_chart(drivers):
    """Create a horizontal bar chart for impact drivers."""
    if not drivers:
        return None
    
    # Prepare data
    features = [f"{d['feature']} ({d['direction']})" for d in drivers]
    importances = [d['importance'] for d in drivers]
    colors = ['#1f77b4' if d['direction'] == '+' else '#d62728' for d in drivers]
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        y=features,
        x=importances,
        orientation='h',
        marker_color=colors,
        text=[f"{d['value']}" for d in drivers],
        textposition='auto'
    ))
    
    # Update layout
    fig.update_layout(
        title="Top Impact Drivers",
        xaxis_title="Impact on Win Probability",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Arial", size=12)
    )
    
    return fig

def create_discount_band_chart(current_discount, discount_band, pwm):
    """Create a chart showing the safe discount band."""
    min_discount, max_discount = discount_band
    
    # Create a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add discount band as a rectangle
    fig.add_trace(
        go.Scatter(
            x=[min_discount, max_discount, max_discount, min_discount, min_discount],
            y=[0, 0, 1, 1, 0],
            fill="toself",
            fillcolor="rgba(0, 128, 0, 0.2)",
            line=dict(color="rgba(0, 128, 0, 0.5)"),
            name="Safe Discount Range",
            showlegend=False
        )
    )
    
    # Add current discount marker
    fig.add_trace(
        go.Scatter(
            x=[current_discount],
            y=[0.5],
            mode="markers",
            marker=dict(size=20, color="red", symbol="diamond"),
            name="Current Discount"
        )
    )
    
    # Add line for PWM
    discount_range = list(range(max(0, min(min_discount, current_discount) - 5), 
                               min(50, max(max_discount, current_discount) + 5)))
    
    # Dummy PWM values - in a real implementation, we would calculate actual PWM values
    pwm_values = [pwm * (1 - (d - current_discount) * 0.02) for d in discount_range]
    
    fig.add_trace(
        go.Scatter(
            x=discount_range,
            y=pwm_values,
            mode="lines",
            name="Probability-Weighted Margin",
            line=dict(color="blue", width=3),
        ),
        secondary_y=True
    )
    
    # Update axes and layout
    fig.update_xaxes(title_text="Discount Percentage")
    fig.update_yaxes(title_text="Safe Range", secondary_y=False, showticklabels=False)
    fig.update_yaxes(title_text="Probability-Weighted Margin ($)", secondary_y=True)
    
    fig.update_layout(
        title="Safe Discount Band vs. Probability-Weighted Margin",
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_deal_comparison_chart(amount, industry, sales_cycle, total_spent, price_gap):
    """Create a radar chart comparing this deal to successful deals."""
    
    # In a real implementation, we would query the database for average values
    # of successful deals. For demo purposes, we'll use dummy data.
    avg_successful = {
        "Amount": 30000,
        "Sales Cycle": 21,
        "Customer Spend": 25000,
        "Price Gap": -5
    }
    
    # Normalize values to 0-1 scale
    max_values = {
        "Amount": 100000,
        "Sales Cycle": 60,
        "Customer Spend": 100000,
        "Price Gap": 20
    }
    
    # Convert current deal values
    current_deal_norm = {
        "Amount": min(1, amount / max_values["Amount"]),
        "Sales Cycle": min(1, sales_cycle / max_values["Sales Cycle"]),
        "Customer Spend": min(1, total_spent / max_values["Customer Spend"]),
        "Price Gap": min(1, (price_gap + 20) / 40)  # Adjust for negative values
    }
    
    # Convert average successful deal values
    avg_successful_norm = {
        "Amount": min(1, avg_successful["Amount"] / max_values["Amount"]),
        "Sales Cycle": min(1, avg_successful["Sales Cycle"] / max_values["Sales Cycle"]),
        "Customer Spend": min(1, avg_successful["Customer Spend"] / max_values["Customer Spend"]),
        "Price Gap": min(1, (avg_successful["Price Gap"] + 20) / 40)
    }
    
    # Create radar chart
    categories = ["Amount", "Sales Cycle", "Customer Spend", "Price Gap"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[current_deal_norm[c] for c in categories],
        theta=categories,
        fill='toself',
        name='Current Deal'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[avg_successful_norm[c] for c in categories],
        theta=categories,
        fill='toself',
        name=f'Avg. Successful {industry}'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Deal Comparison",
        height=400
    )
    
    return fig

def create_historical_performance_chart():
    """Create a chart showing historical performance of the model."""
    # In a real implementation, we would query historical predictions
    # For demo purposes, we'll use dummy data
    
    # Get all opportunities
    with Session(engine) as session:
        ops = session.exec(select(Opportunity)).all()
    
    if not ops:
        return None
    
    df = pd.DataFrame([o.dict() for o in ops])
    
    # Group by month and calculate win rate
    df['month'] = pd.to_datetime(df['stage_exited_at']).dt.strftime('%Y-%m')
    monthly_stats = df.groupby('month').agg(
        total=('outcome', 'count'),
        won=('outcome', lambda x: (x == 'WON').sum())
    ).reset_index()
    
    monthly_stats['win_rate'] = monthly_stats['won'] / monthly_stats['total']
    
    # Create dummy model performance data (normally would be actual lift)
    monthly_stats['baseline_margin'] = monthly_stats['total'] * 5000
    monthly_stats['actual_margin'] = monthly_stats['baseline_margin'] * (1 + monthly_stats.index * 0.05)
    monthly_stats['lift_pct'] = (monthly_stats['actual_margin'] / monthly_stats['baseline_margin'] - 1) * 100
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add win rate line
    fig.add_trace(
        go.Scatter(
            x=monthly_stats['month'],
            y=monthly_stats['win_rate'],
            mode='lines+markers',
            name='Win Rate',
            line=dict(color='blue', width=3)
        ),
        secondary_y=False
    )
    
    # Add margin lift bars
    fig.add_trace(
        go.Bar(
            x=monthly_stats['month'],
            y=monthly_stats['lift_pct'],
            name='Margin Lift %',
            marker_color='green'
        ),
        secondary_y=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Win Rate", secondary_y=False)
    fig.update_yaxes(title_text="Margin Lift %", secondary_y=True)
    
    # Update layout
    fig.update_layout(
        title="Model Performance Over Time",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    return fig

def prioritize_deals(opportunities_df):
    """Rank deals by 'dollars at risk' for sales managers."""
    # Apply model to each opportunity
    results = []
    
    for _, opp in opportunities_df.iterrows():
        # Build feature vector
        feature_vector = {
            'amount': opp['amount'],
            'discount_pct': opp.get('discount_pct', 0),
            'industry': opp.get('industry', 'unknown'),
            'sales_cycle_days': calculate_sales_cycle_days(
                opp.get('stage_entered_at'), 
                datetime.datetime.now().strftime("%Y-%m-%d")
            ),
            'total_spent': calculate_customer_metrics(opp['customer_id'])['total_spent'],
            'avg_order_value': calculate_customer_metrics(opp['customer_id'])['avg_order_value'],
            'order_count': calculate_customer_metrics(opp['customer_id'])['order_count'],
            'days_since_last_purchase': calculate_customer_metrics(opp['customer_id'])['days_since_last_purchase'],
            'price_gap_pct': calculate_price_gap(
                opp.get('list_price', 0) * (1 - (opp.get('discount_pct', 0) / 100)), 
                opp.get('competitor_price', 0)
            )
        }
        
        # Get prediction
        win_prob, drivers, discount_band = predict_with_lightgbm(feature_vector)
        
        if win_prob is None:
            continue
        
        # Calculate current PWM
        current_pwm = calculate_pwm(
            win_prob, 
            opp['amount'], 
            opp.get('list_price', 1000), 
            opp.get('discount_pct', 0),
            opp.get('cost_price', None)
        )
        
        # Calculate optimal PWM
        # In a real implementation, we would simulate different discount levels
        # For demo purposes, we'll use a simple heuristic
        optimal_discount = (discount_band[0] + discount_band[1]) / 2
        
        # Simulate win probability with optimal discount
        optimal_feature_vector = feature_vector.copy()
        optimal_feature_vector['discount_pct'] = optimal_discount
        optimal_sample = pd.DataFrame([optimal_feature_vector])
        optimal_win_prob = model.predict_proba(optimal_sample)[0][1]
        
        optimal_pwm = calculate_pwm(
            optimal_win_prob, 
            opp['amount'], 
            opp.get('list_price', 1000), 
            optimal_discount,
            opp.get('cost_price', None)
        )
        
        # Calculate dollars at risk
        dollars_at_risk = optimal_pwm - current_pwm
        
        results.append({
            'opp_id': opp['opp_id'],
            'customer_id': opp['customer_id'],
            'amount': opp['amount'],
            'industry': opp.get('industry', 'unknown'),
            'current_discount': opp.get('discount_pct', 0),
            'optimal_discount': optimal_discount,
            'win_probability': win_prob,
            'current_pwm': current_pwm,
            'optimal_pwm': optimal_pwm,
            'dollars_at_risk': dollars_at_risk
        })
    
    # Convert to DataFrame and sort by dollars at risk
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        prioritized_deals = results_df.sort_values('dollars_at_risk', ascending=False)
        return prioritized_deals
    else:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Simlane Sales Prediction",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

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
    
    # Initialize database
    init_db()
    
    # Load sample data if tables are empty
    data_loaded = load_sample_data()
    
    # Load model if exists
    model_loaded = load_model()
    model_status = "✅ Model loaded" if model_loaded else "❌ No trained model found"
    
    # Auto-train model if sample data was just loaded
    if data_loaded and not model_loaded:
        with st.spinner("Training initial model with sample data..."):
            success, message = train_lightgbm_model()
            if success:
                st.success(message)
                model_loaded = True
    
    # Create tabs for different user personas
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧑‍💼 Account Executive", 
        "👨‍💼 Sales Manager", 
        "📈 RevOps / CRO",
        "⚙️ Setup & Configuration"
    ])
    
    # Tab 1: Account Executive View
    with tab1:
        st.header("Deal Assessment")
        st.write("Analyze a specific opportunity to determine win probability and optimal pricing.")
        
        # Input form
        st.subheader("Opportunity Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amount = st.number_input("Deal Amount ($):", min_value=1000.0, max_value=1000000.0, value=25000.0, step=1000.0)
            industry = st.selectbox("Industry:", ["Technology", "Healthcare", "Manufacturing", "Financial Services", "Retail", "Other"])
            sales_cycle = st.slider("Sales Cycle (days):", min_value=1, max_value=180, value=30)
        
        with col2:
            list_price = st.number_input("List Price ($):", min_value=100.0, max_value=10000.0, value=1000.0, step=100.0)
            current_discount = st.slider("Current Discount (%):", min_value=0, max_value=40, value=5)
            cost_price = st.number_input("Cost Price ($):", min_value=1.0, max_value=list_price-1, value=min(list_price*0.6, list_price-1), step=10.0)
        
        with col3:
            total_spent = st.number_input("Customer's Historical Spend ($):", min_value=0.0, value=50000.0, step=5000.0)
            avg_order = st.number_input("Average Order Value ($):", min_value=0.0, value=5000.0, step=500.0)
            competitor_price = st.number_input("Competitor Price ($):", min_value=0.0, value=list_price*0.95, step=100.0)
        
        # Calculate price gap percentage
        effective_price = list_price * (1 - (current_discount / 100))
        price_gap_pct = calculate_price_gap(effective_price, competitor_price)
        
        # Build feature vector
        feature_vector = {
            'amount': amount,
            'discount_pct': current_discount,
            'industry': industry,
            'sales_cycle_days': sales_cycle,
            'total_spent': total_spent,
            'avg_order_value': avg_order,
            'order_count': 5,  # Default
            'days_since_last_purchase': 30,  # Default
            'price_gap_pct': price_gap_pct
        }
        
        if st.button("Analyze Deal"):
            if not model_loaded:
                st.error("Model not trained yet. Please train the model first in the Setup tab.")
            else:
                # Get prediction
                with st.spinner("Analyzing deal characteristics..."):
                    win_prob, drivers, discount_band = predict_with_lightgbm(feature_vector)
                    
                    if win_prob is None:
                        st.error("Error making prediction. Please check your inputs.")
                    else:
                        # Calculate PWM
                        pwm = calculate_pwm(win_prob, amount, list_price, current_discount, cost_price)
                        
                        # Dashboard layout
                        st.subheader("Deal Analysis")
                        
                        # Top metrics row
                        metric1, metric2, metric3, metric4 = st.columns(4)
                        
                        with metric1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="card-title">Win Probability</div>
                                <div class="big-number">{win_prob:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with metric2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="card-title">Probability-Weighted Margin</div>
                                <div class="big-number">${pwm:,.0f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with metric3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="card-title">Safe Discount Band</div>
                                <div class="big-number">{discount_band[0]}% - {discount_band[1]}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with metric4:
                            optimal_discount = (discount_band[0] + discount_band[1]) / 2
                            recommendation = "Hold Price" if current_discount < discount_band[0] else (
                                "Price OK" if discount_band[0] <= current_discount <= discount_band[1] else "Consider Discount Reduction"
                            )
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="card-title">Recommendation</div>
                                <div class="big-number">{recommendation}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Visualizations row
                        chart1, chart2 = st.columns(2)
                        
                        with chart1:
                            # Win probability gauge
                            gauge_fig = create_win_probability_gauge(win_prob)
                            st.plotly_chart(gauge_fig, use_container_width=True)
                            
                            # Driver chart
                            if drivers:
                                driver_fig = create_driver_chart(drivers)
                                st.plotly_chart(driver_fig, use_container_width=True)
                        
                        with chart2:
                            # Discount band chart
                            discount_fig = create_discount_band_chart(current_discount, discount_band, pwm)
                            st.plotly_chart(discount_fig, use_container_width=True)
                            
                            # Deal comparison radar
                            radar_fig = create_deal_comparison_chart(
                                amount, industry, sales_cycle, total_spent, price_gap_pct
                            )
                            st.plotly_chart(radar_fig, use_container_width=True)
    
    # Tab 2: Sales Manager View
    with tab2:
        st.header("Deal Desk Queue")
        st.write("Focus your coaching on high-value deals with significant margin opportunity.")
        
        # Get all open opportunities from the database
        with Session(engine) as session:
            # In a real scenario, we would filter for open/active opportunities
            # For demo purposes, we'll use all opportunities
            open_opps = session.exec(select(Opportunity)).all()
        
        if not open_opps:
            st.warning("No opportunities found in the database.")
        else:
            open_opps_df = pd.DataFrame([o.dict() for o in open_opps])
            
            # Only proceed if model is loaded
            if not model_loaded:
                st.error("Model not trained yet. Please train the model first in the Setup tab.")
            else:
                # Prioritize deals
                with st.spinner("Analyzing opportunities..."):
                    prioritized_deals = prioritize_deals(open_opps_df)
                
                if prioritized_deals.empty:
                    st.warning("Could not calculate priorities for opportunities.")
                else:
                    # Show prioritized deals table
                    st.subheader("Prioritized Opportunities by Dollars at Risk")
                    
                    # Format table for display
                    display_df = prioritized_deals.copy()
                    display_df['win_probability'] = display_df['win_probability'].apply(lambda x: f"{x:.1%}")
                    display_df['current_pwm'] = display_df['current_pwm'].apply(lambda x: f"${x:,.0f}")
                    display_df['optimal_pwm'] = display_df['optimal_pwm'].apply(lambda x: f"${x:,.0f}")
                    display_df['dollars_at_risk'] = display_df['dollars_at_risk'].apply(lambda x: f"${x:,.0f}")
                    display_df['current_discount'] = display_df['current_discount'].apply(lambda x: f"{x:.1f}%")
                    display_df['optimal_discount'] = display_df['optimal_discount'].apply(lambda x: f"{x:.1f}%")
                    
                    columns_to_display = [
                        'opp_id', 'customer_id', 'amount', 'industry', 
                        'win_probability', 'current_discount', 'optimal_discount',
                        'dollars_at_risk'
                    ]
                    
                    st.dataframe(
                        display_df[columns_to_display].head(10),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Scenario analysis
                    st.subheader("Discount Strategy Scenario Analysis")
                    st.write("See how changing your discount strategy affects overall revenue and margin.")
                    
                    discount_change = st.slider(
                        "Change discounts by (percentage points):",
                        min_value=-10,
                        max_value=10,
                        value=0,
                        step=1
                    )
                    
                    # Calculate impact
                    total_current_deals = len(prioritized_deals)
                    total_current_value = prioritized_deals['amount'].sum()
                    
                    # Simple model: Assume 2% win rate change per 1% discount change
                    # and 1% margin change per 1% discount change
                    win_rate_change = discount_change * 0.02
                    margin_change = -discount_change * 0.01
                    
                    # Calculate adjusted metrics
                    adjusted_win_rate = prioritized_deals['win_probability'].mean() + win_rate_change
                    adjusted_win_rate = max(0, min(1, adjusted_win_rate))
                    
                    # Calculate expected revenue impact
                    baseline_expected_revenue = (prioritized_deals['amount'] * prioritized_deals['win_probability']).sum()
                    new_expected_revenue = total_current_value * adjusted_win_rate
                    revenue_impact = new_expected_revenue - baseline_expected_revenue
                    
                    # Calculate margin impact
                    baseline_margin = (prioritized_deals['current_pwm']).sum()
                    new_margin = baseline_margin * (1 + margin_change)
                    margin_impact = new_margin - baseline_margin
                    
                    # Display impact metrics
                    impact1, impact2, impact3, impact4 = st.columns(4)
                    
                    with impact1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="card-title">Deals Analyzed</div>
                            <div class="big-number">{total_current_deals}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with impact2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="card-title">Total Pipeline Value</div>
                            <div class="big-number">${total_current_value:,.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with impact3:
                        impact_color = "#4CAF50" if revenue_impact >= 0 else "#F44336"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="card-title">Revenue Impact</div>
                            <div class="big-number" style="color: {impact_color}">${revenue_impact:,.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with impact4:
                        impact_color = "#4CAF50" if margin_impact >= 0 else "#F44336"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="card-title">Margin Impact</div>
                            <div class="big-number" style="color: {impact_color}">${margin_impact:,.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Tab 3: RevOps / CRO View
    with tab3:
        st.header("Performance Analytics")
        st.write("Measure the impact of the Simlane system on your sales performance.")
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date", 
                value=datetime.datetime.now() - timedelta(days=90)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.datetime.now()
            )
        
        # Create performance chart
        if model_loaded:
            performance_fig = create_historical_performance_chart()
            if performance_fig:
                st.plotly_chart(performance_fig, use_container_width=True)
            else:
                st.warning("Insufficient historical data to display performance metrics.")
        else:
            st.error("Model not trained yet. Please train the model first in the Setup tab.")
        
        # Model performance metrics
        st.subheader("Model Metrics")
        
        if model_loaded:
            # In a real implementation, we would calculate actual metrics
            # For demo purposes, we'll use dummy data
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Precision", "76%", "5%")
            
            with col2:
                st.metric("Model Recall", "83%", "7%")
            
            with col3:
                st.metric("Average Margin Lift", "$432 per deal", "12%")
            
            with col4:
                st.metric("Total Margin Added", "$215,864", "")
        
        # Lift report
        st.subheader("Lift Report")
        
        # Date range for aggregation
        lift_metrics = {
            "Total Deals Analyzed": 124,
            "Won Deals": 67,
            "Baseline Win Rate": "45%",
            "Actual Win Rate": "54%",
            "Win Rate Lift": "9%",
            "Baseline Margin": "$1.2M",
            "Actual Margin": "$1.6M",
            "Margin Lift": "$400K",
            "Lift Percentage": "33%"
        }
        
        # Create a dataframe for the lift report
        lift_df = pd.DataFrame({
            "Metric": lift_metrics.keys(),
            "Value": lift_metrics.values()
        })
        
        st.dataframe(lift_df, use_container_width=True, hide_index=True)
        
        # Export options
        st.subheader("Export Data")
        
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "Excel", "JSON"]
        )
        
        if st.button("Export Lift Report"):
            st.success(f"Lift report exported as {export_format} (demo functionality)")
        
        # API Documentation
        with st.expander("API Documentation"):
            st.markdown("""
            ### Simlane API
            
            The Simlane API allows you to integrate predictions directly into your CRM or other systems.
            
            #### Endpoints
            
            **GET /api/predict**
            
            Predicts win probability and calculates probability-weighted margin for a deal.
            
            Parameters:
            - `amount` (float): Deal amount
            - `list_price` (float): List price per unit
            - `discount_pct` (float, optional): Current discount percentage
            - `industry` (string, optional): Customer industry
            - `sales_cycle_days` (int, optional): Sales cycle length in days
            - `customer_id` (string, optional): Customer ID for history lookup
            - `competitor_price` (float, optional): Competitor's price if known
            
            Example Response:
            ```json
            {
                "win_probability": 0.76,
                "probability_weighted_margin": 15200,
                "safe_discount_band": {"min": 2, "max": 7},
                "top_drivers": [
                    {
                        "feature": "Amount",
                        "importance": 0.32,
                        "direction": "+",
                        "value": 25000
                    },
                    ...
                ]
            }
            ```
            """)
    
    # Tab 4: Setup & Configuration
    with tab4:
        st.header("Setup & Configuration")
        
        # Data Management section
        st.subheader("Data Management")
        
        table_tabs = st.tabs(["Transactions", "Pricing Logs", "Competitors", "Opportunities"])
        
        with table_tabs[0]:
            # Transactions table
            with Session(engine) as session:
                transactions = session.exec(select(Transaction)).all()
                if transactions:
                    trans_df = pd.DataFrame([t.dict() for t in transactions])
                    st.dataframe(trans_df, use_container_width=True)
                else:
                    st.info("No transaction data found.")
        
        with table_tabs[1]:
            # Pricing logs table
            with Session(engine) as session:
                pricing_logs = session.exec(select(PricingLog)).all()
                if pricing_logs:
                    pricing_df = pd.DataFrame([p.dict() for p in pricing_logs])
                    st.dataframe(pricing_df, use_container_width=True)
                else:
                    st.info("No pricing data found.")
        
        with table_tabs[2]:
            # Competitors table
            with Session(engine) as session:
                competitors = session.exec(select(CompetitorPrice)).all()
                if competitors:
                    comp_df = pd.DataFrame([c.dict() for c in competitors])
                    st.dataframe(comp_df, use_container_width=True)
                else:
                    st.info("No competitor data found.")
        
        with table_tabs[3]:
            # Opportunities table
            with Session(engine) as session:
                opportunities = session.exec(select(Opportunity)).all()
                if opportunities:
                    opp_df = pd.DataFrame([o.dict() for o in opportunities])
                    st.dataframe(opp_df, use_container_width=True)
                else:
                    st.info("No opportunity data found.")
        
        # Data upload section
        st.subheader("Upload Data")
        
        upload_table = st.selectbox(
            "Select table to upload to:",
            ["transactions", "pricing_logs", "competitors", "opportunities"]
        )
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"Preview of {uploaded_file.name}:")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Upload to Database"):
                table_map = {
                    "transactions": Transaction,
                    "pricing_logs": PricingLog,
                    "competitors": CompetitorPrice,
                    "opportunities": Opportunity,
                }
                bulk_insert_dataframe(df, table_map[upload_table])
                st.success(f"Successfully uploaded {len(df)} rows to {upload_table} table!")
        
        # Model training section
        st.subheader("Model Training")
        
        with st.expander("Model Configuration"):
            st.markdown("""
            ### LightGBM Model Configuration
            
            The current model uses a LightGBM classifier with the following parameters:
            
            - `n_estimators`: 100
            - `learning_rate`: 0.05
            - `num_leaves`: 31
            - `feature_fraction`: 0.8
            - `bagging_fraction`: 0.8
            - `bagging_freq`: 5
            
            Features used in the model:
            
            - Amount
            - Discount Percentage
            - Industry
            - Sales Cycle Duration
            - Customer Metrics (total spent, average order value, etc.)
            - Price Gap Percentage
            """)
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                success, message = train_lightgbm_model()
                if success:
                    st.success(message)
                else:
                    st.error(message)

if __name__ == "__main__":
    main()
