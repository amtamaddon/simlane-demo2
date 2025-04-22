import sys
import os
import streamlit as st
from sqlmodel import Field, SQLModel, create_engine, Session, select
from typing import Optional, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
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
            
            # Sample opportunity data
            opportunity_data = """
opp_id,customer_id,stage_entered_at,stage_exited_at,amount,outcome,industry
OPP1001,C101,2024-01-01,2024-01-15,12000,WON,Technology
OPP1002,C102,2024-01-05,2024-01-25,25000,WON,Healthcare
OPP1003,C103,2024-01-10,2024-02-10,45000,WON,Financial Services
OPP1004,C104,2024-01-15,2024-02-15,8000,LOST,Retail
OPP1005,C105,2024-01-20,2024-02-20,30000,WON,Manufacturing
OPP1006,C106,2024-01-25,2024-02-25,55000,WON,Technology
OPP1007,C107,2024-02-01,2024-03-01,15000,LOST,Healthcare
OPP1008,C108,2024-02-05,2024-03-05,22000,LOST,Retail
OPP1009,C109,2024-02-10,2024-03-10,18000,WON,Financial Services
OPP1010,C110,2024-02-15,2024-03-15,7000,LOST,Manufacturing
OPP1011,C111,2024-02-20,2024-03-20,65000,WON,Technology
OPP1012,C112,2024-02-25,2024-03-25,28000,WON,Healthcare
OPP1013,C113,2024-03-01,2024-04-01,42000,LOST,Financial Services
OPP1014,C114,2024-03-05,2024-04-05,33000,WON,Retail
OPP1015,C115,2024-03-10,2024-04-10,50000,WON,Manufacturing
OPP1016,C116,2024-03-15,2024-04-15,17000,LOST,Technology
OPP1017,C117,2024-03-20,2024-04-20,26000,WON,Healthcare
OPP1018,C118,2024-03-25,2024-04-25,38000,WON,Financial Services
OPP1019,C101,2024-01-30,2024-02-28,10000,WON,Technology
OPP1020,C103,2024-02-15,2024-03-15,35000,LOST,Financial Services
OPP1021,C106,2024-03-05,2024-04-05,48000,WON,Technology
OPP1022,C109,2024-03-20,2024-04-20,22000,WON,Financial Services
OPP1023,C112,2024-04-01,2024-04-30,31000,LOST,Healthcare
OPP1024,C114,2024-04-10,2024-05-10,27000,WON,Retail
OPP1025,C101,2024-04-15,2024-05-15,14000,WON,Technology
OPP1026,C105,2024-04-20,2024-05-20,36000,LOST,Manufacturing
OPP1027,C111,2024-04-25,2024-05-25,72000,WON,Technology
OPP1028,C115,2024-05-01,2024-05-30,45000,WON,Manufacturing
OPP1029,C103,2024-05-05,2024-06-05,29000,LOST,Financial Services
OPP1030,C118,2024-05-10,2024-06-10,41000,WON,Financial Services
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

def get_competitive_position(product_id):
    """Determine competitive position based on pricing."""
    with Session(engine) as session:
        # Get our latest price
        our_price_record = session.exec(
            select(PricingLog).where(PricingLog.product_id == product_id)
        ).first()
        
        # Get competitor prices
        competitor_prices = session.exec(
            select(CompetitorPrice).where(CompetitorPrice.product_id == product_id)
        ).all()
        
        if not our_price_record or not competitor_prices:
            return {
                "price_position": "unknown",
                "price_difference_pct": 0
            }
        
        our_price = our_price_record.final_price or our_price_record.list_price
        if not our_price:
            return {
                "price_position": "unknown",
                "price_difference_pct": 0
            }
        
        # Calculate average competitor price
        comp_prices = [cp.price for cp in competitor_prices if cp.price is not None]
        if not comp_prices:
            return {
                "price_position": "unknown",
                "price_difference_pct": 0
            }
            
        avg_competitor_price = sum(comp_prices) / len(comp_prices)
        price_diff_pct = (our_price - avg_competitor_price) / avg_competitor_price * 100
        
        if price_diff_pct <= -10:
            position = "significantly_lower"
        elif price_diff_pct < 0:
            position = "lower"
        elif price_diff_pct < 10:
            position = "comparable"
        else:
            position = "higher"
            
        return {
            "price_position": position,
            "price_difference_pct": price_diff_pct
        }

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
            "industry": opp.industry if not pd.isna(opp.industry) else "unknown",
            "outcome": 1 if str(opp.outcome).upper() == "WON" else 0
        }
        
        # Calculate sales cycle duration if dates are available
        if not pd.isna(opp.stage_entered_at) and not pd.isna(opp.stage_exited_at):
            # In a real implementation, we'd parse dates and calculate actual duration
            features["sales_cycle_days"] = 30  # Placeholder
        else:
            features["sales_cycle_days"] = None
            
        # Add customer metrics
        customer_metrics = calculate_customer_metrics(opp.customer_id)
        features.update(customer_metrics)
        
        # Add competitive position if we have product info
        # In this simplified version, we don't have product_id in opportunities
        # In a real implementation, we'd link to product data
        features["price_position"] = "unknown"
        features["price_difference_pct"] = 0
        
        enriched_data.append(features)
    
    return pd.DataFrame(enriched_data)

# -----------------------------------------------------------------------------
# Model management
# -----------------------------------------------------------------------------

MODEL_PATH = "propensity_advanced.pkl"
model = None

def load_model():
    global model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return True
    except FileNotFoundError:
        return False

def train_advanced_model():
    global model
    
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
    categorical_features = ['industry', 'price_position']
    numeric_features = ['amount', 'sales_cycle_days', 'total_spent', 
                        'avg_order_value', 'order_count', 
                        'days_since_last_purchase', 'price_difference_pct']
    
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
    
    # Create and train pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    X = enriched_df.drop(['opp_id', 'outcome'], axis=1)
    y = enriched_df['outcome']
    
    # Train model
    pipeline.fit(X, y)
    
    # Evaluate with cross-validation
    try:
        cv_scores = cross_val_score(pipeline, X, y, cv=5)
        avg_score = np.mean(cv_scores)
        score_message = f"Cross-validation accuracy: {avg_score:.2f}"
    except ValueError:
        # Not enough samples for cross-validation
        score_message = "Not enough samples for cross-validation"
    
    # Save the model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    
    model = pipeline
    
    # Get feature importance
    feature_names = None
    try:
        feature_names = (
            numeric_features + 
            list(pipeline.named_steps['preprocessor']
                .named_transformers_['cat']
                .named_steps['onehot']
                .get_feature_names_out(categorical_features))
        )
    except:
        pass
    
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    # Return success and statistics
    return True, f"Model trained successfully with {len(enriched_df)} samples! {score_message}"

def predict_with_advanced_model(amount, industry="Technology", sales_cycle_days=30, total_spent=10000, avg_order_value=2000, order_count=5):
    global model
    if model is None:
        load_model()
        if model is None:
            return None, None
    
    # Create a sample with provided data and defaults for other fields
    sample = pd.DataFrame([{
        'amount': amount,
        'industry': industry,
        'sales_cycle_days': sales_cycle_days,
        'total_spent': total_spent,
        'avg_order_value': avg_order_value,
        'order_count': order_count,
        'days_since_last_purchase': 30,
        'price_position': 'comparable',
        'price_difference_pct': 0
    }])
    
    # Get prediction probability
    probs = model.predict_proba(sample)
    win_prob = probs[0][1]  # Probability of class 1 (win)
    
    # Get feature importance for this prediction (SHAP values would be better,
    # but for simplicity we'll just use global feature importance)
    importance_pairs = []
    try:
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
            
            # Get feature names (simplified approach)
            feature_names = []
            for name, transformer in model.named_steps['preprocessor'].transformers_:
                if name == 'num':
                    feature_names.extend(transformer.get_feature_names_out([
                        'amount', 'sales_cycle_days', 'total_spent', 
                        'avg_order_value', 'order_count', 
                        'days_since_last_purchase', 'price_difference_pct'
                    ]))
                elif name == 'cat':
                    try:
                        feature_names.extend(transformer.named_steps['onehot']
                                            .get_feature_names_out(['industry', 'price_position']))
                    except:
                        # Fallback if feature names can't be extracted
                        feature_names.extend([f'cat_{i}' for i in range(10)])
            
            # Pair feature names with importance values
            importance_pairs = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
    except:
        importance_pairs = []
    
    return win_prob, importance_pairs

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def main():
    st.title("Simlane MVP Dashboard")
    
    # Initialize database
    init_db()
    
    # Load sample data if tables are empty
    data_loaded = load_sample_data()
    
    # Load model if exists
    model_loaded = load_model()
    model_status = "✅ Model loaded" if model_loaded else "❌ No trained model found"
    
    st.sidebar.header("Model Status")
    st.sidebar.write(model_status)
    
    # Auto-train model if sample data was just loaded
    if data_loaded and not model_loaded:
        with st.spinner("Training initial model with sample data..."):
            success, message = train_advanced_model()
            if success:
                st.success(message)
                model_loaded = True
    
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
        st.write("This will train a Random Forest model to predict the probability of winning an opportunity based on multiple features.")
        
        with st.expander("Model Details"):
            st.markdown("""
            ### Features Used in the Model
            
            The enhanced model uses the following features:
            
            **Opportunity-specific features:**
            - Amount
            - Industry
            - Sales cycle duration
            
            **Customer relationship features:**
            - Total historical spend
            - Average order value
            - Number of previous orders
            - Days since last purchase
            
            **Competitive factors:**
            - Price position relative to competitors
            - Price difference percentage
            
            ### Model Architecture
            
            - Random Forest classifier with 100 trees
            - Handles missing values with imputation
            - Standardizes numeric features
            - One-hot encodes categorical features
            """)
        
        if st.button("Train Advanced Model"):
            with st.spinner("Training model..."):
                success, message = train_advanced_model()
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # Tab 3: Make Predictions
    with tab3:
        st.header("Predict Win Probability")
        
        # Input fields for prediction
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Opportunity Amount ($):", min_value=0.0, value=10000.0)
            industry = st.selectbox(
                "Industry:",
                ["Technology", "Healthcare", "Manufacturing", "Financial Services", "Retail", "Other"]
            )
            sales_cycle = st.slider("Sales Cycle (days):", min_value=1, max_value=365, value=30)
        
        with col2:
            total_spent = st.number_input("Customer's Historical Spend ($):", min_value=0.0, value=50000.0)
            avg_order = st.number_input("Average Order Value ($):", min_value=0.0, value=5000.0)
            order_count = st.number_input("Previous Order Count:", min_value=0, value=10)
        
        if st.button("Predict Win Probability"):
            prob, importances = predict_with_advanced_model(
                amount=amount,
                industry=industry,
                sales_cycle_days=sales_cycle,
                total_spent=total_spent,
                avg_order_value=avg_order,
                order_count=order_count
            )
            
            if prob is None:
                st.error("Model not trained yet. Please train the model first.")
            else:
                st.success(f"Win Probability: {prob:.2%}")
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gauge chart
                    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                    
                    # Gauge visualization
                    pos = prob * 180  # convert to angle (half circle)
                    
                    # Create gauge background
                    theta = np.linspace(0, np.pi, 100)
                    ax.plot(theta, [1] * 100, color='lightgray', linewidth=8)
                    
                    # Create gauge value
                    theta_val = np.linspace(0, np.pi * prob, 100)
                    ax.plot(theta_val, [1] * len(theta_val), color='green' if prob > 0.5 else 'orange', linewidth=8)
                    
                    # Customize gauge
                    ax.set_rticks([])  # No radial ticks
                    ax.set_thetagrids([0, 45, 90, 135, 180], labels=['0%', '25%', '50%', '75%', '100%'])
                    ax.spines['polar'].set_visible(False)
                    ax.set_title('Win Probability', fontsize=14)
                    
                    st.pyplot(fig)
                    
                with col2:
                    # Show feature importance if available
                    if importances:
                        st.subheader("Top Factors")
                        # Get top 5 features
                        top_features = importances[:5]
                        
                        feature_names = [name.replace('_', ' ').replace('cat ', '').title() for name, _ in top_features]
                        importance_vals = [imp for _, imp in top_features]
                        
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.barh(feature_names, importance_vals)
                        ax.set_xlabel('Importance')
                        ax.set_title('Key Factors in Prediction')
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
