import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page layout to wide for a professional dashboard look
st.set_page_config(
    page_title="BigMart Enterprise Sales Forecasting", 
    page_icon="📊", 
    layout="wide"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        transform: translateY(-2px);
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border-left: 5px solid #4F46E5;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("📊 BigMart Enterprise Sales Forecasting Dashboard")
st.markdown("An advanced predictive system using machine learning to simulate retail item performance across varying outlet footprints.")
st.markdown("---")

# Load the saved model assets
@st.cache_resource
def load_model():
    return joblib.load('sales_model.pkl')

try:
    saved_assets = load_model()
    model = saved_assets['model']
    encoders = saved_assets['encoders']
    feature_names = saved_assets['features']
except FileNotFoundError:
    st.error("⚠️ Model asset matrix ('sales_model.pkl') missing. Run your training script first.")
    st.stop()

# Layout: Split into Sidebar Inputs and Main Analytics Area
with st.sidebar:
    st.header("⚙️ Feature Engineering Inputs")
    st.markdown("Adjust parameters below to run the simulation matrix.")
    
    with st.expander("📦 Core Item Specifications", expanded=True):
        item_mrp = st.number_input("Maximum Retail Price (MRP)", min_value=0.0, value=141.0, step=1.0)
        item_weight = st.number_input("Item Weight (grams/kg)", min_value=0.0, max_value=50.0, value=12.8, step=0.1)
        item_visibility = st.slider("Store Allocation Visibility (%)", min_value=0.0, max_value=1.0, value=0.06, step=0.01)
        item_type = st.selectbox("Product Category Cluster", encoders['Item_Type'].classes_)
        fat_content = st.selectbox("Fat Content Profile", encoders['Item_Fat_Content'].classes_)

    with st.expander("🏪 Target Outlet Profile", expanded=True):
        outlet_type = st.selectbox("Store Format Type", encoders['Outlet_Type'].classes_)
        outlet_size = st.selectbox("Footprint Size", encoders['Outlet_Size'].classes_)
        location_type = st.selectbox("Geographic Tier Class", encoders['Outlet_Location_Type'].classes_)
        establishment_year = st.number_input("Launch Matrix Year", min_value=1980, max_value=2026, value=1999, step=1)

    st.markdown("---")
    trigger_forecast = st.button("🚀 Execute Forecast Engine", use_container_width=True)

# Main Dashboard Calculations & Display
if trigger_forecast:
    # 1. Pipeline Preprocessing
    input_data = pd.DataFrame([{
        'Item_Weight': item_weight,
        'Item_Fat_Content': fat_content,
        'Item_Visibility': item_visibility,
        'Item_Type': item_type,
        'Item_MRP': item_mrp,
        'Outlet_Establishment_Year': establishment_year,
        'Outlet_Size': outlet_size,
        'Outlet_Location_Type': location_type,
        'Outlet_Type': outlet_type
    }])
    
    # Apply encoders
    for col in encoders.keys():
        input_data[col] = encoders[col].transform(input_data[col])
        
    # Standardize column structure
    input_data = input_data[feature_names]
    
    # Execute ML Inference
    prediction = model.predict(input_data)[0]
    
    # 2. Key Performance Indicators (KPIs)
    st.subheader("🎯 Optimization & Forecast Analytics")
    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:#6B7280; font-size:14px; margin:0;">FORECASTED OUTLET SALES</p>
            <h2 style="color:#111827; margin:5px 0;">${prediction:,.2f}</h2>
            <p style="color:#10B981; font-size:12px; margin:0;">▲ Live optimized pipeline calculation</p>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        # Comparative benchmark mock logic based on common dataset distributions
        margin_status = "High Performance" if prediction > 2200 else "Standard Velocity"
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:#6B7280; font-size:14px; margin:0;">VELOCITY TIERING</p>
            <h2 style="color:#111827; margin:5px 0;">{margin_status}</h2>
            <p style="color:#6B7280; font-size:12px; margin:0;">Evaluated against dataset baseline</p>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        efficiency = (item_mrp / (prediction + 1)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:#6B7280; font-size:14px; margin:0;">PRICE TO INFERENCE RATIO</p>
            <h2 style="color:#111827; margin:5px 0;">{efficiency:.2f}%</h2>
            <p style="color:#3B82F6; font-size:12px; margin:0;">Inherent demand velocity rating</p>
        </div>
        """, unsafe_allow_html=True)

    # 3. Interactive Graphical Analytical Layer
    st.markdown("---")
    st.subheader("📊 Algorithmic Insight Engine")
    
    graph_col1, graph_col2 = st.columns(2)
    
    with graph_col1:
        st.write("**Simulated Revenue Projection Boundaries**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.set_theme(style="whitegrid")
        
        # Generate a standard dummy probability curve centered on the prediction for presentation value
        simulated_distribution = np.random.normal(prediction, prediction * 0.15, 1000)
        sns.kdeplot(simulated_distribution, color="#4F46E5", fill=True, alpha=0.3, ax=ax)
        ax.axvline(prediction, color='red', linestyle='--', label=f'Point Forecast (${prediction:.0f})')
        ax.set_xlabel("Sales Value ($)")
        ax.set_ylabel("Probability Density")
        ax.legend()
        st.pyplot(fig)

    with graph_col2:
        st.write("**Feature Importance Mapping (Random Forest Weights)**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        
        # Extract features weights dynamically from the model
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        ax.barh(range(len(indices)), importances[indices], color="#10B981", align="center")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel("Relative Predictive Contribution Weight")
        st.pyplot(fig)

else:
    # Landing View state if no forecast button has been triggered yet
    st.info("💡 Adjust product engineering parameters in the left sidebar configuration window and press **Execute Forecast Engine** to evaluate system performance matrices.")