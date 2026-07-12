import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Set page layout to wide
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
        background-color: #4F46E5; color: white; font-weight: bold;
        border-radius: 8px; padding: 0.6rem 2rem; border: none; transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #4338CA; transform: translateY(-2px); }
    .metric-card {
        background-color: white; padding: 1.5rem; border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border-left: 5px solid #4F46E5;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 BigMart Enterprise Sales Forecasting Dashboard")
st.markdown("An advanced predictive system using machine learning to simulate retail item performance.")
st.markdown("---")

# LIVE SERVER TRAINING ENGINE (CRASH PROOF)
@st.cache_resource
def train_model_live():
    # Load the compressed small dataset straight from repository
    train_df = pd.read_csv('train_small.csv')
    
    # Preprocessing
    train_df['Item_Fat_Content'] = train_df['Item_Fat_Content'].replace(
        {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}
    )
    train_df['Item_Weight'] = train_df['Item_Weight'].fillna(train_df['Item_Weight'].mean())
    train_df['Outlet_Size'] = train_df['Outlet_Size'].fillna(train_df['Outlet_Size'].mode()[0])
    
    encoders = {}
    features_to_encode = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
    
    for col in features_to_encode:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        encoders[col] = le
        
    X = train_df.drop(columns=['Item_Outlet_Sales'])
    y = train_df['Item_Outlet_Sales']
    
    # Optimized training grid
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X, y)
    
    return model, encoders, X.columns.tolist()

try:
    with st.spinner("⚙️ Synchronizing machine learning pipelines live on Streamlit runtime..."):
        model, encoders, feature_names = train_model_live()
except FileNotFoundError:
    st.error("⚠️ 'train_small.csv' missing from repository. Please upload your dataset to GitHub.")
    st.stop()

# Layout Configuration
with st.sidebar:
    st.header("⚙️ Feature Engineering Inputs")
    
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

if trigger_forecast:
    input_data = pd.DataFrame([{
        'Item_Weight': item_weight, 'Item_Fat_Content': fat_content, 'Item_Visibility': item_visibility,
        'Item_Type': item_type, 'Item_MRP': item_mrp, 'Outlet_Establishment_Year': establishment_year,
        'Outlet_Size': outlet_size, 'Outlet_Location_Type': location_type, 'Outlet_Type': outlet_type
    }])
    
    for col in encoders.keys():
        input_data[col] = encoders[col].transform(input_data[col])
        
    input_data = input_data[feature_names]
    prediction = model.predict(input_data)[0]
    
    st.subheader("🎯 Optimization & Forecast Analytics")
    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        st.markdown(f'<div class="metric-card"><p style="color:#6B7280; font-size:14px; margin:0;">FORECASTED OUTLET SALES</p><h2 style="color:#111827; margin:5px 0;">${prediction:,.2f}</h2><p style="color:#10B981; font-size:12px; margin:0;">▲ Live optimized pipeline calculation</p></div>', unsafe_allow_html=True)
    with kpi2:
        margin_status = "High Performance" if prediction > 2200 else "Standard Velocity"
        st.markdown(f'<div class="metric-card"><p style="color:#6B7280; font-size:14px; margin:0;">VELOCITY TIERING</p><h2 style="color:#111827; margin:5px 0;">{margin_status}</h2><p style="color:#6B7280; font-size:12px; margin:0;">Evaluated against dataset baseline</p></div>', unsafe_allow_html=True)
    with kpi3:
        efficiency = (item_mrp / (prediction + 1)) * 100
        st.markdown(f'<div class="metric-card"><p style="color:#6B7280; font-size:14px; margin:0;">PRICE TO INFERENCE RATIO</p><h2 style="color:#111827; margin:5px 0;">{efficiency:.2f}%</h2><p style="color:#3B82F6; font-size:12px; margin:0;">Inherent demand velocity rating</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Algorithmic Insight Engine")
    graph_col1, graph_col2 = st.columns(2)
    
    with graph_col1:
        st.write("**Simulated Revenue Projection Boundaries (Distribution)**")
        simulated_distribution = np.random.normal(prediction, prediction * 0.15, 200)
        hist_values, bin_edges = np.histogram(simulated_distribution, bins=20)
        chart_data = pd.DataFrame({
            'Sales Boundaries ($)': bin_edges[:-1].astype(int),
            'Probability Density': hist_values
        }).set_index('Sales Boundaries ($)')
        st.area_chart(chart_data, color="#4F46E5")

    with graph_col2:
        st.write("**Feature Importance Mapping (Relative Weights)**")
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Features': feature_names,
            'Contribution Weight': importances
        }).sort_values(by='Contribution Weight', ascending=True).set_index('Features')
        st.bar_chart(importance_df, color="#10B981")
else:
    st.info("💡 Adjust product engineering parameters in the left sidebar configuration window and press **Execute Forecast Engine** to evaluate system performance matrices.")
