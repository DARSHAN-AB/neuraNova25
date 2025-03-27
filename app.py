import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from datetime import datetime
import cv2
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
import pickle
import requests
from bs4 import BeautifulSoup

# Page configuration
st.set_page_config(
    page_title="ForestGuardian 🌳",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'

# Custom CSS with Bright Colors
st.markdown("""
<style>
    /* General Styling */
    body {
        background-color: #F5F7FA;
    }
    /* Sidebar Styling */
    .stButton > button {
        width: 100%;
        text-align: left;
        padding: 0.75rem 1rem;
        background-color: #28A745;
        color: #FFFFFF;
        border: none;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #FFC107;
        color: #2C3E50;
    }
    .sidebar-section {
        padding: 1rem;
        background: linear-gradient(135deg, #28A745, #218838);
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    /* Dashboard Styling */
    .metric-card {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 5px solid #28A745;
    }
    .metric-card h3 {
        color: #2C3E50;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        color: #28A745;
        font-size: 1.5rem;
        font-weight: bold;
    }
    /* News Card Styling */
    .news-card {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        transition: transform 0.2s;
        border-left: 5px solid #FFC107;
    }
    .news-card:hover {
        transform: translateY(-5px);
    }
    .news-title {
        color: #2C3E50;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .news-source {
        color: #FD7E14;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .news-summary {
        color: #333;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .news-link {
        color: #28A745;
        text-decoration: none;
        font-weight: bold;
    }
    .news-link:hover {
        text-decoration: underline;
        color: #FFC107;
    }
    /* Stats Box Styling */
    .stats-box {
        background: linear-gradient(135deg, #28A745, #218838);
        color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .stats-box h3 {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .stats-box p {
        font-size: 1.2rem;
    }
    /* Titles and Headers */
    h1, h2, h3 {
        color: #2C3E50;
    }
    /* Alerts */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h1 style='color: #FFFFFF; font-size: 1.8rem;'>🌳 ForestGuardian</h1>
            <p style='color: #FFC107;'>Forest Monitoring System</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🏠 Dashboard"):
        st.session_state.page = 'Dashboard'
    if st.button("🤖 Model Prediction"):
        st.session_state.page = 'Model Prediction'
    if st.button("📰 Recent News"):
        st.session_state.page = 'Recent News'
    if st.button("📉 Global Stats"):
        st.session_state.page = 'Global Stats'

# Model-related globals
labels = ['Urban Land', 'Agricultural Land', 'Range Land', 'Forest Land']

# Load the DeepLab V3+ model
@st.cache_resource
def load_model():
    json_file = 'model/model.json'
    weights_file = 'model/model.weights.h5'
    with open(json_file, "r") as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    return model

# Simulated real-time data fetch (aligned with Global Deforestation Ratio)
def fetch_real_time_data():
    forest_coverage = "31%"  # Current Forest Cover: 31% of global land (FAO, 2024)
    areas_monitored = "4.7M km²"  # Derived from annual loss context
    alert_accuracy = "90%"  # Hypothetical, based on tech improvements
    risk_areas = "42"  # Hypothetical, based on tropical loss trends
    return forest_coverage, areas_monitored, alert_accuracy, risk_areas

# Simulated news fetching with proper structure
def fetch_recent_news():
    news_articles = [
        {
            "title": "Brazil Amazon: Deforestation dropped 30.6% in 2023",
            "source": "Mongabay, Nov 2024",
            "url": "https://news.mongabay.com",
            "summary": "Lowest in 9 years under new policies."
        },
        {
            "title": "Indonesia: Lost 26,000 sq km in 2024",
            "source": "SCMP, Feb 2025",
            "url": "https://www.scmp.com",
            "summary": "Due to mining and farming expansion."
        },
        {
            "title": "Global: 6.37M hectares lost in 2023",
            "source": "Forest Declaration, Oct 2024",
            "url": "https://forestdeclaration.org",
            "summary": "45% above 2030 targets."
        }
    ]
    return news_articles

# Simulated global stats fetching
def fetch_global_stats():
    stats = {
        "Current Forest Cover": "31% of global land (FAO, 2024)",
        "Annual Loss": "4.7M hectares/year (2010-2020 avg, FAO)",
        "Tropical Loss": "9.1M acres in 2023 (Forest Declaration)"
    }
    return stats

# Main Content Functions
def show_dashboard():
    st.title("🌍 Forest Monitoring Dashboard")
    
    # Fetch real-time data
    forest_coverage, areas_monitored, alert_accuracy, risk_areas = fetch_real_time_data()
    
    # Metrics with real-time data
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Forest Coverage</h3>
            <p>{forest_coverage}</p>
            <small style="color: #FD7E14;">-0.5%</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Areas Monitored</h3>
            <p>{areas_monitored}</p>
            <small style="color: #28A745;">+200 km²</small>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Alert Accuracy</h3>
            <p>{alert_accuracy}</p>
            <small style="color: #28A745;">+1.2%</small>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Risk Areas</h3>
            <p>{risk_areas}</p>
            <small style="color: #FD7E14;">+3</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    st.subheader("📈 Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        dates = pd.date_range(start='2024-01-01', end='2025-03-27', freq='M')
        values = np.random.normal(100, 10, len(dates))
        df = pd.DataFrame({'Date': dates, 'Deforestation (ha)': values})
        fig = px.line(df, x='Date', y='Deforestation (ha)', title='Deforestation Trend Analysis')
        fig.update_traces(line_color='#28A745')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        causes = ['Agriculture', 'Urban Development', 'Logging', 'Natural']
        values = [40, 25, 20, 15]
        fig = px.pie(values=values, names=causes, title='Causes of Deforestation', 
                     color_discrete_sequence=['#28A745', '#FFC107', '#FD7E14', '#2C3E50'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Alerts
    st.subheader("🚨 Recent Alerts")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.error("🚨 High Risk Alert\n\nLarge deforestation detected")
    with col2:
        st.warning("⚠️ Medium Risk Alert\n\nUnusual activity detected")
    with col3:
        st.success("✅ Low Risk Alert\n\nSmall changes detected")

def calculate_forest_area(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([36, 25, 25])
    upper = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    forest_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]
    forest_percentage = (forest_pixels / total_pixels) * 100
    return forest_percentage, mask

def show_model_prediction():
    st.title("🤖 DeepLab V3+ Model Prediction")
    classifier = load_model()
    st.subheader("Upload Two Satellite Images for Deforestation Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file1 = st.file_uploader("Earlier Image", type=['png', 'jpg', 'jpeg'], key="earlier")
        if uploaded_file1:
            image1 = Image.open(uploaded_file1)
            st.image(image1, caption="Earlier Image", use_container_width=True)
    
    with col2:
        uploaded_file2 = st.file_uploader("Later Image", type=['png', 'jpg', 'jpeg'], key="later")
        if uploaded_file2:
            image2 = Image.open(uploaded_file2)
            st.image(image2, caption="Later Image", use_container_width=True)
    
    if st.button("Analyze Deforestation", key="analyze_btn"):
        image1_cv = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        image2_cv = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
        img1 = cv2.resize(image1_cv, (64, 64)).astype('float32') / 255
        img1 = np.expand_dims(img1, axis=0)
        img2 = cv2.resize(image2_cv, (64, 64)).astype('float32') / 255
        img2 = np.expand_dims(img2, axis=0)
        
        preds1 = classifier.predict(img1)
        predict1 = np.argmax(preds1)
        preds2 = classifier.predict(img2)
        predict2 = np.argmax(preds2)
        
        forest_percentage1, mask1 = calculate_forest_area(image1_cv)
        forest_percentage2, mask2 = calculate_forest_area(image2_cv)
        deforestation_percentage = max(0, forest_percentage1 - forest_percentage2)
        
        contours1, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours1:
            if len(contour) > 10:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image1_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image1_cv, f'Class: {labels[predict1]} ({forest_percentage1:.2f}% Forest)', 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours2:
            if len(contour) > 10:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image2_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image2_cv, f'Class: {labels[predict2]} ({forest_percentage2:.2f}% Forest)', 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        image1_rgb = cv2.cvtColor(image1_cv, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(image2_cv, cv2.COLOR_BGR2RGB)
        
        st.subheader("Prediction Results")
        col3, col4 = st.columns(2)
        with col3:
            st.image(image1_rgb, caption="Processed Earlier Image", use_container_width=True)
        with col4:
            st.image(image2_rgb, caption="Processed Later Image", use_container_width=True)
        
        st.write(f"**Earlier Image Classified as:** {labels[predict1]} with {forest_percentage1:.2f}% Forest")
        st.write(f"**Later Image Classified as:** {labels[predict2]} with {forest_percentage2:.2f}% Forest")
        st.write(f"**Deforestation Percentage:** {deforestation_percentage:.2f}%")

def show_recent_news():
    st.title("📰 Recent News About Deforestation")
    news_articles = fetch_recent_news()
    
    if not news_articles:
        st.warning("No news articles available at the moment.")
        return
    
    for article in news_articles:
        st.markdown(f"""
        <div class="news-card">
            <div class="news-title">📰 {article['title']}</div>
            <div class="news-source"><strong>Source:</strong> {article['source']}</div>
            <div class="news-summary">{article['summary']}</div>
            <a href="{article['url']}" target="_blank" class="news-link">Read Full Article</a>
        </div>
        """, unsafe_allow_html=True)

def show_global_stats():
    st.title("📊 Global Deforestation Ratio")
    stats = fetch_global_stats()
    
    for key, value in stats.items():
        st.markdown(f"""
        <div class="stats-box">
            <h3>{key}</h3>
            <p>{value}</p>
        </div>
        """, unsafe_allow_html=True)

# Page Router
if st.session_state.page == 'Dashboard':
    show_dashboard()
elif st.session_state.page == 'Model Prediction':
    show_model_prediction()
elif st.session_state.page == 'Recent News':
    show_recent_news()
elif st.session_state.page == 'Global Stats':
    show_global_stats()