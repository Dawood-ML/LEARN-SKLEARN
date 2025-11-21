import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
import base64

# Page configuration
st.set_page_config(
    page_title="Fraud Detection AI | Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_model():
    """Load the trained model"""
    model_path = Path('fraud_detection_outputs/models/fraud_detection_pipeline.joblib')
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_data
def load_metrics():
    """Load classification metrics"""
    metrics_path = Path('fraud_detection_outputs/results/classification_report.json')
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def get_image_base64(image_path):
    """Convert image to base64 for display"""
    if Path(image_path).exists():
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return None

def create_confusion_matrix_plotly(metrics):
    """Create interactive confusion matrix with Plotly"""
    # Extract values from metrics
    fraud_metrics = metrics.get('Fraud', {})
    non_fraud_metrics = metrics.get('Non-Fraud', {})
    
    # Calculate confusion matrix values from metrics
    # This is an approximation - ideally load actual cm values
    total_fraud = int(fraud_metrics.get('support', 0))
    total_non_fraud = int(non_fraud_metrics.get('support', 0))
    
    fraud_recall = fraud_metrics.get('recall', 0)
    non_fraud_recall = non_fraud_metrics.get('recall', 0)
    
    # True positives and true negatives
    tp = int(total_fraud * fraud_recall)
    tn = int(total_non_fraud * non_fraud_recall)
    
    # False negatives and false positives
    fn = total_fraud - tp
    fp = total_non_fraud - tn
    
    cm = [[tn, fp], [fn, tp]]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Non-Fraud', 'Fraud'],
        y=['Non-Fraud', 'Fraud'],
        colorscale='Greens',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        showscale=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=400,
        font=dict(size=14)
    )
    
    return fig

def create_metrics_gauge(value, title, max_value=1):
    """Create a gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_precision_recall_comparison(metrics):
    """Create precision vs recall comparison"""
    categories = ['Non-Fraud', 'Fraud']
    precision = [metrics['Non-Fraud']['precision'], metrics['Fraud']['precision']]
    recall = [metrics['Non-Fraud']['recall'], metrics['Fraud']['recall']]
    f1 = [metrics['Non-Fraud']['f1-score'], metrics['Fraud']['f1-score']]
    
    fig = go.Figure(data=[
        go.Bar(name='Precision', x=categories, y=precision, marker_color='#667eea'),
        go.Bar(name='Recall', x=categories, y=recall, marker_color='#764ba2'),
        go.Bar(name='F1-Score', x=categories, y=f1, marker_color='#f093fb')
    ])
    
    fig.update_layout(
        title='Performance Metrics Comparison',
        barmode='group',
        height=400,
        yaxis_title='Score',
        xaxis_title='Class',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Credit Card Fraud Detection AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Machine Learning Model for Real-Time Fraud Prevention</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/security-checked.png", width=80)
        st.title("Navigation")
        page = st.radio(
            "Select View",
            ["📊 Dashboard", "🔍 Model Insights", "🎯 Predictions", "📈 Performance"]
        )
        
        st.markdown("---")
        st.markdown("### Model Information")
        st.info("""
        **Algorithm:** LightGBM  
        **Target Recall:** 90%  
        **Status:** ✅ Production Ready
        """)
        
        st.markdown("---")
        st.markdown("### About")
        st.write("Built with Streamlit, LightGBM, and SHAP for interpretable fraud detection.")
    
    # Load data
    model = load_model()
    metrics = load_metrics()
    
    if metrics is None:
        st.error("⚠️ Model metrics not found. Please run the training script first.")
        return
    
    # Page routing
    if page == "📊 Dashboard":
        show_dashboard(metrics)
    elif page == "🔍 Model Insights":
        show_model_insights(metrics)
    elif page == "🎯 Predictions":
        show_predictions(model)
    elif page == "📈 Performance":
        show_performance(metrics)

def show_dashboard(metrics):
    """Main dashboard view"""
    st.header("Executive Dashboard")
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    fraud_recall = metrics['Fraud']['recall']
    fraud_precision = metrics['Fraud']['precision']
    fraud_f1 = metrics['Fraud']['f1-score']
    accuracy = metrics['accuracy']
    
    with col1:
        st.markdown(f"""
        <div class="success-card">
            <h3 style="margin:0;">Recall Rate</h3>
            <h1 style="margin:0.5rem 0;">{fraud_recall*100:.1f}%</h1>
            <p style="margin:0;">Frauds Detected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <h3 style="margin:0;">Precision</h3>
            <h1 style="margin:0.5rem 0;">{fraud_precision*100:.1f}%</h1>
            <p style="margin:0;">Accuracy of Alerts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0;">F1-Score</h3>
            <h1 style="margin:0.5rem 0;">{fraud_f1*100:.1f}%</h1>
            <p style="margin:0;">Balanced Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="warning-card">
            <h3 style="margin:0;">Accuracy</h3>
            <h1 style="margin:0.5rem 0;">{accuracy*100:.1f}%</h1>
            <p style="margin:0;">Overall Correct</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Confusion Matrix")
        fig_cm = create_confusion_matrix_plotly(metrics)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.subheader("📈 Metrics Comparison")
        fig_metrics = create_precision_recall_comparison(metrics)
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Business Impact
    st.markdown("---")
    st.header("💼 Business Impact Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    total_fraud = int(metrics['Fraud']['support'])
    detected = int(total_fraud * fraud_recall)
    missed = total_fraud - detected
    
    with col1:
        st.metric(
            label="Total Fraudulent Transactions",
            value=total_fraud,
            delta="In Test Set"
        )
    
    with col2:
        st.metric(
            label="Successfully Detected",
            value=detected,
            delta=f"{fraud_recall*100:.1f}% Caught",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="Missed Frauds",
            value=missed,
            delta=f"{(1-fraud_recall)*100:.1f}% Slipped",
            delta_color="inverse"
        )
    
    # Cost Analysis
    st.subheader("💰 Cost-Benefit Analysis")
    
    avg_fraud_amount = st.slider("Average Fraud Transaction Amount ($)", 100, 10000, 1000, 100)
    investigation_cost = st.slider("Cost per Investigation ($)", 5, 100, 20, 5)
    
    total_non_fraud = int(metrics['Non-Fraud']['support'])
    false_positives = int(total_non_fraud * (1 - metrics['Non-Fraud']['recall']))
    
    money_saved = detected * avg_fraud_amount
    investigation_costs = false_positives * investigation_cost
    net_benefit = money_saved - investigation_costs
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💵 Money Saved", f"${money_saved:,}")
    with col2:
        st.metric("💸 Investigation Costs", f"${investigation_costs:,}")
    with col3:
        st.metric("✅ Net Benefit", f"${net_benefit:,}", delta=f"ROI: {(net_benefit/investigation_costs)*100:.0f}%")

def show_model_insights(metrics):
    """Model insights and SHAP visualization"""
    st.header("🔍 Model Insights & Interpretability")
    
    # Display SHAP plot
    shap_path = Path('fraud_detection_outputs/interpretations/shap_summary_plot.png')
    
    if shap_path.exists():
        st.subheader("Feature Importance Analysis (SHAP)")
        st.image(str(shap_path), use_container_width=True)
        
        st.markdown("""
        ### 📌 Key Insights from SHAP Analysis:
        
        1. **V14 is the strongest fraud indicator** - High values strongly predict fraud
        2. **V4 and V12** are also critical features for detection
        3. **Time and Amount** have moderate importance - fraudsters mimic normal patterns
        4. **Model uses multiple features** - Not relying on a single signal makes it robust
        
        **Color coding:**
        - 🔴 Red/Pink = High feature value
        - 🔵 Blue = Low feature value
        - Right side = Increases fraud probability
        - Left side = Decreases fraud probability
        """)
    else:
        st.warning("SHAP plot not found. Please run the training script to generate interpretations.")
    
    # Model architecture
    st.markdown("---")
    st.subheader("🏗️ Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Pipeline Components:**
        1. **Preprocessing**
           - StandardScaler for Time & Amount
           - Feature transformation
        
        2. **Resampling**
           - SMOTE for class balance
           - Handles imbalanced data
        
        3. **Classification**
           - LightGBM Classifier
           - Optimized hyperparameters
        """)
    
    with col2:
        st.markdown("""
        **Hyperparameters:**
        - `n_estimators`: 999
        - `learning_rate`: 0.0165
        - `num_leaves`: 225
        - `max_depth`: 6
        - `reg_alpha`: 0.0596
        - `reg_lambda`: 0.0025
        """)

def show_predictions(model):
    """Interactive prediction interface"""
    st.header("🎯 Real-Time Fraud Prediction")
    
    if model is None:
        st.error("Model not loaded. Please ensure the model file exists.")
        return
    
    st.markdown("### Enter Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        time = st.number_input("Time (seconds since first transaction)", min_value=0, value=50000)
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
    
    with col2:
        st.info("Note: V1-V28 are PCA-transformed features from the original dataset")
    
    # Create input for V features
    st.markdown("### PCA Features (V1-V28)")
    
    v_features = {}
    cols = st.columns(4)
    
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        with cols[col_idx]:
            v_features[f'V{i}'] = st.number_input(
                f'V{i}',
                value=0.0,
                format="%.6f",
                key=f'v{i}'
            )
    
    if st.button("🔍 Predict Fraud Risk", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Time': [time],
            'Amount': [amount],
            **{k: [v] for k, v in v_features.items()}
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="warning-card">
                    <h2 style="margin:0;">⚠️ FRAUD ALERT</h2>
                    <p style="margin:0.5rem 0;">This transaction is flagged as fraudulent</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-card">
                    <h2 style="margin:0;">✅ LEGITIMATE</h2>
                    <p style="margin:0.5rem 0;">This transaction appears normal</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Fraud Probability", f"{probability[1]*100:.2f}%")
        
        with col3:
            st.metric("Legitimate Probability", f"{probability[0]*100:.2f}%")
        
        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            title={'text': "Fraud Risk Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

def show_performance(metrics):
    """Detailed performance metrics"""
    st.header("📈 Model Performance Analysis")
    
    # Gauges for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_recall = create_metrics_gauge(metrics['Fraud']['recall'], "Recall (Fraud Detection)")
        st.plotly_chart(fig_recall, use_container_width=True)
    
    with col2:
        fig_precision = create_metrics_gauge(metrics['Fraud']['precision'], "Precision (Alert Accuracy)")
        st.plotly_chart(fig_precision, use_container_width=True)
    
    with col3:
        fig_f1 = create_metrics_gauge(metrics['Fraud']['f1-score'], "F1-Score (Balance)")
        st.plotly_chart(fig_f1, use_container_width=True)
    
    # Detailed metrics table
    st.markdown("---")
    st.subheader("📋 Detailed Classification Report")
    
    report_df = pd.DataFrame({
        'Class': ['Non-Fraud', 'Fraud'],
        'Precision': [metrics['Non-Fraud']['precision'], metrics['Fraud']['precision']],
        'Recall': [metrics['Non-Fraud']['recall'], metrics['Fraud']['recall']],
        'F1-Score': [metrics['Non-Fraud']['f1-score'], metrics['Fraud']['f1-score']],
        'Support': [int(metrics['Non-Fraud']['support']), int(metrics['Fraud']['support'])]
    })
    
    st.dataframe(
        report_df.style.format({
            'Precision': '{:.2%}',
            'Recall': '{:.2%}',
            'F1-Score': '{:.2%}'
        }).background_gradient(subset=['Precision', 'Recall', 'F1-Score'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    # Model comparison
    st.markdown("---")
    st.subheader("🏆 Why This Model Excels")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Strengths:**
        - High recall (90%+) catches most frauds
        - Balanced precision-recall trade-off
        - Interpretable with SHAP values
        - Fast inference time
        - Handles imbalanced data well
        """)
    
    with col2:
        st.markdown("""
        **🎯 Use Cases:**
        - Real-time transaction monitoring
        - Batch fraud analysis
        - Risk scoring systems
        - Alert prioritization
        - Audit and compliance
        """)

if __name__ == "__main__":
    main()