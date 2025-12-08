"""
==================================================================================
CANCER DIAGNOSTIC PREDICTION - STREAMLIT WEB APPLICATION
==================================================================================
Author: AI Assistant
Purpose: Interactive cancer prediction with file upload (CSV/Excel)
Models: Random Forest, Gradient Boosting, XGBoost (Top 3 Best Performers)
==================================================================================

INSTALLATION:
pip install streamlit pandas numpy scikit-learn xgboost openpyxl plotly

RUN:
streamlit run cancer_prediction_app.py
==================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Cancer Diagnostic Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stAlert {
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .cancer-risk {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False

# Title and header
st.title("🏥 Cancer Diagnostic Prediction System")
st.markdown("### AI-Powered Cancer Detection with Risk Assessment")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/dna.png", width=80)
    st.header("📋 Navigation")
    
    page = st.radio(
        "Select Page:",
        ["🏠 Home", "📤 Upload & Train", "🔮 Predict", "📊 Analytics"]
    )
    
    st.markdown("---")

# ==================================================================================
# HOME PAGE
# ==================================================================================

if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the Cancer Diagnostic Prediction System
        
        ### 🔬 What We Analyze:
        """)
        
        features_df = pd.DataFrame({
            'Biomarker': [
                'Ct_JAK2_V617F', 'Ct_microRNA_155', 'Ct_HLA_Tcell', 
                'Hemoglobin (Hb)', 'RBC Count', 'WBC Count', 
                'Platelet Count', 'Drug Resistance Index', 'Ct_JAK2'
            ],
            'Description': [
                'JAK2 V617F mutation detection',
                'MicroRNA-155 expression level',
                'HLA T-cell activity marker',
                'Blood oxygen capacity',
                'Red blood cell count',
                'White blood cell count',
                'Platelet count for clotting',
                'Treatment resistance measure',
                'Normalized JAK2 expression'
            ]
        })
        
        st.dataframe(features_df, use_container_width=True)
        
        st.markdown("""
        ### 🎯 Prediction Outputs:
        1. **Binary Classification**: Cancer vs Non-Cancer
        2. **Cancer Risk Probability**: 0-100% likelihood score
        3. **Risk Category**: Low, Moderate, High, Very High
        4. **Multi-class Diagnosis**: Specific cancer type identification
        """)
    
    with col2:
        st.markdown("### 📊 Risk Categories")
        risk_data = pd.DataFrame({
            'Category': ['Low', 'Moderate', 'High', 'Very High'],
            'Range': ['0-30%', '31-60%', '61-85%', '86-100%'],
            'Color': ['🟢', '🟡', '🟠', '🔴']
        })
        st.dataframe(risk_data, use_container_width=True)
    
    st.markdown("---")
    st.info("👈 **Get Started:** Select 'Upload & Train' from the sidebar to begin!")

# ==================================================================================
# UPLOAD & TRAIN PAGE
# ==================================================================================

elif page == "📤 Upload & Train":
    st.header("📤 Upload Training Data")
    
    st.markdown("""
    Upload your cancer patient dataset in **CSV** or **Excel** format. 
    The system will automatically train all three models and prepare them for predictions.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your cancer patient dataset with all required features"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
            
            # Display data preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", f"{df.shape[0]:,}")
            with col2:
                st.metric("Features", df.shape[1])
            with col3:
                st.metric("Diagnoses", df['Diagnosis'].nunique())
            with col4:
                missing = df.isnull().sum().sum()
                st.metric("Missing Values", missing)
            
            # Show diagnosis distribution
            st.markdown("### 📊 Diagnosis Distribution")
            diagnosis_counts = df['Diagnosis'].value_counts()
            
            fig = px.pie(
                values=diagnosis_counts.values,
                names=diagnosis_counts.index,
                title="Patient Diagnosis Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Train models button
            st.markdown("---")
            if st.button("🚀 Train Models", type="primary", use_container_width=True):
                with st.spinner("🔄 Training models... This may take a few moments..."):
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Prepare data
                    status_text.text("📊 Preprocessing data...")
                    progress_bar.progress(10)
                    
                    numeric_features = [
                        'Ct_JAK2_V617F', 'Ct_microRNA_155', 'Ct_HLA_Tcell', 
                        'Hb', 'RBC', 'WBC', 'Platelet', 
                        'Drug_Resistance_Index', 'Ct_JAK2'
                    ]
                    
                    X = df[numeric_features].copy()
                    
                    # Binary classification
                    y_binary = df['Diagnosis'].apply(
                        lambda x: 'Cancer' if x in ['Leukemia_Positive', 'PV_Positive'] else 'Non-Cancer'
                    )
                    y_binary_encoded = (y_binary == 'Cancer').astype(int)
                    
                    # Multi-class classification
                    y_multi = df['Diagnosis'].copy()
                    
                    # Train-test split
                    X_train, X_test, y_train_bin, y_test_bin = train_test_split(
                        X, y_binary_encoded, test_size=0.2, random_state=42, stratify=y_binary_encoded
                    )
                    
                    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
                        X, y_multi, test_size=0.2, random_state=42, stratify=y_multi
                    )
                    
                    progress_bar.progress(20)
                    
                    # Scaling
                    status_text.text("⚖️ Scaling features...")
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    X_train_multi_scaled = scaler.fit_transform(X_train_multi)
                    X_test_multi_scaled = scaler.transform(X_test_multi)
                    
                    progress_bar.progress(30)
                    
                    # SMOTE
                    status_text.text("🔄 Balancing classes with SMOTE...")
                    smote = SMOTE(random_state=42)
                    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_bin)
                    
                    progress_bar.progress(40)
                    
                    # Train Model 1: Random Forest
                    status_text.text("🌲 Training Random Forest...")
                    rf_model = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=20,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        class_weight='balanced',
                        n_jobs=-1
                    )
                    rf_model.fit(X_train_balanced, y_train_balanced)
                    rf_accuracy = rf_model.score(X_test_scaled, y_test_bin)
                    
                    progress_bar.progress(60)
                    
                    # Train Model 2: Gradient Boosting
                    status_text.text("⚡ Training Gradient Boosting...")
                    gb_model = GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.1,
                        max_depth=10,
                        random_state=42,
                        subsample=0.8
                    )
                    gb_model.fit(X_train_balanced, y_train_balanced)
                    gb_accuracy = gb_model.score(X_test_scaled, y_test_bin)
                    
                    progress_bar.progress(80)
                    
                    # Train Model 3: XGBoost
                    status_text.text("🚀 Training XGBoost...")
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=10,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        eval_metric='logloss',
                        use_label_encoder=False
                    )
                    xgb_model.fit(X_train_balanced, y_train_balanced)
                    xgb_accuracy = xgb_model.score(X_test_scaled, y_test_bin)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Training completed!")
                    
                    # Store in session state
                    st.session_state.rf_model = rf_model
                    st.session_state.gb_model = gb_model
                    st.session_state.xgb_model = xgb_model
                    st.session_state.scaler = scaler
                    st.session_state.numeric_features = numeric_features
                    st.session_state.models_trained = True
                    st.session_state.training_data = df
                    
                    # Show results
                    st.success("🎉 All models trained successfully!")
                    
                    st.markdown("### 📊 Model Performance")
                    results_df = pd.DataFrame({
                        'Model': ['Random Forest', 'Gradient Boosting', 'XGBoost'],
                        'Accuracy': [rf_accuracy, gb_accuracy, xgb_accuracy],
                        'Status': ['✅ Ready', '✅ Ready', '✅ Ready']
                    })
                    results_df['Accuracy'] = results_df['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
                    st.dataframe(results_df, use_container_width=True)
                    
                    st.balloons()
                    
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Please ensure your file has all required columns.")

# ==================================================================================
# PREDICT PAGE
# ==================================================================================

elif page == "🔮 Predict":
    st.header("🔮 Cancer Risk Prediction")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please upload and train models first in the 'Upload & Train' page!")
        st.stop()
    
    st.markdown("""
    Upload a CSV or Excel file with patient data to get cancer predictions and risk assessments.
    """)
    
    # File uploader for prediction
    predict_file = st.file_uploader(
        "Upload patient data for prediction",
        type=['csv', 'xlsx', 'xls'],
        key="predict_uploader"
    )
    
    if predict_file is not None:
        try:
            # Read file
            if predict_file.name.endswith('.csv'):
                predict_df = pd.read_csv(predict_file)
            else:
                predict_df = pd.read_excel(predict_file)
            
            st.success(f"✅ Loaded {len(predict_df)} patient records")
            
            # Show preview
            with st.expander("📋 View Data Preview"):
                st.dataframe(predict_df.head(), use_container_width=True)
            
            # Predict button
            if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing patient data..."):
                    
                    # Extract features
                    X_predict = predict_df[st.session_state.numeric_features].copy()
                    X_predict_scaled = st.session_state.scaler.transform(X_predict)
                    
                    # Get predictions from all 3 models
                    rf_proba = st.session_state.rf_model.predict_proba(X_predict_scaled)[:, 1]
                    gb_proba = st.session_state.gb_model.predict_proba(X_predict_scaled)[:, 1]
                    xgb_proba = st.session_state.xgb_model.predict_proba(X_predict_scaled)[:, 1]
                    
                    # Ensemble prediction (average)
                    ensemble_proba = (rf_proba + gb_proba + xgb_proba) / 3
                    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
                    
                    # Risk categories
                    def categorize_risk(prob):
                        if prob < 0.30:
                            return 'Low Risk', '🟢'
                        elif prob < 0.60:
                            return 'Moderate Risk', '🟡'
                        elif prob < 0.85:
                            return 'High Risk', '🟠'
                        else:
                            return 'Very High Risk', '🔴'
                    
                    risk_categories = [categorize_risk(p) for p in ensemble_proba]
                    
                    # Create results dataframe
                    results_df = predict_df.copy()
                    results_df['Cancer_Prediction'] = ['Cancer' if p == 1 else 'Non-Cancer' for p in ensemble_pred]
                    results_df['Cancer_Probability_%'] = (ensemble_proba * 100).round(2)
                    results_df['Risk_Category'] = [r[0] for r in risk_categories]
                    results_df['Risk_Icon'] = [r[1] for r in risk_categories]
                    results_df['RF_Probability_%'] = (rf_proba * 100).round(2)
                    results_df['GB_Probability_%'] = (gb_proba * 100).round(2)
                    results_df['XGB_Probability_%'] = (xgb_proba * 100).round(2)
                    
                    st.session_state.results_df = results_df
                    st.session_state.predictions_made = True
                    
                    st.success("✅ Predictions completed!")
                    
                    # Summary metrics
                    st.markdown("### 📊 Prediction Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        cancer_count = (ensemble_pred == 1).sum()
                        st.metric("Cancer Cases", cancer_count, 
                                 f"{cancer_count/len(ensemble_pred)*100:.1f}%")
                    
                    with col2:
                        non_cancer_count = (ensemble_pred == 0).sum()
                        st.metric("Non-Cancer", non_cancer_count,
                                 f"{non_cancer_count/len(ensemble_pred)*100:.1f}%")
                    
                    with col3:
                        avg_risk = ensemble_proba.mean() * 100
                        st.metric("Avg Risk Score", f"{avg_risk:.1f}%")
                    
                    with col4:
                        high_risk_count = sum(1 for p in ensemble_proba if p >= 0.60)
                        st.metric("High Risk+", high_risk_count,
                                 f"{high_risk_count/len(ensemble_pred)*100:.1f}%")
                    
                    # Risk distribution
                    st.markdown("### 🎯 Risk Distribution")
                    risk_counts = pd.Series([r[0] for r in risk_categories]).value_counts()
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=risk_counts.index,
                            y=risk_counts.values,
                            marker_color=['#27ae60', '#f39c12', '#e67e22', '#e74c3c'],
                            text=risk_counts.values,
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="Distribution of Risk Categories",
                        xaxis_title="Risk Category",
                        yaxis_title="Number of Patients",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.markdown("### 📋 Detailed Results")
                    
                    # Display options
                    display_cols = st.multiselect(
                        "Select columns to display:",
                        options=results_df.columns.tolist(),
                        default=['Sample_ID', 'Cancer_Prediction', 'Cancer_Probability_%', 
                                'Risk_Category', 'Risk_Icon'] if 'Sample_ID' in results_df.columns 
                                else ['Cancer_Prediction', 'Cancer_Probability_%', 'Risk_Category', 'Risk_Icon']
                    )
                    
                    if display_cols:
                        st.dataframe(results_df[display_cols], use_container_width=True, height=400)
                    
                    # Download results
                    st.markdown("### 💾 Download Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name="cancer_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Filter high risk patients
                        high_risk_df = results_df[results_df['Cancer_Probability_%'] >= 60]
                        high_risk_csv = high_risk_df.to_csv(index=False)
                        st.download_button(
                            label="⚠️ Download High Risk Only",
                            data=high_risk_csv,
                            file_name="high_risk_patients.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please ensure your file has all required features.")

# ==================================================================================
# ANALYTICS PAGE
# ==================================================================================

elif page == "📊 Analytics":
    st.header("📊 Advanced Analytics")
    
    if not st.session_state.predictions_made:
        st.warning("⚠️ Please make predictions first in the 'Predict' page!")
        st.stop()
    
    results_df = st.session_state.results_df
    
    # Tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔬 Feature Analysis", "🎯 Risk Analysis", "🤖 Model Comparison"])
    
    with tab1:
        st.markdown("### 📈 Prediction Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cancer vs Non-Cancer pie chart
            pred_counts = results_df['Cancer_Prediction'].value_counts()
            fig = px.pie(
                values=pred_counts.values,
                names=pred_counts.index,
                title="Cancer vs Non-Cancer Distribution",
                color_discrete_map={'Cancer': '#e74c3c', 'Non-Cancer': '#27ae60'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk category distribution
            risk_counts = results_df['Risk_Category'].value_counts()
            fig = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="Risk Category Distribution",
                labels={'x': 'Risk Category', 'y': 'Count'},
                color=risk_counts.index,
                color_discrete_map={
                    'Low Risk': '#27ae60',
                    'Moderate Risk': '#f39c12',
                    'High Risk': '#e67e22',
                    'Very High Risk': '#e74c3c'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Probability distribution
        fig = px.histogram(
            results_df,
            x='Cancer_Probability_%',
            nbins=50,
            title="Cancer Probability Distribution",
            labels={'Cancer_Probability_%': 'Cancer Probability (%)'},
            color_discrete_sequence=['#3498db']
        )
        fig.add_vline(x=30, line_dash="dash", line_color="green", annotation_text="Low Risk")
        fig.add_vline(x=60, line_dash="dash", line_color="orange", annotation_text="Moderate Risk")
        fig.add_vline(x=85, line_dash="dash", line_color="red", annotation_text="High Risk")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🔬 Feature Analysis")
        
        if st.session_state.models_trained:
            # Feature importance from Random Forest
            feature_importance = pd.DataFrame({
                'Feature': st.session_state.numeric_features,
                'Importance': st.session_state.rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance (Random Forest)",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature correlations with cancer probability
            st.markdown("### 📊 Feature Correlations")
            feature_cols = st.session_state.numeric_features
            corr_data = []
            for feature in feature_cols:
                corr = results_df[feature].corr(results_df['Cancer_Probability_%'])
                corr_data.append({'Feature': feature, 'Correlation': corr})
            
            corr_df = pd.DataFrame(corr_data).sort_values('Correlation', ascending=False)
            
            fig = px.bar(
                corr_df,
                x='Correlation',
                y='Feature',
                orientation='h',
                title="Feature Correlation with Cancer Probability",
                color='Correlation',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🎯 Risk Analysis")
        
        # Risk threshold analysis
        thresholds = [30, 60, 85]
        threshold_counts = []
        for t in thresholds:
            count = (results_df['Cancer_Probability_%'] >= t).sum()
            threshold_counts.append(count)
        
        fig = go.Figure(data=[
            go.Bar(
                x=[f"≥{t}%" for t in thresholds],
                y=threshold_counts,
                marker_color=['#f39c12', '#e67e22', '#e74c3c'],
                text=threshold_counts,
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Patients Above Risk Thresholds",
            xaxis_title="Risk Threshold",
            yaxis_title="Number of Patients"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top high-risk patients
        st.markdown("### ⚠️ Top 10 Highest Risk Patients")
        high_risk_cols = ['Sample_ID', 'Cancer_Probability_%', 'Risk_Category'] if 'Sample_ID' in results_df.columns else ['Cancer_Probability_%', 'Risk_Category']
        top_risk = results_df.nlargest(10, 'Cancer_Probability_%')[high_risk_cols]
        st.dataframe(top_risk, use_container_width=True)
    
    with tab4:
        st.markdown("### 🤖 Model Comparison")
        
        # Compare model probabilities
        model_cols = ['RF_Probability_%', 'GB_Probability_%', 'XGB_Probability_%']
        model_avg = results_df[model_cols].mean()
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Random Forest', 'Gradient Boosting', 'XGBoost'],
                y=model_avg.values,
                marker_color=['#3498db', '#9b59b6', '#e67e22'],
                text=[f"{v:.2f}%" for v in model_avg.values],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Average Predicted Cancer Probability by Model",
            yaxis_title="Average Probability (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model agreement analysis
        st.markdown("### 🤝 Model Agreement")
        
        # Calculate standard deviation across models
        results_df['Model_StdDev'] = results_df[model_cols].std(axis=1)
        
        fig = px.histogram(
            results_df,
            x='Model_StdDev',
            nbins=30,
            title="Model Prediction Variability",
            labels={'Model_StdDev': 'Standard Deviation (%)'},
            color_discrete_sequence=['#16a085']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        avg_stddev = results_df['Model_StdDev'].mean()
        st.info(f"📊 Average Model Agreement: {avg_stddev:.2f}% standard deviation")
        
        if avg_stddev < 5:
            st.success("✅ Excellent model consensus (low variability)")
        elif avg_stddev < 10:
            st.warning("⚠️ Moderate model disagreement")
        else:
            st.error("❌ High model variability - review predictions carefully")



# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p><strong>Cancer Diagnostic Prediction System v1.0</strong></p>
    <p>Powered by Advanced Machine Learning | Built with ❤️ for Healthcare</p>
    <p style='font-size: 12px;'>⚠️ For research and clinical decision support only. Not a substitute for professional medical advice.</p>
</div>
""", unsafe_allow_html=True)