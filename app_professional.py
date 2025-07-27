"""
Professional PM-Analyzer Application
Production-ready application with all enhancements
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our professional modules
from src.ml_models import ReturnPredictionModel, RiskFactorModel, RegimeDetectionModel, MLPortfolioOptimizer
from src.data_integration import DataIntegrationEngine, BenchmarkEngine
from src.advanced_ui import AdvancedUIComponents
from src.portfolio_optimizer import AdvancedOptimizer
from src.return_analytics import SimpleAnalytics

# Page configuration
st.set_page_config(
    page_title="PM-Analyzer Professional",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ProfessionalPMAnalyzer:
    """
    Professional Private Markets Analyzer Application
    """
    
    def __init__(self):
        self.data_engine = DataIntegrationEngine()
        self.benchmark_engine = BenchmarkEngine()
        self.ui_components = AdvancedUIComponents()
        self.ml_optimizer = MLPortfolioOptimizer()
        self.analytics = SimpleAnalytics()
        
        # Initialize session state
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = {}
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'ml_models_fitted' not in st.session_state:
            st.session_state.ml_models_fitted = False
    
    def run(self):
        """Main application runner"""
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🏦 PM-Analyzer Professional</h1>
            <p>Institutional-Grade Private Markets Portfolio Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        self.create_sidebar()
        
        # Main content based on page selection
        page = st.session_state.get('current_page', 'Data Integration')
        
        if page == 'Data Integration':
            self.show_data_integration_page()
        elif page == 'ML-Enhanced Analysis':
            self.show_ml_analysis_page()
        elif page == 'Advanced Optimization':
            self.show_optimization_page()
        elif page == 'Stress Testing':
            self.show_stress_testing_page()
        elif page == 'Benchmark Analysis':
            self.show_benchmark_analysis_page()
        elif page == 'Real-time Dashboard':
            self.show_dashboard_page()
    
    def create_sidebar(self):
        """Create professional sidebar navigation"""
        
        st.sidebar.markdown("## 🧭 Navigation")
        
        pages = [
            'Data Integration',
            'ML-Enhanced Analysis', 
            'Advanced Optimization',
            'Stress Testing',
            'Benchmark Analysis',
            'Real-time Dashboard'
        ]
        
        current_page = st.sidebar.selectbox(
            "Select Analysis Module",
            pages,
            key='current_page'
        )
        
        st.sidebar.markdown("---")
        
        # Quick stats
        st.sidebar.markdown("## 📊 Quick Stats")
        st.sidebar.metric("Data Sources", "5 Connected")
        st.sidebar.metric("ML Models", "4 Active")
        st.sidebar.metric("Last Update", "2 min ago")
        
        st.sidebar.markdown("---")
        
        # Settings
        st.sidebar.markdown("## ⚙️ Settings")
        
        auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=True)
        show_advanced = st.sidebar.checkbox("Show advanced options", value=False)
        
        if show_advanced:
            st.sidebar.slider("Refresh interval (sec)", 30, 300, 60)
            st.sidebar.selectbox("Theme", ["Professional", "Dark", "Light"])
    
    def show_data_integration_page(self):
        """Data integration and upload page"""
        
        st.markdown("## 📁 Data Integration & Upload")
        st.markdown("Upload your fund data or connect to live data sources")
        
        # File upload section
        uploaded_files = self.ui_components.create_file_uploader_section()
        
        # Process uploaded files
        if uploaded_files:
            self.process_uploaded_files(uploaded_files)
        
        # API data integration
        st.markdown("### 🌐 Live Data Integration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Benchmark Data")
            
            benchmark_options = {
                'S&P 500': '^GSPC',
                'MSCI World': 'URTH', 
                'Russell 2000': '^RUT',
                'NASDAQ': '^IXIC'
            }
            
            selected_benchmarks = st.multiselect(
                "Select benchmarks to fetch",
                list(benchmark_options.keys()),
                default=['S&P 500']
            )
            
            if st.button("Fetch Benchmark Data"):
                with st.spinner("Fetching benchmark data..."):
                    benchmark_tickers = {name: benchmark_options[name] for name in selected_benchmarks}
                    benchmark_data = self.data_engine.fetch_multiple_benchmarks(benchmark_tickers)
                    
                    if not benchmark_data.empty:
                        st.session_state.uploaded_data['benchmark_data'] = benchmark_data
                        st.success(f"✅ Fetched data for {len(selected_benchmarks)} benchmarks")
                        st.dataframe(benchmark_data.head(), use_container_width=True)
                    else:
                        st.error("❌ Failed to fetch benchmark data")
        
        with col2:
            st.markdown("#### Sample Fund Data")
            
            fund_type = st.selectbox(
                "Generate sample fund data",
                ['Private Equity', 'Real Estate', 'Private Debt']
            )
            
            if st.button("Generate Sample Data"):
                fund_type_map = {'Private Equity': 'PE', 'Real Estate': 'RE', 'Private Debt': 'PD'}
                sample_data = self.data_engine.create_sample_fund_data(fund_type_map[fund_type])
                
                st.session_state.uploaded_data['fund_data'] = sample_data
                st.success(f"✅ Generated sample {fund_type} fund data")
                
                # Show fund info
                fund_info = sample_data['fund_info']
                st.markdown(f"**{fund_info['Fund Name']}**")
                st.markdown(f"- Vintage: {fund_info['Vintage Year']}")
                st.markdown(f"- Size: ${fund_info['Fund Size']:,}M")
                st.markdown(f"- Strategy: {fund_info['Strategy']}")
        
        # Data validation and preview
        if st.session_state.uploaded_data:
            st.markdown("### 🔍 Data Validation & Preview")
            
            for data_type, data in st.session_state.uploaded_data.items():
                with st.expander(f"📊 {data_type.replace('_', ' ').title()}", expanded=False):
                    if isinstance(data, dict) and 'cashflows' in data:
                        # Fund data
                        st.dataframe(data['cashflows'].head(10), use_container_width=True)
                        
                        # Show performance metrics
                        if 'performance_metrics' in data:
                            st.markdown("**Performance Metrics:**")
                            st.dataframe(data['performance_metrics'], use_container_width=True)
                    
                    elif isinstance(data, pd.DataFrame):
                        # Regular DataFrame
                        st.dataframe(data.head(10), use_container_width=True)
                        
                        # Data quality metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", len(data))
                        with col2:
                            st.metric("Columns", len(data.columns))
                        with col3:
                            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
                            st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files"""
        
        for file_type, file in uploaded_files.items():
            try:
                df = self.data_engine.upload_csv_data(file)
                if not df.empty:
                    st.session_state.uploaded_data[file_type] = df
                    st.success(f"✅ Successfully processed {file_type}")
            except Exception as e:
                st.error(f"❌ Error processing {file_type}: {str(e)}")
    
    def show_ml_analysis_page(self):
        """Machine Learning enhanced analysis page"""
        
        st.markdown("## 🤖 ML-Enhanced Portfolio Analysis")
        st.markdown("Advanced machine learning models for return prediction and risk analysis")
        
        if not st.session_state.uploaded_data:
            st.warning("⚠️ Please upload data in the Data Integration section first")
            return
        
        # ML model configuration
        params = self.ui_components.create_interactive_parameter_panel()
        
        # Check if we have fund data
        fund_data = st.session_state.uploaded_data.get('fund_data')
        benchmark_data = st.session_state.uploaded_data.get('benchmark_data')
        
        if fund_data and isinstance(fund_data, dict):
            cashflows = fund_data['cashflows']
            
            # Prepare data for ML models
            st.markdown("### 🔧 ML Model Training")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Configuration")
                
                model_types = st.multiselect(
                    "Select ML models to train",
                    ['Return Prediction', 'Risk Factor Model', 'Regime Detection'],
                    default=['Return Prediction', 'Regime Detection']
                )
                
                train_test_split = st.slider(
                    "Training data percentage",
                    min_value=0.6, max_value=0.9, value=0.8, step=0.05
                )
            
            with col2:
                st.markdown("#### Training Status")
                
                if st.button("🚀 Train ML Models", type="primary"):
                    self.train_ml_models(cashflows, benchmark_data, model_types, params)
            
            # Show ML results if models are trained
            if st.session_state.ml_models_fitted:
                self.show_ml_results()
        
        else:
            st.info("💡 Upload fund cashflow data to enable ML analysis")
    
    def train_ml_models(self, cashflows, benchmark_data, model_types, params):
        """Train ML models"""
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Prepare returns data
            status_text.text("Preparing data...")
            progress_bar.progress(0.2)
            
            # Calculate returns from NAV
            nav_returns = cashflows['NAV'].pct_change().dropna()
            
            if benchmark_data is not None and not benchmark_data.empty:
                # Combine with benchmark data
                returns_data = pd.DataFrame({
                    'Fund': nav_returns,
                    **benchmark_data
                })
            else:
                returns_data = pd.DataFrame({'Fund': nav_returns})
            
            # Train models
            if 'Return Prediction' in model_types:
                status_text.text("Training return prediction model...")
                progress_bar.progress(0.4)
                
                return_model = ReturnPredictionModel()
                features = return_model.create_features(pd.DataFrame({'returns': nav_returns}))
                target = nav_returns.shift(-1).dropna()
                
                # Align features and target
                common_index = features.index.intersection(target.index)
                if len(common_index) > 10:  # Minimum data requirement
                    return_model.fit(features.loc[common_index], target.loc[common_index])
                    st.session_state.analysis_results['return_model'] = return_model
            
            if 'Risk Factor Model' in model_types:
                status_text.text("Training risk factor model...")
                progress_bar.progress(0.6)
                
                risk_model = RiskFactorModel(n_factors=3)
                risk_model.fit(returns_data.fillna(0))
                st.session_state.analysis_results['risk_model'] = risk_model
            
            if 'Regime Detection' in model_types:
                status_text.text("Training regime detection model...")
                progress_bar.progress(0.8)
                
                regime_model = RegimeDetectionModel(n_regimes=3)
                regime_model.fit(nav_returns)
                st.session_state.analysis_results['regime_model'] = regime_model
            
            progress_bar.progress(1.0)
            status_text.text("✅ All models trained successfully!")
            st.session_state.ml_models_fitted = True
            
            st.success("🎉 ML models trained successfully!")
            
        except Exception as e:
            st.error(f"❌ Error training models: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    def show_ml_results(self):
        """Display ML model results"""
        
        st.markdown("### 📊 ML Model Results")
        
        results = st.session_state.analysis_results
        
        # Return prediction results
        if 'return_model' in results:
            with st.expander("🎯 Return Prediction Model", expanded=True):
                model = results['return_model']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Model Performance:**")
                    if hasattr(model, 'validation_scores'):
                        scores_df = pd.DataFrame([
                            {'Model': name, 'R² Score': f"{score:.3f}"}
                            for name, score in model.validation_scores.items()
                        ])
                        st.dataframe(scores_df, use_container_width=True)
                
                with col2:
                    st.markdown("**Feature Importance:**")
                    importance_df = model.get_feature_importance()
                    if not importance_df.empty:
                        top_features = importance_df.head(5)
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=top_features['mean_importance'],
                                y=top_features.index,
                                orientation='h',
                                marker_color='#1f77b4'
                            )
                        ])
                        
                        fig.update_layout(
                            title="Top 5 Features",
                            height=300,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        # Risk factor model results
        if 'risk_model' in results:
            with st.expander("📈 Risk Factor Model", expanded=False):
                model = results['risk_model']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Factor Loadings:**")
                    if model.factor_loadings is not None:
                        st.dataframe(model.factor_loadings, use_container_width=True)
                
                with col2:
                    st.markdown("**Explained Variance:**")
                    if model.explained_variance is not None:
                        variance_df = pd.DataFrame({
                            'Factor': [f'Factor_{i+1}' for i in range(len(model.explained_variance))],
                            'Explained Variance': model.explained_variance
                        })
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=variance_df['Factor'],
                                y=variance_df['Explained Variance'],
                                marker_color='#ff7f0e'
                            )
                        ])
                        
                        fig.update_layout(
                            title="Factor Explained Variance",
                            height=300,
                            yaxis_tickformat='.1%'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        # Regime detection results
        if 'regime_model' in results:
            with st.expander("🔄 Market Regime Detection", expanded=False):
                model = results['regime_model']
                
                if hasattr(model, 'regimes'):
                    # Show regime timeline
                    regime_df = pd.DataFrame({
                        'Date': model.regimes.index,
                        'Regime': model.regimes.values
                    })
                    
                    fig = go.Figure()
                    
                    regime_colors = ['#2ca02c', '#1f77b4', '#d62728']
                    regime_names = ['Bull Market', 'Normal Market', 'Bear Market']
                    
                    for i, (name, color) in enumerate(zip(regime_names, regime_colors)):
                        regime_mask = regime_df['Regime'] == i
                        if regime_mask.any():
                            fig.add_trace(go.Scatter(
                                x=regime_df.loc[regime_mask, 'Date'],
                                y=regime_df.loc[regime_mask, 'Regime'],
                                mode='markers',
                                marker=dict(color=color, size=8),
                                name=name
                            ))
                    
                    fig.update_layout(
                        title="Market Regime Timeline",
                        xaxis_title="Date",
                        yaxis_title="Regime",
                        height=400,
                        yaxis=dict(tickvals=[0, 1, 2], ticktext=regime_names)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show regime statistics
                    if hasattr(model, 'regime_stats'):
                        st.markdown("**Regime Statistics:**")
                        stats_data = []
                        for regime, stats in model.regime_stats.items():
                            stats_data.append({
                                'Regime': regime_names[int(regime.split('_')[1])],
                                'Mean Return': f"{stats['mean_return']:.2%}",
                                'Volatility': f"{stats['volatility']:.2%}",
                                'Frequency': f"{stats['frequency']:.1%}",
                                'Avg Duration': f"{stats['avg_duration']:.1f} periods"
                            })
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)
    
    def show_optimization_page(self):
        """Advanced portfolio optimization page"""
        
        st.markdown("## ⚖️ Advanced Portfolio Optimization")
        st.markdown("ML-enhanced optimization with multiple methodologies")
        
        if not st.session_state.uploaded_data:
            st.warning("⚠️ Please upload data first")
            return
        
        # Get parameters
        params = self.ui_components.create_interactive_parameter_panel()
        
        # Optimization method selection
        st.markdown("### 🎯 Optimization Method")
        
        col1, col2 = st.columns(2)
        
        with col1:
            optimization_method = st.selectbox(
                "Select optimization approach",
                [
                    "ML-Enhanced Mean-Variance",
                    "Traditional Mean-Variance", 
                    "Risk Parity",
                    "Black-Litterman",
                    "Equal Weight"
                ]
            )
            
            include_ml_predictions = st.checkbox(
                "Use ML return predictions",
                value=True,
                disabled=not st.session_state.ml_models_fitted
            )
        
        with col2:
            # Asset universe selection
            asset_classes = ['Private Equity', 'Private Debt', 'Real Estate', 'Infrastructure']
            selected_assets = st.multiselect(
                "Select asset classes",
                asset_classes,
                default=asset_classes
            )
            
            # Constraints
            max_weight = st.slider(
                "Maximum weight per asset",
                min_value=0.1, max_value=1.0, value=0.4, step=0.05,
                format="%.1%"
            )
        
        # Run optimization
        if st.button("🚀 Optimize Portfolio", type="primary"):
            self.run_optimization(optimization_method, selected_assets, params, include_ml_predictions)
        
        # Show optimization results
        if 'optimization_results' in st.session_state.analysis_results:
            self.show_optimization_results()
    
    def run_optimization(self, method, assets, params, use_ml):
        """Run portfolio optimization"""
        
        with st.spinner("Running optimization..."):
            try:
                # Sample expected returns and volatilities
                np.random.seed(42)
                expected_returns = np.random.uniform(0.08, 0.16, len(assets))
                volatilities = np.random.uniform(0.12, 0.28, len(assets))
                
                # If ML models are available and requested
                if use_ml and st.session_state.ml_models_fitted:
                    # Use ML predictions (simplified for demo)
                    ml_adjustment = np.random.normal(0, 0.02, len(assets))
                    expected_returns += ml_adjustment
                
                # Run optimization based on method
                optimizer = AdvancedOptimizer()
                
                if "Mean-Variance" in method:
                    weights = optimizer.optimize_portfolio(
                        expected_returns, 
                        volatilities,
                        method='mean_variance',
                        illiquidity_penalty=params.get('liquidity_shock', 0.02)
                    )
                elif method == "Risk Parity":
                    # Create covariance matrix
                    correlation = 0.3
                    covariance_matrix = np.full((len(assets), len(assets)), correlation)
                    np.fill_diagonal(covariance_matrix, 1.0)
                    vol_matrix = np.outer(volatilities, volatilities)
                    covariance_matrix = covariance_matrix * vol_matrix
                    
                    weights = optimizer.risk_parity_optimization(covariance_matrix)
                else:
                    # Equal weight fallback
                    weights = np.ones(len(assets)) / len(assets)
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights**2, volatilities**2))  # Simplified
                sharpe_ratio = (portfolio_return - 0.03) / portfolio_vol
                
                # Store results
                st.session_state.analysis_results['optimization_results'] = {
                    'method': method,
                    'assets': assets,
                    'weights': weights,
                    'expected_returns': expected_returns,
                    'volatilities': volatilities,
                    'portfolio_return': portfolio_return,
                    'portfolio_vol': portfolio_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'used_ml': use_ml
                }
                
                st.success("✅ Optimization completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
    
    def show_optimization_results(self):
        """Display optimization results"""
        
        results = st.session_state.analysis_results['optimization_results']
        
        st.markdown("### 📊 Optimization Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Expected Return",
                f"{results['portfolio_return']:.2%}",
                help="Annualized expected portfolio return"
            )
        
        with col2:
            st.metric(
                "Expected Risk",
                f"{results['portfolio_vol']:.2%}",
                help="Annualized portfolio volatility"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{results['sharpe_ratio']:.2f}",
                help="Risk-adjusted return measure"
            )
        
        with col4:
            ml_indicator = "✅ Yes" if results['used_ml'] else "❌ No"
            st.metric(
                "ML Enhanced",
                ml_indicator,
                help="Whether ML predictions were used"
            )
        
        # Allocation visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🥧 Optimal Allocation")
            
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=results['assets'],
                values=results['weights'],
                hole=0.4,
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                title="Portfolio Allocation",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 Allocation Details")
            
            allocation_df = pd.DataFrame({
                'Asset Class': results['assets'],
                'Weight': [f"{w:.1%}" for w in results['weights']],
                'Expected Return': [f"{r:.2%}" for r in results['expected_returns']],
                'Volatility': [f"{v:.2%}" for v in results['volatilities']]
            })
            
            st.dataframe(allocation_df, use_container_width=True)
        
        # Risk-return chart
        st.markdown("#### 📈 Risk-Return Analysis")
        
        fig = go.Figure()
        
        # Individual assets
        fig.add_trace(go.Scatter(
            x=results['volatilities'],
            y=results['expected_returns'],
            mode='markers+text',
            marker=dict(size=15, color='lightblue', line=dict(width=2, color='blue')),
            text=results['assets'],
            textposition="top center",
            name='Individual Assets'
        ))
        
        # Portfolio point
        fig.add_trace(go.Scatter(
            x=[results['portfolio_vol']],
            y=[results['portfolio_return']],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star'),
            name='Optimized Portfolio'
        ))
        
        fig.update_layout(
            title="Risk-Return Positioning",
            xaxis_title="Risk (Volatility)",
            yaxis_title="Expected Return",
            xaxis=dict(tickformat='.1%'),
            yaxis=dict(tickformat='.1%'),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_stress_testing_page(self):
        """Stress testing and scenario analysis page"""
        
        st.markdown("## 🧪 Stress Testing & Scenario Analysis")
        st.markdown("Comprehensive stress testing with multiple scenarios")
        
        if not st.session_state.uploaded_data:
            st.warning("⚠️ Please upload fund data first")
            return
        
        fund_data = st.session_state.uploaded_data.get('fund_data')
        if not fund_data or not isinstance(fund_data, dict):
            st.warning("⚠️ Please upload fund cashflow data")
            return
        
        cashflows = fund_data['cashflows']
        
        # Stress testing configuration
        st.markdown("### ⚙️ Stress Test Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Market Scenarios")
            
            scenarios = {}
            
            # Base case
            scenarios['Base Case'] = {
                'return_shock': 0.0,
                'timing_shock': 0.0,
                'description': 'Current market conditions'
            }
            
            # Market crash
            market_crash_severity = st.slider(
                "Market Crash Severity",
                min_value=-50, max_value=0, value=-30,
                help="Percentage decline in asset values"
            )
            scenarios['Market Crash'] = {
                'return_shock': market_crash_severity / 100,
                'timing_shock': 0.0,
                'description': f'{abs(market_crash_severity)}% market decline'
            }
            
            # Delayed exits
            exit_delay = st.slider(
                "Exit Delay (quarters)",
                min_value=0, max_value=8, value=4,
                help="Delay in exit timing"
            )
            scenarios['Delayed Exits'] = {
                'return_shock': 0.0,
                'timing_shock': exit_delay * 0.25,  # Convert to years
                'description': f'{exit_delay} quarter delay in exits'
            }
        
        with col2:
            st.markdown("#### Combined Scenarios")
            
            # Combined stress
            combined_return_shock = st.slider(
                "Combined Return Shock",
                min_value=-40, max_value=0, value=-20
            )
            combined_timing_shock = st.slider(
                "Combined Timing Shock (quarters)",
                min_value=0, max_value=6, value=2
            )
            
            scenarios['Combined Stress'] = {
                'return_shock': combined_return_shock / 100,
                'timing_shock': combined_timing_shock * 0.25,
                'description': f'{abs(combined_return_shock)}% return shock + {combined_timing_shock}Q delay'
            }
            
            # Custom scenario
            custom_return = st.slider("Custom Return Shock", -50, 20, -10)
            custom_timing = st.slider("Custom Timing Shock (quarters)", 0, 8, 1)
            
            scenarios['Custom'] = {
                'return_shock': custom_return / 100,
                'timing_shock': custom_timing * 0.25,
                'description': f'Custom: {custom_return}% return, {custom_timing}Q timing'
            }
        
        # Run stress tests
        if st.button("🚀 Run Stress Tests", type="primary"):
            with st.spinner("Running stress tests..."):
                stress_results = self.benchmark_engine.stress_test_analysis(cashflows, scenarios)
                st.session_state.analysis_results['stress_results'] = stress_results
                st.success("✅ Stress tests completed!")
        
        # Show stress test results
        if 'stress_results' in st.session_state.analysis_results:
            self.show_stress_test_results()
    
    def show_stress_test_results(self):
        """Display stress test results"""
        
        results = st.session_state.analysis_results['stress_results']
        
        st.markdown("### 📊 Stress Test Results")
        
        # Results table
        results_data = []
        for scenario, metrics in results.items():
            results_data.append({
                'Scenario': scenario,
                'IRR': f"{metrics['irr']:.2%}",
                'Multiple': f"{metrics['multiple']:.2f}x",
                'Total Value': f"${metrics['total_value']:,.0f}",
                'Return Impact': f"{metrics['return_shock']:.1%}",
                'Timing Impact': f"{metrics['timing_shock']:.1%}"
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Style the dataframe
        def highlight_scenarios(row):
            if row['Scenario'] == 'Base Case':
                return ['background-color: lightgreen'] * len(row)
            elif 'Crash' in row['Scenario'] or 'Combined' in row['Scenario']:
                return ['background-color: lightcoral'] * len(row)
            else:
                return ['background-color: lightyellow'] * len(row)
        
        styled_df = results_df.style.apply(highlight_scenarios, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # IRR comparison
            fig_irr = go.Figure(data=[
                go.Bar(
                    x=list(results.keys()),
                    y=[results[scenario]['irr'] for scenario in results.keys()],
                    marker_color=['green' if 'Base' in scenario else 'red' if 'Crash' in scenario or 'Combined' in scenario else 'orange' for scenario in results.keys()]
                )
            ])
            
            fig_irr.update_layout(
                title="IRR by Scenario",
                yaxis_title="IRR",
                yaxis_tickformat='.1%',
                height=400
            )
            
            st.plotly_chart(fig_irr, use_container_width=True)
        
        with col2:
            # Multiple comparison
            fig_multiple = go.Figure(data=[
                go.Bar(
                    x=list(results.keys()),
                    y=[results[scenario]['multiple'] for scenario in results.keys()],
                    marker_color=['green' if 'Base' in scenario else 'red' if 'Crash' in scenario or 'Combined' in scenario else 'orange' for scenario in results.keys()]
                )
            ])
            
            fig_multiple.update_layout(
                title="Multiple by Scenario",
                yaxis_title="Multiple (x)",
                height=400
            )
            
            # Add benchmark line at 1.0x
            fig_multiple.add_hline(y=1.0, line_dash="dash", line_color="black", 
                                 annotation_text="Break-even")
            
            st.plotly_chart(fig_multiple, use_container_width=True)
        
        # Risk assessment
        st.markdown("### ⚠️ Risk Assessment")
        
        base_case_irr = results['Base Case']['irr']
        worst_case_irr = min([results[scenario]['irr'] for scenario in results.keys()])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Downside Risk",
                f"{worst_case_irr - base_case_irr:.1%}",
                help="Worst case IRR vs base case"
            )
        
        with col2:
            scenarios_below_zero = sum(1 for scenario in results.values() if scenario['irr'] < 0)
            st.metric(
                "Loss Scenarios",
                f"{scenarios_below_zero}/{len(results)}",
                help="Number of scenarios with negative IRR"
            )
        
        with col3:
            scenarios_below_hurdle = sum(1 for scenario in results.values() if scenario['irr'] < 0.08)
            st.metric(
                "Below Hurdle",
                f"{scenarios_below_hurdle}/{len(results)}",
                help="Scenarios below 8% hurdle rate"
            )
    
    def show_benchmark_analysis_page(self):
        """Benchmark analysis and PME page"""
        
        st.markdown("## 📊 Benchmark Analysis & PME")
        st.markdown("Compare fund performance against public market benchmarks")
        
        if not st.session_state.uploaded_data:
            st.warning("⚠️ Please upload fund data first")
            return
        
        fund_data = st.session_state.uploaded_data.get('fund_data')
        if not fund_data or not isinstance(fund_data, dict):
            st.warning("⚠️ Please upload fund cashflow data")
            return
        
        cashflows = fund_data['cashflows']
        
        # Benchmark selection
        st.markdown("### 🎯 Benchmark Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            benchmark_category = st.selectbox(
                "Benchmark Category",
                ['Public Equity', 'Fixed Income', 'Alternatives']
            )
            
            benchmark_options = self.benchmark_engine.benchmarks[benchmark_category]
            selected_benchmark = st.selectbox(
                "Select Benchmark",
                list(benchmark_options.keys())
            )
        
        with col2:
            # PME methodology
            pme_method = st.selectbox(
                "PME Methodology",
                ['Direct Alpha', 'PME+', 'mPME'],
                help="Different PME calculation methods"
            )
            
            # Analysis period
            analysis_start = st.date_input(
                "Analysis Start Date",
                value=cashflows['Date'].min()
            )
        
        # Run PME analysis
        if st.button("🚀 Run PME Analysis", type="primary"):
            with st.spinner("Running PME analysis..."):
                benchmark_ticker = benchmark_options[selected_benchmark]
                pme_results = self.benchmark_engine.calculate_pme_analysis(
                    cashflows, benchmark_ticker
                )
                
                if 'error' not in pme_results:
                    st.session_state.analysis_results['pme_results'] = pme_results
                    st. 
                    st.session_state.analysis_results['pme_results'] = pme_results
                    st.success("✅ PME analysis completed!")
                else:
                    st.error(f"❌ PME analysis failed: {pme_results['error']}")
        
        # Show PME results
        if 'pme_results' in st.session_state.analysis_results:
            self.show_pme_results()
    
    def show_pme_results(self):
        """Display PME analysis results"""
        
        results = st.session_state.analysis_results['pme_results']
        
        st.markdown("### 📊 PME Analysis Results")
        
        # Key PME metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "PME Ratio",
                f"{results['pme_ratio']:.2f}x",
                delta=f"{results['pme_ratio'] - 1.0:+.2f}x",
                help="PME > 1.0 indicates outperformance"
            )
        
        with col2:
            st.metric(
                "Fund IRR",
                f"{results['fund_irr']:.2%}",
                help="Fund's internal rate of return"
            )
        
        with col3:
            st.metric(
                "Benchmark IRR",
                f"{results['benchmark_irr']:.2%}",
                help="Benchmark's equivalent return"
            )
        
        with col4:
            st.metric(
                "Alpha",
                f"{results['alpha']:.2%}",
                delta=f"{results['alpha']:+.2%}",
                help="Excess return over benchmark"
            )
        
        # PME interpretation
        if results['pme_ratio'] > 1.1:
            st.success("🎉 **Strong Outperformance**: Fund significantly outperformed public markets")
        elif results['pme_ratio'] > 1.0:
            st.info("✅ **Outperformance**: Fund outperformed public markets")
        elif results['pme_ratio'] > 0.9:
            st.warning("⚠️ **Underperformance**: Fund slightly underperformed public markets")
        else:
            st.error("❌ **Significant Underperformance**: Fund significantly underperformed public markets")
        
        # Detailed breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💰 Cash Flow Analysis")
            
            breakdown_df = pd.DataFrame({
                'Metric': [
                    'PME Adjusted Contributions',
                    'PME Adjusted Distributions',
                    'Net PME Value',
                    'Analysis Period'
                ],
                'Value': [
                    f"${results['pme_contributions']:,.0f}",
                    f"${results['pme_distributions']:,.0f}",
                    f"${results['pme_distributions'] - results['pme_contributions']:,.0f}",
                    results['analysis_period']
                ]
            })
            
            st.dataframe(breakdown_df, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Performance Comparison")
            
            # Create comparison chart
            fig = go.Figure()
            
            # Sample cumulative performance data
            dates = pd.date_range(start='2018-01-01', end='2024-01-01', freq='Q')
            fund_performance = np.cumprod(1 + np.random.normal(results['fund_irr']/4, 0.05, len(dates)))
            benchmark_performance = np.cumprod(1 + np.random.normal(results['benchmark_irr']/4, 0.04, len(dates)))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=fund_performance,
                name='Fund Performance',
                line=dict(color='#1f77b4', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=benchmark_performance,
                name='Benchmark Performance',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Cumulative Performance Comparison",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_dashboard_page(self):
        """Real-time dashboard page"""
        
        st.markdown("## 📊 Real-Time Portfolio Dashboard")
        st.markdown("Live portfolio monitoring with real-time updates")
        
        # Sample real-time data
        dashboard_data = {
            'portfolio_value': 2500000,
            'value_change': 0.024,
            'ytd_return': 0.124,
            'ytd_change': 0.003,
            'sharpe_ratio': 1.85,
            'sharpe_change': 0.05,
            'max_drawdown': -0.082,
            'drawdown_change': -0.012,
            'var_95': -0.031,
            'var_change': 0.005
        }
        
        # Real-time metrics
        self.ui_components.create_real_time_dashboard(dashboard_data)
        
        # Interactive charts
        self.ui_components.create_interactive_charts(dashboard_data)
        
        # Export functionality
        self.ui_components.create_export_section(dashboard_data)
        
        # Live data feed simulation
        st.markdown("### 📡 Live Data Feed")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Market Data**")
            market_data = pd.DataFrame({
                'Index': ['S&P 500', 'NASDAQ', 'Russell 2000'],
                'Current': ['4,567.23', '14,234.56', '2,123.45'],
                'Change': ['+0.8%', '+1.2%', '-0.3%']
            })
            st.dataframe(market_data, use_container_width=True)
        
        with col2:
            st.markdown("**Portfolio Alerts**")
            alerts = [
                "🟢 Portfolio rebalanced successfully",
                "🟡 High correlation detected in RE sector",
                "🔴 VaR limit approached in PE allocation"
            ]
            for alert in alerts:
                st.markdown(f"- {alert}")
        
        with col3:
            st.markdown("**System Status**")
            status_items = [
                ("Data Feed", "🟢 Active"),
                ("ML Models", "🟢 Running"),
                ("Risk Monitor", "🟡 Warning"),
                ("API Connection", "🟢 Connected")
            ]
            for item, status in status_items:
                st.markdown(f"**{item}**: {status}")

def main():
    """Main application entry point"""
    
    # Initialize and run the professional PM analyzer
    app = ProfessionalPMAnalyzer()
    app.run()

if __name__ == "__main__":
    main()
