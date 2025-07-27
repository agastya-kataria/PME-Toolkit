"""
PM-Analyzer: Private Markets Portfolio Analyzer
Main entry point for the application
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from data_ingestion import DataIngestion
    from return_analytics import ReturnAnalytics
    from monte_carlo import MonteCarloSimulator
    from portfolio_optimizer import PortfolioOptimizer
    from visualization import DashboardVisualizer
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

def main():
    st.set_page_config(
        page_title="PM-Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏦 Private Markets Portfolio Analyzer")
    st.markdown("*Comprehensive analysis tool for private equity, debt, and real estate investments*")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Portfolio Overview", "Return Analytics", "Monte Carlo Simulation", "Portfolio Optimization", "Risk Metrics"]
    )
    
    # Initialize components
    try:
        data_ingestion = DataIngestion()
        return_analytics = ReturnAnalytics()
        monte_carlo = MonteCarloSimulator()
        optimizer = PortfolioOptimizer()
        visualizer = DashboardVisualizer()
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        st.stop()
    
    if page == "Portfolio Overview":
        show_portfolio_overview(data_ingestion, visualizer)
    elif page == "Return Analytics":
        show_return_analytics(return_analytics, visualizer)
    elif page == "Monte Carlo Simulation":
        show_monte_carlo(monte_carlo, visualizer)
    elif page == "Portfolio Optimization":
        show_portfolio_optimization(optimizer, visualizer)
    elif page == "Risk Metrics":
        show_risk_metrics(return_analytics, visualizer)

def show_portfolio_overview(data_ingestion, visualizer):
    st.header("📈 Portfolio Overview")
    
    # Sample data generation
    sample_data = data_ingestion.generate_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Asset Allocation")
        allocation_fig = visualizer.create_allocation_chart(sample_data['allocations'])
        st.plotly_chart(allocation_fig, use_container_width=True)
    
    with col2:
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Total AUM', 'YTD Return', 'Sharpe Ratio', 'Max Drawdown'],
            'Value': ['$2.5B', '12.4%', '1.85', '-8.2%']
        })
        st.dataframe(metrics_df, hide_index=True)
    
    st.subheader("Cash Flow Timeline")
    cashflow_fig = visualizer.create_cashflow_chart(sample_data['cashflows'])
    st.plotly_chart(cashflow_fig, use_container_width=True)

def show_return_analytics(return_analytics, visualizer):
    st.header("📊 Return Analytics")
    
    # Input parameters
    col1, col2 = st.columns(2)
    with col1:
        investment_amount = st.number_input("Initial Investment ($M)", value=100.0, min_value=1.0)
    with col2:
        time_horizon = st.slider("Time Horizon (Years)", min_value=1, max_value=15, value=7)
    
    # Generate sample returns data
    dates = pd.date_range(start='2020-01-01', periods=time_horizon*12, freq='M')
    returns = np.random.normal(0.08/12, 0.15/np.sqrt(12), len(dates))
    nav_values = investment_amount * np.cumprod(1 + returns)
    
    # Calculate metrics
    mwir = return_analytics.calculate_mwir(nav_values, dates)
    twir = return_analytics.calculate_twir(returns)
    pme = return_analytics.calculate_pme(nav_values, dates)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Money-Weighted IRR", f"{mwir:.2%}")
    with col2:
        st.metric("Time-Weighted Return", f"{twir:.2%}")
    with col3:
        st.metric("Public Market Equivalent", f"{pme:.2f}x")
    
    # Performance chart
    performance_fig = visualizer.create_performance_chart(dates, nav_values)
    st.plotly_chart(performance_fig, use_container_width=True)

def show_monte_carlo(monte_carlo, visualizer):
    st.header("🎲 Monte Carlo Simulation")
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        num_simulations = st.slider("Number of Simulations", 100, 10000, 1000)
    with col2:
        expected_return = st.slider("Expected Annual Return", 0.05, 0.20, 0.12)
    with col3:
        volatility = st.slider("Annual Volatility", 0.10, 0.40, 0.25)
    
    # Run simulation
    if st.button("Run Monte Carlo Simulation"):
        with st.spinner("Running simulation..."):
            results = monte_carlo.run_simulation(
                num_simulations=num_simulations,
                expected_return=expected_return,
                volatility=volatility,
                time_horizon=5
            )
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Simulation Results")
            percentiles = np.percentile(results['final_values'], [5, 25, 50, 75, 95])
            results_df = pd.DataFrame({
                'Percentile': ['5th', '25th', '50th (Median)', '75th', '95th'],
                'Final Value ($M)': [f"${p:.1f}" for p in percentiles]
            })
            st.dataframe(results_df, hide_index=True)
        
        with col2:
            st.subheader("Risk Metrics")
            prob_loss = (results['final_values'] < 100).mean()
            expected_shortfall = np.mean(results['final_values'][results['final_values'] < np.percentile(results['final_values'], 5)])
            
            st.metric("Probability of Loss", f"{prob_loss:.1%}")
            st.metric("Expected Shortfall (5%)", f"${expected_shortfall:.1f}M")
        
        # Visualization
        monte_carlo_fig = visualizer.create_monte_carlo_chart(results)
        st.plotly_chart(monte_carlo_fig, use_container_width=True)

def show_portfolio_optimization(optimizer, visualizer):
    st.header("⚖️ Portfolio Optimization")
    
    # Asset classes
    asset_classes = ['Private Equity', 'Private Debt', 'Real Estate', 'Infrastructure']
    
    st.subheader("Expected Returns and Risk")
    expected_returns = []
    volatilities = []
    
    for i, asset in enumerate(asset_classes):
        col1, col2 = st.columns(2)
        with col1:
            ret = st.slider(f"{asset} Expected Return", 0.05, 0.20, 0.08 + i*0.02, key=f"ret_{i}")
            expected_returns.append(ret)
        with col2:
            vol = st.slider(f"{asset} Volatility", 0.10, 0.40, 0.15 + i*0.03, key=f"vol_{i}")
            volatilities.append(vol)
    
    # Optimization constraints
    st.subheader("Constraints")
    col1, col2 = st.columns(2)
    with col1:
        target_return = st.slider("Target Return", 0.08, 0.18, 0.12)
    with col2:
        illiquidity_penalty = st.slider("Illiquidity Penalty", 0.0, 0.05, 0.02)
    
    if st.button("Optimize Portfolio"):
        # Run optimization
        optimal_weights = optimizer.optimize_portfolio(
            expected_returns, volatilities, target_return, illiquidity_penalty
        )
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Optimal Allocation")
            allocation_df = pd.DataFrame({
                'Asset Class': asset_classes,
                'Weight': [f"{w:.1%}" for w in optimal_weights]
            })
            st.dataframe(allocation_df, hide_index=True)
        
        with col2:
            st.subheader("Portfolio Metrics")
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(optimal_weights**2, np.array(volatilities)**2))
            sharpe_ratio = (portfolio_return - 0.03) / portfolio_vol
            
            st.metric("Expected Return", f"{portfolio_return:.2%}")
            st.metric("Expected Volatility", f"{portfolio_vol:.2%}")
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        # Allocation chart
        allocation_fig = visualizer.create_allocation_chart(dict(zip(asset_classes, optimal_weights)))
        st.plotly_chart(allocation_fig, use_container_width=True)

def show_risk_metrics(return_analytics, visualizer):
    st.header("⚠️ Risk Metrics")
    
    # Generate sample data
    dates = pd.date_range(start='2020-01-01', periods=48, freq='M')
    returns = np.random.normal(0.08/12, 0.15/np.sqrt(12), len(dates))
    
    # Calculate risk metrics
    var_95 = return_analytics.calculate_var(returns, confidence_level=0.95)
    cvar_95 = return_analytics.calculate_cvar(returns, confidence_level=0.95)
    max_drawdown = return_analytics.calculate_max_drawdown(returns)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Value at Risk (95%)", f"{var_95:.2%}")
    with col2:
        st.metric("Conditional VaR (95%)", f"{cvar_95:.2%}")
    with col3:
        st.metric("Maximum Drawdown", f"{max_drawdown:.2%}")
    
    # Risk decomposition chart
    risk_fig = visualizer.create_risk_decomposition_chart()
    st.plotly_chart(risk_fig, use_container_width=True)

if __name__ == "__main__":
    main()
