"""
Basic usage examples for PM-Analyzer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data_ingestion import DataIngestion
from src.return_analytics import ReturnAnalytics
from src.monte_carlo import MonteCarloSimulator
from src.portfolio_optimizer import PortfolioOptimizer

def main():
    print("PM-Analyzer Basic Usage Examples")
    print("=" * 40)
    
    # Initialize components
    data_ingestion = DataIngestion()
    analytics = ReturnAnalytics()
    simulator = MonteCarloSimulator(random_seed=42)
    optimizer = PortfolioOptimizer()
    
    # Example 1: Generate and analyze sample data
    print("\n1. Generating Sample Data")
    sample_data = data_ingestion.generate_sample_data()
    print(f"Generated data for {len(sample_data['allocations'])} asset classes")
    print(f"Cash flow data: {len(sample_data['cashflows'])} periods")
    
    # Example 2: Calculate return metrics
    print("\n2. Return Analytics")
    
    # Generate sample returns
    dates = pd.date_range(start='2020-01-01', periods=36, freq='M')
    returns = np.random.normal(0.08/12, 0.15/np.sqrt(12), len(dates))
    nav_values = 100 * np.cumprod(1 + returns)
    
    # Calculate metrics
    mwir = analytics.calculate_mwir(nav_values, dates)
    twir = analytics.calculate_twir(returns)
    sharpe = analytics.calculate_sharpe_ratio(returns)
    max_dd = analytics.calculate_max_drawdown(returns)
    var_95 = analytics.calculate_var(returns, 0.95)
    
    print(f"Money-Weighted IRR: {mwir:.2%}")
    print(f"Time-Weighted Return: {twir:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Maximum Drawdown: {max_dd:.2%}")
    print(f"Value at Risk (95%): {var_95:.2%}")
    
    # Example 3: Monte Carlo Simulation
    print("\n3. Monte Carlo Simulation")
    
    mc_results = simulator.run_simulation(
        num_simulations=1000,
        expected_return=0.12,
        volatility=0.25,
        time_horizon=5,
        initial_value=100.0
    )
    
    stats = mc_results['statistics']
    risk_metrics = mc_results['risk_metrics']
    
    print(f"Expected Final Value: ${stats['mean']:.1f}M")
    print(f"Standard Deviation: ${stats['std']:.1f}M")
    print(f"5th Percentile: ${stats['percentiles']['5th']:.1f}M")
    print(f"95th Percentile: ${stats['percentiles']['95th']:.1f}M")
    print(f"Probability of Loss: {risk_metrics['prob_loss']:.1%}")
    
    # Example 4: Portfolio Optimization
    print("\n4. Portfolio Optimization")
    
    asset_classes = ['Private Equity', 'Private Debt', 'Real Estate', 'Infrastructure']
    expected_returns = [0.14, 0.10, 0.12, 0.11]
    volatilities = [0.25, 0.15, 0.20, 0.18]
    
    # Optimize for maximum Sharpe ratio
    optimal_weights = optimizer.optimize_portfolio(
        expected_returns,
        volatilities,
        illiquidity_penalty=0.02
    )
    
    print("Optimal Portfolio Allocation:")
    for asset, weight in zip(asset_classes, optimal_weights):
        print(f"  {asset}: {weight:.1%}")
    
    # Calculate portfolio metrics
    portfolio_return = np.dot(optimal_weights, expected_returns)
    portfolio_vol = np.sqrt(np.dot(optimal_weights**2, np.array(volatilities)**2))
    portfolio_sharpe = (portfolio_return - 0.03) / portfolio_vol
    
    print(f"\nPortfolio Metrics:")
    print(f"  Expected Return: {portfolio_return:.2%}")
    print(f"  Expected Volatility: {portfolio_vol:.2%}")
    print(f"  Sharpe Ratio: {portfolio_sharpe:.2f}")
    
    # Example 5: Scenario Analysis
    print("\n5. Scenario Analysis")
    
    base_case = {
        'num_simulations': 500,
        'expected_return': 0.12,
        'volatility': 0.25,
        'time_horizon': 5
    }
    
    scenarios = {
        'bull_market': {'expected_return': 0.18, 'volatility': 0.20},
        'bear_market': {'expected_return': 0.06, 'volatility': 0.35},
        'high_volatility': {'volatility': 0.40}
    }
    
    scenario_results = simulator.scenario_analysis(base_case, scenarios)
    
    print("Scenario Analysis Results (Expected Final Value):")
    for scenario, results in scenario_results.items():
        mean_value = results['statistics']['mean']
        prob_loss = results['risk_metrics']['prob_loss']
        print(f"  {scenario.replace('_', ' ').title()}: ${mean_value:.1f}M (Loss Prob: {prob_loss:.1%})")

if __name__ == "__main__":
    main()
