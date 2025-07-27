"""
Advanced analytics examples for PM-Analyzer
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

def demonstrate_pme_analysis():
    """Demonstrate Public Market Equivalent analysis"""
    print("Public Market Equivalent (PME) Analysis")
    print("-" * 40)
    
    analytics = ReturnAnalytics()
    
    # Simulate private equity cash flows
    dates = pd.date_range('2018-01-01', '2023-12-31', freq='Q')
    
    # Typical PE J-curve pattern
    contributions = np.array([25, 30, 20, 15, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    distributions = np.array([0, 0, 0, 5, 10, 15, 25, 30, 35, 40, 30, 20, 15, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Calculate NAV progression
    nav_values = np.cumsum(contributions - distributions) + np.cumsum(np.random.normal(2, 5, len(dates)))
    nav_values = np.maximum(nav_values, 0)  # Ensure non-negative NAV
    
    # Calculate PME
    pme = analytics.calculate_pme(nav_values, dates)
    mwir = analytics.calculate_mwir(nav_values, dates, contributions - distributions)
    
    print(f"Private Equity Investment Analysis:")
    print(f"  Total Contributions: ${np.sum(contributions):.1f}M")
    print(f"  Total Distributions: ${np.sum(distributions):.1f}M")
    print(f"  Final NAV: ${nav_values[-1]:.1f}M")
    print(f"  Money-Weighted IRR: {mwir:.2%}")
    print(f"  Public Market Equivalent: {pme:.2f}x")
    
    if pme > 1.0:
        print(f"  → PE investment outperformed public markets by {(pme-1)*100:.1f}%")
    else:
        print(f"  → PE investment underperformed public markets by {(1-pme)*100:.1f}%")

def demonstrate_cashflow_modeling():
    """Demonstrate cash flow modeling for private equity"""
    print("\nPrivate Equity Cash Flow Modeling")
    print("-" * 40)
    
    simulator = MonteCarloSimulator(random_seed=123)
    
    # Simulate cash flows for a typical PE fund
    cashflow_results = simulator.simulate_cashflows(
        num_simulations=1000,
        commitment=100.0,
        drawdown_schedule=[0.25, 0.35, 0.25, 0.15, 0.0, 0.0, 0.0, 0.0],
        distribution_schedule=[0.0, 0.0, 0.05, 0.15, 0.25, 0.30, 0.20, 0.05]
    )
    
    # Analyze results
    total_drawdowns = np.sum(cashflow_results['drawdowns'], axis=1)
    total_distributions = np.sum(cashflow_results['distributions'], axis=1)
    net_multiple = total_distributions / total_drawdowns
    
    print(f"Cash Flow Simulation Results (1000 simulations):")
    print(f"  Average Total Drawdowns: ${np.mean(total_drawdowns):.1f}M")
    print(f"  Average Total Distributions: ${np.mean(total_distributions):.1f}M")
    print(f"  Average Net Multiple: {np.mean(net_multiple):.2f}x")
    print(f"  Probability of 2x+ Multiple: {np.mean(net_multiple >= 2.0):.1%}")
    print(f"  Probability of Loss: {np.mean(net_multiple < 1.0):.1%}")

def demonstrate_risk_attribution():
    """Demonstrate risk attribution analysis"""
    print("\nRisk Attribution Analysis")
    print("-" * 40)
    
    analytics = ReturnAnalytics()
    
    # Simulate portfolio with multiple asset classes
    asset_classes = ['Private Equity', 'Private Debt', 'Real Estate', 'Infrastructure']
    weights = {'Private Equity': 0.4, 'Private Debt': 0.3, 'Real Estate': 0.2, 'Infrastructure': 0.1}
    
    # Generate correlated returns
    np.random.seed(42)
    n_periods = 36
    
    # Create correlation structure
    correlation_matrix = np.array([
        [1.0, 0.3, 0.4, 0.5],
        [0.3, 1.0, 0.2, 0.3],
        [0.4, 0.2, 1.0, 0.4],
        [0.5, 0.3, 0.4, 1.0]
    ])
    
    # Generate correlated random variables
    random_vars = np.random.multivariate_normal([0, 0, 0, 0], correlation_matrix, n_periods)
    
    # Convert to returns with different volatilities
    volatilities = [0.25, 0.15, 0.20, 0.18]
    expected_returns = [0.14, 0.10, 0.12, 0.11]
    
    asset_returns = {}
    for i, asset in enumerate(asset_classes):
        monthly_return = expected_returns[i] / 12
        monthly_vol = volatilities[i] / np.sqrt(12)
        asset_returns[asset] = monthly_return + monthly_vol * random_vars[:, i]
    
    # Calculate portfolio return
    portfolio_returns = np.sum([weights[asset] * asset_returns[asset] for asset in asset_classes], axis=0)
    
    # Performance attribution
    attribution = analytics.generate_performance_attribution(
        portfolio_returns, asset_returns, weights
    )
    
    print("Performance Attribution:")
    total_return = np.mean(portfolio_returns) * 12  # Annualized
    
    for asset, attr in attribution.items():
        contribution_pct = (attr['contribution'] * 12) / total_return * 100
        print(f"  {asset}:")
        print(f"    Weight: {attr['weight']:.1%}")
        print(f"    Return: {attr['return']*12:.2%}")
        print(f"    Contribution: {attr['contribution']*12:.2%} ({contribution_pct:.1f}% of total)")
    
    print(f"\nTotal Portfolio Return: {total_return:.2%}")

def demonstrate_optimization_constraints():
    """Demonstrate portfolio optimization with various constraints"""
    print("\nAdvanced Portfolio Optimization")
    print("-" * 40)
    
    optimizer = PortfolioOptimizer()
    
    asset_classes = ['Private Equity', 'Private Debt', 'Real Estate', 'Infrastructure', 'Hedge Funds']
    expected_returns = [0.14, 0.10, 0.12, 0.11, 0.09]
    volatilities = [0.25, 0.15, 0.20, 0.18, 0.12]
    
    # Scenario 1: Unconstrained optimization
    weights_unconstrained = optimizer.optimize_portfolio(expected_returns, volatilities)
    
    # Scenario 2: With minimum allocations
    constraints_min = {
        'min_weights': [0.2, 0.1, 0.1, 0.05, 0.0]  # Minimum allocations
    }
    weights_min_constrained = optimizer.optimize_portfolio(
        expected_returns, volatilities, constraints=constraints_min
    )
    
    # Scenario 3: With maximum allocations
    constraints_max = {
        'max_weights': [0.4, 0.3, 0.3, 0.2, 0.2]  # Maximum allocations
    }
    weights_max_constrained = optimizer.optimize_portfolio(
        expected_returns, volatilities, constraints=constraints_max
    )
    
    # Compare results
    scenarios = [
        ("Unconstrained", weights_unconstrained),
        ("Min Constraints", weights_min_constrained),
        ("Max Constraints", weights_max_constrained)
    ]
    
    print("Optimization Results Comparison:")
    print(f"{'Asset':<15} {'Unconstrained':<12} {'Min Constr.':<12} {'Max Constr.':<12}")
    print("-" * 60)
    
    for i, asset in enumerate(asset_classes):
        print(f"{asset:<15} {weights_unconstrained[i]:<11.1%} {weights_min_constrained[i]:<11.1%} {weights_max_constrained[i]:<11.1%}")
    
    # Calculate portfolio metrics for each scenario
    print(f"\nPortfolio Metrics:")
    print(f"{'Scenario':<15} {'Return':<8} {'Risk':<8} {'Sharpe':<8}")
    print("-" * 40)
    
    for name, weights in scenarios:
        port_return = np.dot(weights, expected_returns)
        port_risk = np.sqrt(np.dot(weights**2, np.array(volatilities)**2))
        sharpe = (port_return - 0.03) / port_risk
        print(f"{name:<15} {port_return:<7.2%} {port_risk:<7.2%} {sharpe:<7.2f}")

def main():
    """Run all advanced analytics demonstrations"""
    print("PM-Analyzer Advanced Analytics Examples")
    print("=" * 50)
    
    demonstrate_pme_analysis()
    demonstrate_cashflow_modeling()
    demonstrate_risk_attribution()
    demonstrate_optimization_constraints()
    
    print("\n" + "=" * 50)
    print("Advanced analytics demonstration complete!")

if __name__ == "__main__":
    main()
