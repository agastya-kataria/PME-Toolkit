"""
PM-Analyzer with Historical Backtesting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.optimize import minimize, fsolve
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="PM-Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class HistoricalDataGenerator:
    """Generate realistic historical data for backtesting"""
    
    @staticmethod
    def generate_historical_returns(start_date='2015-01-01', end_date='2024-12-31', freq='M'):
        """Generate synthetic historical returns for private markets asset classes"""
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_periods = len(dates)
        
        # Set random seed for reproducible results
        np.random.seed(42)
        
        # Asset class parameters (annual)
        asset_params = {
            'Private Equity': {'mean': 0.14, 'vol': 0.25, 'autocorr': 0.3},
            'Private Debt': {'mean': 0.10, 'vol': 0.15, 'autocorr': 0.2},
            'Real Estate': {'mean': 0.12, 'vol': 0.20, 'autocorr': 0.4},
            'Infrastructure': {'mean': 0.11, 'vol': 0.18, 'autocorr': 0.35},
            'Public Equity': {'mean': 0.10, 'vol': 0.16, 'autocorr': 0.1}  # Benchmark
        }
        
        # Generate correlated returns with autocorrelation
        returns_data = {}
        
        for asset, params in asset_params.items():
            # Convert annual parameters to monthly
            monthly_mean = params['mean'] / 12
            monthly_vol = params['vol'] / np.sqrt(12)
            autocorr = params['autocorr']
            
            # Generate returns with autocorrelation (AR(1) process)
            returns = np.zeros(n_periods)
            returns[0] = np.random.normal(monthly_mean, monthly_vol)
            
            for i in range(1, n_periods):
                returns[i] = (autocorr * returns[i-1] + 
                             (1 - autocorr) * monthly_mean + 
                             np.sqrt(1 - autocorr**2) * monthly_vol * np.random.normal())
            
            # Add market cycles and regime changes
            returns = HistoricalDataGenerator._add_market_cycles(returns, dates)
            
            returns_data[asset] = returns
        
        # Create DataFrame
        returns_df = pd.DataFrame(returns_data, index=dates)
        
        # Add some cross-correlation
        correlation_matrix = np.array([
            [1.0, 0.3, 0.4, 0.5, 0.6],  # PE
            [0.3, 1.0, 0.2, 0.3, 0.4],  # PD
            [0.4, 0.2, 1.0, 0.4, 0.5],  # RE
            [0.5, 0.3, 0.4, 1.0, 0.4],  # Infra
            [0.6, 0.4, 0.5, 0.4, 1.0]   # Public
        ])
        
        # Apply correlation structure
        returns_corr = HistoricalDataGenerator._apply_correlation(returns_df.values, correlation_matrix)
        returns_df = pd.DataFrame(returns_corr, index=dates, columns=returns_df.columns)
        
        return returns_df
    
    @staticmethod
    def _add_market_cycles(returns, dates):
        """Add market cycles and crisis periods"""
        returns_adj = returns.copy()
        
        for i, date in enumerate(dates):
            # COVID crisis (2020)
            if date.year == 2020 and date.month in [3, 4]:
                returns_adj[i] -= 0.15  # 15% additional decline
            elif date.year == 2020 and date.month in [5, 6, 7]:
                returns_adj[i] += 0.08  # Recovery
            
            # Market stress (2018)
            elif date.year == 2018 and date.month in [10, 11, 12]:
                returns_adj[i] -= 0.05
            
            # Bull market (2016-2017)
            elif date.year in [2016, 2017]:
                returns_adj[i] += 0.01
        
        return returns_adj
    
    @staticmethod
    def _apply_correlation(returns_matrix, correlation_matrix):
        """Apply correlation structure to returns"""
        # Cholesky decomposition for correlation
        try:
            L = np.linalg.cholesky(correlation_matrix)
            # Transform returns
            returns_corr = returns_matrix @ L.T
            return returns_corr
        except np.linalg.LinAlgError:
            # If correlation matrix is not positive definite, return original
            return returns_matrix

class BacktestEngine:
    """Comprehensive backtesting engine for portfolio strategies"""
    
    def __init__(self, returns_data, rebalance_freq='Q'):
        self.returns_data = returns_data
        self.rebalance_freq = rebalance_freq
        self.asset_names = list(returns_data.columns)
        
    def run_backtest(self, optimization_method, method_params=None, 
                    lookback_window=36, min_history=24, initial_capital=1000000):
        """
        Run historical backtest of optimization strategy
        
        Parameters:
        - optimization_method: str, optimization method to use
        - method_params: dict, parameters for optimization method
        - lookback_window: int, months of data to use for optimization
        - min_history: int, minimum months before starting backtest
        - initial_capital: float, starting portfolio value
        """
        
        if method_params is None:
            method_params = {}
        
        # Get rebalancing dates
        rebalance_dates = self._get_rebalance_dates()
        
        # Initialize results storage
        results = {
            'dates': [],
            'portfolio_values': [],
            'weights_history': [],
            'returns': [],
            'benchmark_values': [],
            'drawdowns': [],
            'turnover': [],
            'optimization_metrics': []
        }
        
        # Initialize portfolio
        portfolio_value = initial_capital
        benchmark_value = initial_capital
        previous_weights = None
        
        # Equal weight benchmark (or use Public Equity if available)
        if 'Public Equity' in self.asset_names:
            benchmark_returns = self.returns_data['Public Equity']
        else:
            benchmark_returns = self.returns_data.mean(axis=1)
        
        optimizer = AdvancedOptimizer()
        
        for i, rebal_date in enumerate(rebalance_dates):
            # Skip if not enough history
            if rebal_date < self.returns_data.index[min_history]:
                continue
            
            # Get historical data for optimization
            hist_start = max(0, self.returns_data.index.get_loc(rebal_date) - lookback_window)
            hist_end = self.returns_data.index.get_loc(rebal_date)
            
            hist_returns = self.returns_data.iloc[hist_start:hist_end]
            
            if len(hist_returns) < min_history:
                continue
            
            # Calculate expected returns and volatilities from historical data
            expected_returns = hist_returns.mean() * 12  # Annualize
            volatilities = hist_returns.std() * np.sqrt(12)  # Annualize
            
            # Run optimization
            try:
                weights = optimizer.optimize_portfolio(
                    expected_returns.values, 
                    volatilities.values, 
                    method=optimization_method,
                    **method_params
                )
                
                # Calculate optimization metrics
                opt_metrics = optimizer.calculate_portfolio_metrics(
                    weights, expected_returns.values, volatilities.values
                )
                
            except Exception as e:
                # Fallback to equal weights if optimization fails
                weights = np.ones(len(self.asset_names)) / len(self.asset_names)
                opt_metrics = {'return': np.nan, 'risk': np.nan, 'sharpe': np.nan}
            
            # Calculate turnover
            turnover = 0
            if previous_weights is not None:
                turnover = np.sum(np.abs(weights - previous_weights)) / 2
            
            # Get next rebalancing date or end date
            if i < len(rebalance_dates) - 1:
                next_rebal_date = rebalance_dates[i + 1]
            else:
                next_rebal_date = self.returns_data.index[-1]
            
            # Get returns between rebalancing dates
            period_start = self.returns_data.index.get_loc(rebal_date)
            period_end = self.returns_data.index.get_loc(next_rebal_date) + 1
            period_returns = self.returns_data.iloc[period_start:period_end]
            
            # Calculate portfolio performance for this period
            for date_idx, date in enumerate(period_returns.index):
                if date_idx == 0:  # Rebalancing date
                    results['dates'].append(date)
                    results['portfolio_values'].append(portfolio_value)
                    results['benchmark_values'].append(benchmark_value)
                    results['weights_history'].append(weights.copy())
                    results['returns'].append(0)
                    results['turnover'].append(turnover)
                    results['optimization_metrics'].append(opt_metrics)
                else:
                    # Calculate returns
                    period_return = period_returns.iloc[date_idx]
                    portfolio_return = np.dot(weights, period_return)
                    benchmark_return = benchmark_returns.loc[date]
                    
                    # Update values
                    portfolio_value *= (1 + portfolio_return)
                    benchmark_value *= (1 + benchmark_return)
                    
                    results['dates'].append(date)
                    results['portfolio_values'].append(portfolio_value)
                    results['benchmark_values'].append(benchmark_value)
                    results['weights_history'].append(weights.copy())
                    results['returns'].append(portfolio_return)
                    results['turnover'].append(0)  # Only non-zero on rebalancing dates
                    results['optimization_metrics'].append(opt_metrics)
            
            previous_weights = weights
        
        # Calculate drawdowns
        portfolio_values = np.array(results['portfolio_values'])
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        results['drawdowns'] = drawdowns.tolist()
        
        return self._create_backtest_results(results)
    
    def _get_rebalance_dates(self):
        """Get rebalancing dates based on frequency"""
        if self.rebalance_freq == 'M':
            return self.returns_data.index
        elif self.rebalance_freq == 'Q':
            return self.returns_data.index[::3]  # Every 3 months
        elif self.rebalance_freq == 'A':
            return self.returns_data.index[::12]  # Every 12 months
        else:
            return self.returns_data.index[::3]  # Default to quarterly
    
    def _create_backtest_results(self, results):
        """Create structured backtest results"""
        dates = pd.to_datetime(results['dates'])
        
        # Create main results DataFrame
        results_df = pd.DataFrame({
            'Date': dates,
            'Portfolio_Value': results['portfolio_values'],
            'Benchmark_Value': results['benchmark_values'],
            'Portfolio_Return': results['returns'],
            'Drawdown': results['drawdowns'],
            'Turnover': results['turnover']
        })
        
        # Calculate performance metrics
        portfolio_returns = np.array(results['returns'][1:])  # Skip first zero return
        benchmark_returns = np.diff(results['benchmark_values']) / np.array(results['benchmark_values'][:-1])
        
        # Performance statistics
        performance_stats = self._calculate_performance_stats(
            portfolio_returns, benchmark_returns, results['portfolio_values'], results['benchmark_values']
        )
        
        # Weights history
        weights_df = pd.DataFrame(
            results['weights_history'], 
            index=dates, 
            columns=self.asset_names
        )
        
        return {
            'results_df': results_df,
            'weights_df': weights_df,
            'performance_stats': performance_stats,
            'optimization_metrics': results['optimization_metrics']
        }
    
    def _calculate_performance_stats(self, portfolio_returns, benchmark_returns, 
                                   portfolio_values, benchmark_values):
        """Calculate comprehensive performance statistics"""
        
        # Annualized returns
        total_periods = len(portfolio_returns)
        years = total_periods / 12
        
        portfolio_total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        benchmark_total_return = (benchmark_values[-1] / benchmark_values[0]) - 1
        
        portfolio_annual_return = (1 + portfolio_total_return) ** (1/years) - 1
        benchmark_annual_return = (1 + benchmark_total_return) ** (1/years) - 1
        
        # Volatility
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(12)
        benchmark_vol = np.std(benchmark_returns) * np.sqrt(12)
        
        # Sharpe ratio
        risk_free_rate = 0.03
        portfolio_sharpe = (portfolio_annual_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        benchmark_sharpe = (benchmark_annual_return - risk_free_rate) / benchmark_vol if benchmark_vol > 0 else 0
        
        # Maximum drawdown
        portfolio_values_arr = np.array(portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values_arr)
        drawdowns = (portfolio_values_arr - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Information ratio
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(12)
        information_ratio = np.mean(excess_returns) * 12 / tracking_error if tracking_error > 0 else 0
        
        # Win rate
        win_rate = np.mean(portfolio_returns > benchmark_returns)
        
        # Calmar ratio
        calmar_ratio = portfolio_annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        return {
            'Portfolio Annual Return': portfolio_annual_return,
            'Benchmark Annual Return': benchmark_annual_return,
            'Excess Return': portfolio_annual_return - benchmark_annual_return,
            'Portfolio Volatility': portfolio_vol,
            'Benchmark Volatility': benchmark_vol,
            'Portfolio Sharpe': portfolio_sharpe,
            'Benchmark Sharpe': benchmark_sharpe,
            'Information Ratio': information_ratio,
            'Maximum Drawdown': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Win Rate': win_rate,
            'Tracking Error': tracking_error,
            'Total Return': portfolio_total_return,
            'Benchmark Total Return': benchmark_total_return
        }
    
    def compare_strategies(self, strategies, **backtest_params):
        """Compare multiple optimization strategies"""
        comparison_results = {}
        
        for strategy_name, (method, params) in strategies.items():
            print(f"Backtesting {strategy_name}...")
            results = self.run_backtest(method, params, **backtest_params)
            comparison_results[strategy_name] = results
        
        return comparison_results

class SimpleDataGenerator:
    """Simple data generation for demo purposes"""
    
    @staticmethod
    def generate_sample_data():
        """Generate sample private markets data"""
        # Sample allocations
        allocations = {
            'Private Equity': 0.35,
            'Private Debt': 0.25,
            'Real Estate': 0.30,
            'Infrastructure': 0.10
        }
        
        # Sample cash flows
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='Q')
        np.random.seed(42)
        cashflows = pd.DataFrame({
            'Date': dates,
            'Contributions': np.random.normal(50, 15, len(dates)),
            'Distributions': np.random.normal(30, 20, len(dates))
        })
        cashflows['Distributions'] = np.maximum(cashflows['Distributions'], 0)
        cashflows['Net_Cash_Flow'] = cashflows['Contributions'] - cashflows['Distributions']
        
        return {'allocations': allocations, 'cashflows': cashflows}

class SimpleAnalytics:
    """Simplified analytics calculations"""
    
    def __init__(self):
        self.risk_free_rate = 0.03
    
    def calculate_twir(self, returns):
        """Calculate time-weighted return"""
        if len(returns) == 0:
            return 0.0
        return np.prod(1 + returns) ** (12 / len(returns)) - 1
    
    def calculate_sharpe_ratio(self, returns):
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = np.mean(returns) - self.risk_free_rate / 12
        return excess_returns / np.std(returns)
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def calculate_var(self, returns, confidence_level=0.95):
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence_level) * 100)

class AdvancedOptimizer:
    """Advanced portfolio optimizer with multiple methodologies"""
    
    def __init__(self):
        self.risk_free_rate = 0.03
    
    def optimize_portfolio(self, expected_returns, volatilities, method='mean_variance', **kwargs):
        """Main optimization function with multiple methods"""
        expected_returns = np.array(expected_returns)
        volatilities = np.array(volatilities)
        
        if method == 'mean_variance':
            return self._mean_variance_optimization(expected_returns, volatilities, **kwargs)
        elif method == 'risk_parity':
            return self._risk_parity_optimization(expected_returns, volatilities, **kwargs)
        elif method == 'black_litterman':
            return self._black_litterman_optimization(expected_returns, volatilities, **kwargs)
        elif method == 'equal_weight':
            return np.ones(len(expected_returns)) / len(expected_returns)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _mean_variance_optimization(self, expected_returns, volatilities, target_return=None, 
                                  illiquidity_penalty=0.02, constraints=None):
        """Mean-variance optimization with illiquidity penalty"""
        n_assets = len(expected_returns)
        
        # Default illiquidity scores (higher = more illiquid)
        illiquidity_scores = np.array([0.8, 0.6, 0.7, 0.9])[:n_assets]
        if len(illiquidity_scores) < n_assets:
            illiquidity_scores = np.pad(illiquidity_scores, (0, n_assets - len(illiquidity_scores)), constant_values=0.7)
        
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights**2, volatilities**2)  # Simplified covariance
            illiquidity_cost = illiquidity_penalty * np.dot(weights, illiquidity_scores)
            
            if target_return is None:
                # Maximize Sharpe ratio
                sharpe = (portfolio_return - self.risk_free_rate) / np.sqrt(portfolio_variance)
                return -sharpe + illiquidity_cost
            else:
                # Minimize variance subject to target return
                return portfolio_variance + illiquidity_cost
        
        # Constraints
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        
        if target_return is not None:
            constraints_list.append({
                'type': 'eq', 
                'fun': lambda x: np.dot(x, expected_returns) - target_return
            })
        
        bounds = [(0, 1) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints_list=constraints_list)
            return result.x if result.success else x0
        except:
            return x0
    
    def _risk_parity_optimization(self, expected_returns, volatilities, **kwargs):
        """Risk parity optimization - equal risk contribution"""
        n_assets = len(expected_returns)
        
        # Create simplified covariance matrix (diagonal with some correlation)
        correlation = kwargs.get('correlation', 0.3)
        covariance_matrix = np.full((n_assets, n_assets), correlation)
        np.fill_diagonal(covariance_matrix, 1.0)
        
        # Scale by volatilities
        vol_matrix = np.outer(volatilities, volatilities)
        covariance_matrix = covariance_matrix * vol_matrix
        
        def risk_budget_objective(weights):
            """Minimize sum of squared differences in risk contributions"""
            weights = np.maximum(weights, 1e-8)  # Avoid division by zero
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            
            if portfolio_vol < 1e-8:
                return 1e6
            
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Target equal risk contribution
            target_contrib = np.ones(n_assets) / n_assets * np.sum(contrib)
            return np.sum((contrib - target_contrib)**2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0.01, 0.8) for _ in range(n_assets)]  # Min 1%, max 80%
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(risk_budget_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            return result.x if result.success else x0
        except:
            return x0
    
    def _black_litterman_optimization(self, expected_returns, volatilities, 
                                    market_caps=None, views=None, tau=0.025, **kwargs):
        """Black-Litterman optimization"""
        n_assets = len(expected_returns)
        
        # Default market caps (equal if not provided)
        if market_caps is None:
            market_caps = np.ones(n_assets)
        market_caps = np.array(market_caps)
        
        # Equilibrium weights (market cap weighted)
        w_eq = market_caps / np.sum(market_caps)
        
        # Create covariance matrix
        correlation = kwargs.get('correlation', 0.3)
        covariance_matrix = np.full((n_assets, n_assets), correlation)
        np.fill_diagonal(covariance_matrix, 1.0)
        vol_matrix = np.outer(volatilities, volatilities)
        covariance_matrix = covariance_matrix * vol_matrix
        
        # Implied equilibrium returns (reverse optimization)
        risk_aversion = kwargs.get('risk_aversion', 3.0)
        pi = risk_aversion * np.dot(covariance_matrix, w_eq)
        
        if views is None or len(views) == 0:
            # No views - return equilibrium portfolio
            return w_eq
        
        # Process views
        n_views = len(views)
        P = np.zeros((n_views, n_assets))  # Picking matrix
        Q = np.zeros(n_views)  # View returns
        
        for i, (asset_idx, view_return) in enumerate(views.items()):
            if isinstance(asset_idx, int) and 0 <= asset_idx < n_assets:
                P[i, asset_idx] = 1.0
                Q[i] = view_return
        
        # View uncertainty matrix (simplified)
        omega = tau * np.dot(P, np.dot(covariance_matrix, P.T))
        if np.linalg.det(omega) == 0:
            omega += np.eye(n_views) * 1e-8
        
        # Black-Litterman formula
        try:
            tau_sigma = tau * covariance_matrix
            tau_sigma_inv = np.linalg.inv(tau_sigma)
            omega_inv = np.linalg.inv(omega)
            
            # New expected returns
            M1 = tau_sigma_inv + np.dot(P.T, np.dot(omega_inv, P))
            M2 = np.dot(tau_sigma_inv, pi) + np.dot(P.T, np.dot(omega_inv, Q))
            
            mu_bl = np.dot(np.linalg.inv(M1), M2)
            
            # New covariance matrix
            sigma_bl = np.linalg.inv(M1)
            
            # Optimize with Black-Litterman inputs
            def objective(weights):
                return np.dot(weights.T, np.dot(sigma_bl, weights))
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
            bounds = [(0, 1) for _ in range(n_assets)]
            
            result = minimize(objective, w_eq, method='SLSQP', bounds=bounds, constraints=constraints)
            return result.x if result.success else w_eq
            
        except np.linalg.LinAlgError:
            return w_eq
    
    def calculate_portfolio_metrics(self, weights, expected_returns, volatilities, correlation=0.3):
        """Calculate portfolio performance metrics"""
        weights = np.array(weights)
        expected_returns = np.array(expected_returns)
        volatilities = np.array(volatilities)
        
        # Portfolio return
        portfolio_return = np.dot(weights, expected_returns)
        
        # Portfolio risk (with correlation)
        n_assets = len(weights)
        covariance_matrix = np.full((n_assets, n_assets), correlation)
        np.fill_diagonal(covariance_matrix, 1.0)
        vol_matrix = np.outer(volatilities, volatilities)
        covariance_matrix = covariance_matrix * vol_matrix
        
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        # Risk contributions for risk parity analysis
        marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_risk if portfolio_risk > 0 else np.zeros(n_assets)
        risk_contrib = weights * marginal_contrib
        risk_contrib_pct = risk_contrib / np.sum(risk_contrib) if np.sum(risk_contrib) > 0 else np.zeros(n_assets)
        
        return {
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe': sharpe_ratio,
            'risk_contributions': risk_contrib_pct
        }
    
    def efficient_frontier(self, expected_returns, volatilities, num_points=20, correlation=0.3):
        """Generate efficient frontier points"""
        expected_returns = np.array(expected_returns)
        volatilities = np.array(volatilities)
        n_assets = len(expected_returns)
        
        # Create covariance matrix
        covariance_matrix = np.full((n_assets, n_assets), correlation)
        np.fill_diagonal(covariance_matrix, 1.0)
        vol_matrix = np.outer(volatilities, volatilities)
        covariance_matrix = covariance_matrix * vol_matrix
        
        # Range of target returns
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        target_returns = np.linspace(min_return * 0.8, max_return * 1.2, num_points)
        
        risks = []
        returns = []
        
        for target_return in target_returns:
            def objective(weights):
                return np.dot(weights.T, np.dot(covariance_matrix, weights))
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}
            ]
            
            bounds = [(0, 1) for _ in range(n_assets)]
            x0 = np.ones(n_assets) / n_assets
            
            try:
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                if result.success:
                    portfolio_return = np.dot(result.x, expected_returns)
                    portfolio_risk = np.sqrt(np.dot(result.x.T, np.dot(covariance_matrix, result.x)))
                    risks.append(portfolio_risk)
                    returns.append(portfolio_return)
            except:
                continue
        
        return np.array(risks), np.array(returns)

class BacktestVisualizer:
    """Visualization functions for backtesting results"""
    
    @staticmethod
    def create_performance_chart(backtest_results):
        """Create performance comparison chart"""
        results_df = backtest_results['results_df']
        
        fig = go.Figure()
        
        # Portfolio performance
        fig.add_trace(go.Scatter(
            x=results_df['Date'],
            y=results_df['Portfolio_Value'],
            name='Portfolio',
            line=dict(color='blue', width=3)
        ))
        
        # Benchmark performance
        fig.add_trace(go.Scatter(
            x=results_df['Date'],
            y=results_df['Benchmark_Value'],
            name='Benchmark',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Portfolio vs Benchmark Performance",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_drawdown_chart(backtest_results):
        """Create drawdown chart"""
        results_df = backtest_results['results_df']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=results_df['Date'],
            y=results_df['Drawdown'],
            fill='tonexty',
            name='Drawdown',
            line=dict(color='red'),
            fillcolor='rgba(255,0,0,0.3)'
        ))
        
        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            yaxis=dict(tickformat='.1%'),
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_weights_chart(backtest_results):
        """Create weights evolution chart"""
        weights_df = backtest_results['weights_df']
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, asset in enumerate(weights_df.columns):
            fig.add_trace(go.Scatter(
                x=weights_df.index,
                y=weights_df[asset],
                name=asset,
                stackgroup='one',
                line=dict(color=colors[i % len(colors)]),
                mode='lines'
            ))
        
        fig.update_layout(
            title="Portfolio Weights Evolution",
            xaxis_title="Date",
            yaxis_title="Weight",
            yaxis=dict(tickformat='.0%'),
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_rolling_metrics_chart(backtest_results, window=12):
        """Create rolling performance metrics chart"""
        results_df = backtest_results['results_df']
        
        # Calculate rolling metrics
        portfolio_returns = results_df['Portfolio_Return'].rolling(window=window)
        rolling_sharpe = (portfolio_returns.mean() * 12) / (portfolio_returns.std() * np.sqrt(12))
        rolling_vol = portfolio_returns.std() * np.sqrt(12)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility'),
            vertical_spacing=0.1
        )
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=rolling_sharpe,
                name='Rolling Sharpe',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Rolling Volatility
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=rolling_vol,
                name='Rolling Volatility',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"Rolling Performance Metrics ({window}-Month Window)",
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_strategy_comparison_chart(comparison_results):
        """Create strategy comparison chart"""
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (strategy_name, results) in enumerate(comparison_results.items()):
            results_df = results['results_df']
            
            fig.add_trace(go.Scatter(
                x=results_df['Date'],
                y=results_df['Portfolio_Value'],
                name=strategy_name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title="Strategy Comparison",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            height=500
        )
        
        return fig

class SimpleVisualizer:
    """Simple visualization functions"""
    
    @staticmethod
    def create_allocation_chart(allocations):
        """Create allocation pie chart"""
        labels = list(allocations.keys())
        values = list(allocations.values())
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            height=400
        )
        return fig
    
    @staticmethod
    def create_cashflow_chart(cashflows):
        """Create cash flow chart"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cash Flows Over Time', 'Cumulative Cash Flows'),
            vertical_spacing=0.1
        )
        
        # Cash flows over time
        fig.add_trace(
            go.Bar(
                x=cashflows['Date'],
                y=cashflows['Contributions'],
                name='Contributions',
                marker_color='red',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=cashflows['Date'],
                y=cashflows['Distributions'],
                name='Distributions',
                marker_color='green',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Cumulative cash flows
        fig.add_trace(
            go.Scatter(
                x=cashflows['Date'],
                y=cashflows['Contributions'].cumsum(),
                name='Cumulative Contributions',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=cashflows['Date'],
                y=cashflows['Distributions'].cumsum(),
                name='Cumulative Distributions',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title="Cash Flow Analysis")
        return fig
    
    @staticmethod
    def create_performance_chart(dates, nav_values):
        """Create performance chart"""
        # Generate benchmark
        benchmark_returns = np.random.normal(0.08/12, 0.12/np.sqrt(12), len(dates))
        benchmark_values = nav_values[0] * np.cumprod(1 + benchmark_returns)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=nav_values,
            name='Portfolio NAV',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_values,
            name='Benchmark',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Value ($M)",
            height=400
        )
        return fig
    
    @staticmethod
    def create_monte_carlo_chart(paths, final_values):
        """Create Monte Carlo visualization"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Sample Paths', 'Final Value Distribution'),
            column_widths=[0.6, 0.4]
        )
        
        # Sample paths
        time_steps = np.arange(paths.shape[1])
        for i in range(min(50, len(paths))):
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=paths[i],
                    mode='lines',
                    line=dict(width=1, color='lightblue'),
                    opacity=0.3,
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Mean path
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=np.mean(paths, axis=0),
                mode='lines',
                line=dict(width=3, color='red'),
                name='Mean Path'
            ),
            row=1, col=1
        )
        
        # Distribution
        fig.add_trace(
            go.Histogram(
                x=final_values,
                nbinsx=30,
                name='Final Values',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        fig.update_layout(title="Monte Carlo Results", height=500)
        return fig
    
    @staticmethod
    def create_efficient_frontier_chart(risks, returns, optimal_point=None):
        """Create efficient frontier visualization"""
        fig = go.Figure()
        
        if len(risks) > 0 and len(returns) > 0:
            # Efficient frontier
            fig.add_trace(go.Scatter(
                x=risks,
                y=returns,
                mode='lines+markers',
                name='Efficient Frontier',
                line=dict(color='blue', width=3),
                marker=dict(size=6)
            ))
            
            # Optimal portfolio point
            if optimal_point:
                fig.add_trace(go.Scatter(
                    x=[optimal_point['risk']],
                    y=[optimal_point['return']],
                    mode='markers',
                    name='Optimal Portfolio',
                    marker=dict(color='red', size=15, symbol='star')
                ))
        
        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Risk (Volatility)",
            yaxis_title="Expected Return",
            xaxis=dict(tickformat='.1%'),
            yaxis=dict(tickformat='.1%'),
            height=500
        )
        return fig
    
    @staticmethod
    def create_risk_contribution_chart(asset_names, risk_contributions):
        """Create risk contribution chart for risk parity analysis"""
        fig = go.Figure(data=[
            go.Bar(
                x=asset_names,
                y=risk_contributions,
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(asset_names)],
                text=[f"{rc:.1%}" for rc in risk_contributions],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Risk Contribution by Asset",
            xaxis_title="Asset Class",
            yaxis_title="Risk Contribution",
            yaxis=dict(tickformat='.1%'),
            height=400
        )
        return fig

def run_monte_carlo(num_sims=1000, expected_return=0.12, volatility=0.25, time_horizon=5):
    """Run Monte Carlo simulation"""
    np.random.seed(42)
    
    dt = 1/12
    n_steps = int(time_horizon / dt)
    
    paths = np.zeros((num_sims, n_steps + 1))
    paths[:, 0] = 100.0
    
    for i in range(n_steps):
        random_shocks = np.random.normal(0, 1, num_sims)
        drift = (expected_return - 0.5 * volatility**2) * dt
        diffusion = volatility * np.sqrt(dt) * random_shocks
        paths[:, i + 1] = paths[:, i] * np.exp(drift + diffusion)
    
    final_values = paths[:, -1]
    
    return {
        'paths': paths,
        'final_values': final_values,
        'mean': np.mean(final_values),
        'std': np.std(final_values),
        'percentiles': {
            '5th': np.percentile(final_values, 5),
            '25th': np.percentile(final_values, 25),
            '50th': np.percentile(final_values, 50),
            '75th': np.percentile(final_values, 75),
            '95th': np.percentile(final_values, 95)
        },
        'prob_loss': np.mean(final_values < 100)
    }

def main():
    st.title("🏦 Private Markets Portfolio Analyzer")
    st.markdown("*Comprehensive analysis tool for private equity, debt, and real estate investments*")
    
    # Initialize components
    data_gen = SimpleDataGenerator()
    analytics = SimpleAnalytics()
    optimizer = AdvancedOptimizer()
    visualizer = SimpleVisualizer()
    backtest_viz = BacktestVisualizer()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Portfolio Overview", "Return Analytics", "Monte Carlo Simulation", "Portfolio Optimization", "Historical Backtesting"]
    )
    
    if page == "Portfolio Overview":
        st.header("📈 Portfolio Overview")
        
        sample_data = data_gen.generate_sample_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Asset Allocation")
            fig = visualizer.create_allocation_chart(sample_data['allocations'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Key Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['Total AUM', 'YTD Return', 'Sharpe Ratio', 'Max Drawdown'],
                'Value': ['$2.5B', '12.4%', '1.85', '-8.2%']
            })
            st.dataframe(metrics_df, hide_index=True)
        
        st.subheader("Cash Flow Analysis")
        fig = visualizer.create_cashflow_chart(sample_data['cashflows'])
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Return Analytics":
        st.header("📊 Return Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            investment_amount = st.number_input("Initial Investment ($M)", value=100.0, min_value=1.0)
        with col2:
            time_horizon = st.slider("Time Horizon (Years)", min_value=1, max_value=15, value=7)
        
        # Generate sample data
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=time_horizon*12, freq='M')
        returns = np.random.normal(0.08/12, 0.15/np.sqrt(12), len(dates))
        nav_values = investment_amount * np.cumprod(1 + returns)
        
        # Calculate metrics
        twir = analytics.calculate_twir(returns)
        sharpe = analytics.calculate_sharpe_ratio(returns)
        max_dd = analytics.calculate_max_drawdown(returns)
        var_95 = analytics.calculate_var(returns, 0.95)
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Time-Weighted Return", f"{twir:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{max_dd:.2%}")
        with col4:
            st.metric("VaR (95%)", f"{var_95:.2%}")
        
        # Performance chart
        fig = visualizer.create_performance_chart(dates, nav_values)
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Monte Carlo Simulation":
        st.header("🎲 Monte Carlo Simulation")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            num_sims = st.slider("Simulations", 100, 5000, 1000)
        with col2:
            expected_return = st.slider("Expected Return", 0.05, 0.20, 0.12)
        with col3:
            volatility = st.slider("Volatility", 0.10, 0.40, 0.25)
        
        if st.button("Run Simulation"):
            with st.spinner("Running simulation..."):
                results = run_monte_carlo(num_sims, expected_return, volatility, 5)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Results")
                results_df = pd.DataFrame({
                    'Percentile': ['5th', '25th', '50th', '75th', '95th'],
                    'Value ($M)': [f"${results['percentiles'][p]:.1f}" for p in ['5th', '25th', '50th', '75th', '95th']]
                })
                st.dataframe(results_df, hide_index=True)
            
            with col2:
                st.subheader("Risk Metrics")
                st.metric("Expected Value", f"${results['mean']:.1f}M")
                st.metric("Standard Deviation", f"${results['std']:.1f}M")
                st.metric("Probability of Loss", f"{results['prob_loss']:.1%}")
            
            fig = visualizer.create_monte_carlo_chart(results['paths'], results['final_values'])
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Portfolio Optimization":
        st.header("⚖️ Advanced Portfolio Optimization")
        
        asset_classes = ['Private Equity', 'Private Debt', 'Real Estate', 'Infrastructure']
        
        # Optimization method selection
        st.subheader("Optimization Method")
        method = st.selectbox(
            "Choose optimization approach:",
            ["mean_variance", "risk_parity", "black_litterman", "equal_weight"],
            format_func=lambda x: {
                "mean_variance": "Mean-Variance Optimization",
                "risk_parity": "Risk Parity",
                "black_litterman": "Black-Litterman",
                "equal_weight": "Equal Weight"
            }[x]
        )
        
        st.subheader("Asset Parameters")
        expected_returns = []
        volatilities = []
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Expected Returns**")
            for i, asset in enumerate(asset_classes):
                ret = st.slider(f"{asset}", 0.05, 0.20, 0.08 + i*0.02, key=f"ret_{i}", format="%.1%")
                expected_returns.append(ret)
        
        with col2:
            st.write("**Volatilities**")
            for i, asset in enumerate(asset_classes):
                vol = st.slider(f"{asset}", 0.10, 0.40, 0.15 + i*0.03, key=f"vol_{i}", format="%.1%")
                volatilities.append(vol)
        
        # Method-specific parameters
        method_params = {}
        
        if method == "mean_variance":
            st.subheader("Mean-Variance Parameters")
            col1, col2 = st.columns(2)
            with col1:
                target_return = st.slider("Target Return (optional)", 0.08, 0.18, 0.12, format="%.1%")
                use_target = st.checkbox("Use target return constraint")
            with col2:
                illiquidity_penalty = st.slider("Illiquidity Penalty", 0.0, 0.05, 0.02, format="%.2%")
            
            if use_target:
                method_params['target_return'] = target_return
            method_params['illiquidity_penalty'] = illiquidity_penalty
        
        elif method == "risk_parity":
            st.subheader("Risk Parity Parameters")
            correlation = st.slider("Asset Correlation", 0.0, 0.8, 0.3, format="%.1%")
            method_params['correlation'] = correlation
        
        elif method == "black_litterman":
            st.subheader("Black-Litterman Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                # Market capitalizations (equilibrium weights)
                st.write("**Market Weights (Equilibrium)**")
                market_caps = []
                for i, asset in enumerate(asset_classes):
                    cap = st.slider(f"{asset} Market Weight", 0.1, 0.6, 0.25, key=f"cap_{i}", format="%.1%")
                    market_caps.append(cap)
                
                # Normalize market caps
                market_caps = np.array(market_caps)
                market_caps = market_caps / np.sum(market_caps)
                method_params['market_caps'] = market_caps
            
            with col2:
                st.write("**Investor Views**")
                views = {}
                
                # Allow user to set views on specific assets
                for i, asset in enumerate(asset_classes):
                    has_view = st.checkbox(f"View on {asset}", key=f"view_check_{i}")
                    if has_view:
                        view_return = st.slider(
                            f"{asset} Expected Return", 
                            0.05, 0.25, expected_returns[i], 
                            key=f"view_{i}", 
                            format="%.1%"
                        )
                        views[i] = view_return
                
                method_params['views'] = views
                
                tau = st.slider("Tau (Uncertainty)", 0.01, 0.1, 0.025, format="%.3f")
                correlation = st.slider("Asset Correlation", 0.0, 0.8, 0.3, format="%.1%", key="bl_corr")
                method_params['tau'] = tau
                method_params['correlation'] = correlation
        
        # Run optimization
        if st.button("Optimize Portfolio", type="primary"):
            with st.spinner("Optimizing portfolio..."):
                weights = optimizer.optimize_portfolio(expected_returns, volatilities, method=method, **method_params)
                metrics = optimizer.calculate_portfolio_metrics(weights, expected_returns, volatilities)
            
            # Display results
            st.subheader("Optimization Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Optimal Allocation**")
                weights_df = pd.DataFrame({
                    'Asset': asset_classes,
                    'Weight': [f"{w:.1%}" for w in weights]
                })
                st.dataframe(weights_df, hide_index=True)
            
            with col2:
                st.write("**Portfolio Metrics**")
                st.metric("Expected Return", f"{metrics['return']:.2%}")
                st.metric("Expected Risk", f"{metrics['risk']:.2%}")
                st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
            
            with col3:
                st.write("**Risk Contributions**")
                risk_contrib_df = pd.DataFrame({
                    'Asset': asset_classes,
                    'Risk Contrib': [f"{rc:.1%}" for rc in metrics['risk_contributions']]
                })
                st.dataframe(risk_contrib_df, hide_index=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Allocation pie chart
                allocation_dict = dict(zip(asset_classes, weights))
                fig_allocation = visualizer.create_allocation_chart(allocation_dict)
                st.plotly_chart(fig_allocation, use_container_width=True)
            
            with col2:
                # Risk contribution chart
                fig_risk = visualizer.create_risk_contribution_chart(asset_classes, metrics['risk_contributions'])
                st.plotly_chart(fig_risk, use_container_width=True)
            
            # Efficient Frontier (for mean-variance method)
            if method == "mean_variance":
                st.subheader("Efficient Frontier Analysis")
                
                with st.spinner("Generating efficient frontier..."):
                    risks, returns = optimizer.efficient_frontier(expected_returns, volatilities)
                    
                    optimal_point = {
                        'risk': metrics['risk'],
                        'return': metrics['return']
                    }
                    
                    fig_frontier = visualizer.create_efficient_frontier_chart(risks, returns, optimal_point)
                    st.plotly_chart(fig_frontier, use_container_width=True)
    
    elif page == "Historical Backtesting":
        st.header("📈 Historical Backtesting")
        st.markdown("*Validate optimization strategies using historical data simulation*")
        
        # Generate historical data
        if 'historical_data' not in st.session_state:
            with st.spinner("Generating historical market data..."):
                st.session_state.historical_data = HistoricalDataGenerator.generate_historical_returns()
        
        historical_data = st.session_state.historical_data
        
        # Backtest parameters
        st.subheader("Backtest Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            backtest_start = st.date_input(
                "Backtest Start Date",
                value=datetime(2018, 1, 1),
                min_value=datetime(2015, 1, 1),
                max_value=datetime(2022, 1, 1)
            )
            
            backtest_end = st.date_input(
                "Backtest End Date", 
                value=datetime(2024, 1, 1),
                min_value=datetime(2019, 1, 1),
                max_value=datetime(2024, 12, 31)
            )
        
        with col2:
            rebalance_freq = st.selectbox(
                "Rebalancing Frequency",
                ["Q", "A"],
                format_func=lambda x: {"Q": "Quarterly", "A": "Annual"}[x]
            )
            
            lookback_window = st.slider(
                "Lookback Window (Months)",
                min_value=12, max_value=60, value=36
            )
        
        with col3:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                value=1000000, min_value=100000, step=100000
            )
            
            min_history = st.slider(
                "Min History (Months)",
                min_value=12, max_value=36, value=24
            )
        
        # Strategy selection
        st.subheader("Strategy Selection")
        
        strategies_to_test = st.multiselect(
            "Select strategies to backtest:",
            ["Mean-Variance", "Risk Parity", "Black-Litterman", "Equal Weight"],
            default=["Mean-Variance", "Risk Parity", "Equal Weight"]
        )
        
        # Strategy parameters
        strategy_params = {}
        
        if "Mean-Variance" in strategies_to_test:
            with st.expander("Mean-Variance Parameters"):
                mv_illiquidity_penalty = st.slider(
                    "Illiquidity Penalty", 0.0, 0.05, 0.02, 
                    key="mv_illiq", format="%.2%"
                )
                strategy_params["Mean-Variance"] = ("mean_variance", {"illiquidity_penalty": mv_illiquidity_penalty})
        
        if "Risk Parity" in strategies_to_test:
            with st.expander("Risk Parity Parameters"):
                rp_correlation = st.slider(
                    "Asset Correlation", 0.0, 0.8, 0.3, 
                    key="rp_corr", format="%.1%"
                )
                strategy_params["Risk Parity"] = ("risk_parity", {"correlation": rp_correlation})
        
        if "Black-Litterman" in strategies_to_test:
            with st.expander("Black-Litterman Parameters"):
                bl_tau = st.slider(
                    "Tau (Uncertainty)", 0.01, 0.1, 0.025, 
                    key="bl_tau", format="%.3f"
                )
                bl_correlation = st.slider(
                    "Asset Correlation", 0.0, 0.8, 0.3, 
                    key="bl_corr", format="%.1%"
                )
                strategy_params["Black-Litterman"] = ("black_litterman", {
                    "tau": bl_tau, 
                    "correlation": bl_correlation
                })
        
        if "Equal Weight" in strategies_to_test:
            strategy_params["Equal Weight"] = ("equal_weight", {})
        
        # Run backtest
        if st.button("Run Backtest", type="primary"):
            if len(strategies_to_test) == 0:
                st.error("Please select at least one strategy to backtest.")
            else:
                # Filter historical data by date range
                mask = (historical_data.index >= pd.to_datetime(backtest_start)) & \
                       (historical_data.index <= pd.to_datetime(backtest_end))
                backtest_data = historical_data.loc[mask]
                
                # Remove Public Equity for optimization (keep as benchmark)
                optimization_data = backtest_data.drop('Public Equity', axis=1, errors='ignore')
                
                # Initialize backtest engine
                backtest_engine = BacktestEngine(optimization_data, rebalance_freq)
                
                with st.spinner("Running backtests... This may take a moment."):
                    # Run backtests for all strategies
                    comparison_results = backtest_engine.compare_strategies(
                        strategy_params,
                        lookback_window=lookback_window,
                        min_history=min_history,
                        initial_capital=initial_capital
                    )
                
                # Display results
                st.subheader("Backtest Results")
                
                # Performance comparison table
                st.write("**Performance Summary**")
                
                performance_data = []
                for strategy_name, results in comparison_results.items():
                    stats = results['performance_stats']
                    performance_data.append({
                        'Strategy': strategy_name,
                        'Annual Return': f"{stats['Portfolio Annual Return']:.2%}",
                        'Volatility': f"{stats['Portfolio Volatility']:.2%}",
                        'Sharpe Ratio': f"{stats['Portfolio Sharpe']:.2f}",
                        'Max Drawdown': f"{stats['Maximum Drawdown']:.2%}",
                        'Information Ratio': f"{stats['Information Ratio']:.2f}",
                        'Win Rate': f"{stats['Win Rate']:.1%}"
                    })
                
                performance_df = pd.DataFrame(performance_data)
                st.dataframe(performance_df, hide_index=True)
                
                # Performance charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Strategy comparison chart
                    fig_comparison = backtest_viz.create_strategy_comparison_chart(comparison_results)
                    st.plotly_chart(fig_comparison, use_container_width=True)
                
                with col2:
                    # Individual strategy performance (first strategy)
                    first_strategy = list(comparison_results.keys())[0]
                    fig_performance = backtest_viz.create_performance_chart(comparison_results[first_strategy])
                    st.plotly_chart(fig_performance, use_container_width=True)
                
                # Detailed analysis for each strategy
                for strategy_name, results in comparison_results.items():
                    with st.expander(f"Detailed Analysis: {strategy_name}"):
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Drawdown chart
                            fig_drawdown = backtest_viz.create_drawdown_chart(results)
                            st.plotly_chart(fig_drawdown, use_container_width=True)
                        
                        with col2:
                            # Weights evolution
                            fig_weights = backtest_viz.create_weights_chart(results)
                            st.plotly_chart(fig_weights, use_container_width=True)
                        
                        # Rolling metrics
                        fig_rolling = backtest_viz.create_rolling_metrics_chart(results)
                        st.plotly_chart(fig_rolling, use_container_width=True)
                        
                        # Detailed statistics
                        st.write("**Detailed Performance Statistics**")
                        stats = results['performance_stats']
                        
                        detailed_stats = pd.DataFrame([
                            {'Metric': 'Total Return', 'Value': f"{stats['Total Return']:.2%}"},
                            {'Metric': 'Benchmark Total Return', 'Value': f"{stats['Benchmark Total Return']:.2%}"},
                            {'Metric': 'Excess Return', 'Value': f"{stats['Excess Return']:.2%}"},
                            {'Metric': 'Tracking Error', 'Value': f"{stats['Tracking Error']:.2%}"},
                            {'Metric': 'Calmar Ratio', 'Value': f"{stats['Calmar Ratio']:.2f}"},
                            {'Metric': 'Win Rate', 'Value': f"{stats['Win Rate']:.1%}"}
                        ])
                        
                        st.dataframe(detailed_stats, hide_index=True)
                
                # Strategy insights
                st.subheader("Strategy Insights")
                
                # Find best performing strategy
                best_strategy = max(comparison_results.keys(), 
                                  key=lambda x: comparison_results[x]['performance_stats']['Portfolio Sharpe'])
                
                best_return = max(comparison_results.keys(),
                                key=lambda x: comparison_results[x]['performance_stats']['Portfolio Annual Return'])
                
                lowest_drawdown = min(comparison_results.keys(),
                                    key=lambda x: comparison_results[x]['performance_stats']['Maximum Drawdown'])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Best Sharpe Ratio",
                        best_strategy,
                        f"{comparison_results[best_strategy]['performance_stats']['Portfolio Sharpe']:.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Highest Return",
                        best_return,
                        f"{comparison_results[best_return]['performance_stats']['Portfolio Annual Return']:.2%}"
                    )
                
                with col3:
                    st.metric(
                        "Lowest Drawdown",
                        lowest_drawdown,
                        f"{comparison_results[lowest_drawdown]['performance_stats']['Maximum Drawdown']:.2%}"
                    )
                
                # Key takeaways
                st.info(f"""
                **Key Takeaways from Backtest:**
                
                • **Best Risk-Adjusted Performance**: {best_strategy} achieved the highest Sharpe ratio
                • **Highest Returns**: {best_return} generated the highest annual returns
                • **Most Conservative**: {lowest_drawdown} had the smallest maximum drawdown
                • **Backtest Period**: {backtest_start} to {backtest_end}
                • **Rebalancing**: {rebalance_freq}ly rebalancing with {lookback_window}-month lookback window
                
                *Note: Past performance does not guarantee future results. This analysis uses simulated historical data for demonstration purposes.*
                """)
                
                # Export results option
                if st.button("📊 Export Results to CSV"):
                    # Combine all results into exportable format
                    export_data = []
                    
                    for strategy_name, results in comparison_results.items():
                        results_df = results['results_df']
                        results_df['Strategy'] = strategy_name
                        export_data.append(results_df)
                    
                    combined_df = pd.concat(export_data, ignore_index=True)
                    csv = combined_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Backtest Results",
                        data=csv,
                        file_name=f"backtest_results_{backtest_start}_{backtest_end}.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
