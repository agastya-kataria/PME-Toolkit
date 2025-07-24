"""
Return Analytics Module
Implements various return calculation methodologies for private markets
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
from scipy.optimize import fsolve
import warnings

class ReturnAnalytics:
    """
    Comprehensive return analytics for private markets investments
    """
    
    def __init__(self):
        self.risk_free_rate = 0.03  # 3% risk-free rate
        
    def calculate_mwir(self, nav_values: np.ndarray, dates: pd.DatetimeIndex, 
                       cashflows: np.ndarray = None) -> float:
        """
        Calculate Money-Weighted Internal Rate of Return (MWIR)
        Also known as IRR - accounts for timing and size of cash flows
        """
        if cashflows is None:
            # Assume initial investment and final distribution
            cashflows = np.zeros(len(nav_values))
            cashflows[0] = -nav_values[0]  # Initial investment
            cashflows[-1] = nav_values[-1]  # Final value
        
        def npv(rate):
            time_periods = [(date - dates[0]).days / 365.25 for date in dates]
            return sum(cf / (1 + rate) ** t for cf, t in zip(cashflows, time_periods))
        
        try:
            irr = fsolve(npv, 0.1)[0]
            return irr
        except:
            return np.nan
    
    def calculate_twir(self, returns: np.ndarray) -> float:
        """
        Calculate Time-Weighted Rate of Return (TWIR)
        Geometric mean of period returns - eliminates impact of cash flow timing
        """
        if len(returns) == 0:
            return 0.0
        
        # Convert to annual return
        periods_per_year = 12  # Assuming monthly returns
        geometric_mean = np.prod(1 + returns) ** (periods_per_year / len(returns)) - 1
        return geometric_mean
    
    def calculate_pme(self, nav_values: np.ndarray, dates: pd.DatetimeIndex, 
                      benchmark_returns: np.ndarray = None) -> float:
        """
        Calculate Public Market Equivalent (PME)
        Compares private market performance to public market benchmark
        """
        if benchmark_returns is None:
            # Use sample S&P 500 returns
            benchmark_returns = np.random.normal(0.08/12, 0.15/np.sqrt(12), len(nav_values))
        
        # Calculate cumulative benchmark performance
        benchmark_cumulative = np.cumprod(1 + benchmark_returns)
        
        # PME = (Distributions + NAV) / (Contributions adjusted for benchmark performance)
        initial_investment = nav_values[0]
        final_value = nav_values[-1]
        benchmark_final = benchmark_cumulative[-1]
        
        pme = final_value / (initial_investment * benchmark_final)
        return pme
    
    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = np.mean(returns) - self.risk_free_rate / 12
        return excess_returns / np.std(returns) if np.std(returns) > 0 else 0
    
    def calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = np.mean(returns) - self.risk_free_rate / 12
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        return excess_returns / downside_std if downside_std > 0 else 0
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = self.calculate_twir(returns)
        max_dd = abs(self.calculate_max_drawdown(returns))
        return annual_return / max_dd if max_dd > 0 else 0
    
    def calculate_information_ratio(self, returns: np.ndarray, 
                                   benchmark_returns: np.ndarray) -> float:
        """Calculate Information ratio"""
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        return np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0
    
    def generate_performance_attribution(self, portfolio_returns: np.ndarray,
                                       asset_returns: Dict[str, np.ndarray],
                                       weights: Dict[str, float]) -> Dict:
        """Generate performance attribution analysis"""
        attribution = {}
        
        for asset, returns in asset_returns.items():
            weight = weights.get(asset, 0)
            contribution = weight * np.mean(returns)
            attribution[asset] = {
                'weight': weight,
                'return': np.mean(returns),
                'contribution': contribution
            }
        
        return attribution
