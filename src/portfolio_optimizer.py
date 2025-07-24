"""
Portfolio Optimization Module
Implements mean-variance optimization with illiquidity constraints
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
import warnings

class PortfolioOptimizer:
    """
    Portfolio optimization engine for private markets with illiquidity constraints
    """
    
    def __init__(self):
        self.risk_free_rate = 0.03
    
    def optimize_portfolio(self, expected_returns: List[float],
                          volatilities: List[float],
                          target_return: float = None,
                          illiquidity_penalty: float = 0.02,
                          constraints: Dict = None) -> np.ndarray:
        """
        Optimize portfolio weights using mean-variance with illiquidity penalty
        
        Args:
            expected_returns: Expected returns for each asset class
            volatilities: Volatilities for each asset class
            target_return: Target portfolio return (if None, maximize Sharpe ratio)
            illiquidity_penalty: Penalty for illiquid assets
            constraints: Additional constraints dictionary
        
        Returns:
            Optimal portfolio weights
        """
        
        n_assets = len(expected_returns)
        expected_returns = np.array(expected_returns)
        volatilities = np.array(volatilities)
        
        # Default illiquidity scores (higher = more illiquid)
        illiquidity_scores = np.array([0.8, 0.6, 0.7, 0.9])  # PE, PD, RE, Infra
        if len(illiquidity_scores) != n_assets:
            illiquidity_scores = np.ones(n_assets) * 0.7
        
        def objective(weights):
            """Objective function to minimize"""
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights**2, volatilities**2)  # Simplified - assumes no correlation
            
            # Add illiquidity penalty
            illiquidity_cost = illiquidity_penalty * np.dot(weights, illiquidity_scores)
            
            if target_return is None:
                # Maximize Sharpe ratio (minimize negative Sharpe)
                sharpe = (portfolio_return - self.risk_free_rate) / np.sqrt(portfolio_variance)
                return -sharpe + illiquidity_cost
            else:
                # Minimize variance subject to target return
                return portfolio_variance + illiquidity_cost
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]
        
        if target_return is not None:
            constraints_list.append({
                'type': 'eq', 
                'fun': lambda x: np.dot(x, expected_returns) - target_return
            })
        
        # Add custom constraints if provided
        if constraints:
            if 'min_weights' in constraints:
                for i, min_weight in enumerate(constraints['min_weights']):
                    constraints_list.append({
                        'type': 'ineq',
                        'fun': lambda x, idx=i, min_w=min_weight: x[idx] - min_w
                    })
            
            if 'max_weights' in constraints:
                for i, max_weight in enumerate(constraints['max_weights']):
                    constraints_list.append({
                        'type': 'ineq',
                        'fun': lambda x, idx=i, max_w=max_weight: max_w - x[idx]
                    })
        
        # Bounds (0 to 1 for each weight)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}")
            return x0  # Return equal weights if optimization fails
        
        return result.x
    
    def efficient_frontier(self, expected_returns: List[float],
                          covariance_matrix: np.ndarray,
                          num_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate efficient frontier
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            num_points: Number of points on the frontier
        
        Returns:
            Tuple of (risks, returns) for efficient frontier
        """
        
        expected_returns = np.array(expected_returns)
        n_assets = len(expected_returns)
        
        # Range of target returns
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        target_returns = np.linspace(min_return, max_return, num_points)
        
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
            
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                portfolio_return = np.dot(result.x, expected_returns)
                portfolio_risk = np.sqrt(np.dot(result.x.T, np.dot(covariance_matrix, result.x)))
                
                risks.append(portfolio_risk)
                returns.append(portfolio_return)
        
        return np.array(risks), np.array(returns)
    
    def black_litterman_optimization(self, market_caps: np.ndarray,
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   views: Dict = None,
                                   tau: float = 0.025) -> np.ndarray:
        """
        Black-Litterman portfolio optimization
        
        Args:
            market_caps: Market capitalizations (proxy for equilibrium weights)
            expected_returns: Historical expected returns
            covariance_matrix: Covariance matrix
            views: Investor views {asset_index: expected_return}
            tau: Scaling factor for uncertainty
        
        Returns:
            Optimal portfolio weights
        """
        
        # Equilibrium weights (market cap weighted)
        w_eq = market_caps / np.sum(market_caps)
        
        # Implied equilibrium returns
        risk_aversion = 3.0  # Typical risk aversion parameter
        pi = risk_aversion * np.dot(covariance_matrix, w_eq)
        
        if views is None:
            # No views - return equilibrium portfolio
            return w_eq
        
        # Process views
        n_assets = len(expected_returns)
        n_views = len(views)
        
        P = np.zeros((n_views, n_assets))  # Picking matrix
        Q = np.zeros(n_views)  # View returns
        
        for i, (asset_idx, view_return) in enumerate(views.items()):
            P[i, asset_idx] = 1.0
            Q[i] = view_return
        
        # View uncertainty (simplified)
        omega = tau * np.dot(P, np.dot(covariance_matrix, P.T))
        
        # Black-Litterman formula
        tau_sigma = tau * covariance_matrix
        
        # New expected returns
        M1 = np.linalg.inv(tau_sigma)
        M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
        M3 = np.dot(np.linalg.inv(tau_sigma), pi)
        M4 = np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
        
        mu_bl = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
        
        # New covariance matrix
        sigma_bl = np.linalg.inv(M1 + M2)
        
        # Optimize with Black-Litterman inputs
        def objective(weights):
            return np.dot(weights.T, np.dot(sigma_bl, weights))
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0, 1) for _ in range(n_assets)]
        x0 = w_eq
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else w_eq
    
    def risk_parity_optimization(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Risk parity portfolio optimization
        
        Args:
            covariance_matrix: Covariance matrix of returns
        
        Returns:
            Risk parity weights
        """
        
        n_assets = covariance_matrix.shape[0]
        
        def risk_budget_objective(weights):
            """Minimize sum of squared differences in risk contributions"""
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Target equal risk contribution
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((contrib - target_contrib)**2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0.01, 1) for _ in range(n_assets)]  # Minimum 1% allocation
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            risk_budget_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else x0
