"""
Unit tests for portfolio optimization module
"""

import pytest
import numpy as np
from src.portfolio_optimizer import PortfolioOptimizer

class TestPortfolioOptimizer:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = PortfolioOptimizer()
        self.expected_returns = [0.08, 0.10, 0.12, 0.14]
        self.volatilities = [0.15, 0.18, 0.20, 0.25]
    
    def test_optimize_portfolio_basic(self):
        """Test basic portfolio optimization"""
        weights = self.optimizer.optimize_portfolio(
            self.expected_returns,
            self.volatilities
        )
        
        # Check that weights sum to 1
        assert abs(np.sum(weights) - 1.0) < 1e-6
        
        # Check that all weights are non-negative
        assert np.all(weights >= 0)
        
        # Check correct number of weights
        assert len(weights) == len(self.expected_returns)
    
    def test_optimize_portfolio_with_target_return(self):
        """Test optimization with target return constraint"""
        target_return = 0.11
        
        weights = self.optimizer.optimize_portfolio(
            self.expected_returns,
            self.volatilities,
            target_return=target_return
        )
        
        # Check that weights sum to 1
        assert abs(np.sum(weights) - 1.0) < 1e-6
        
        # Check that portfolio return matches target (approximately)
        portfolio_return = np.dot(weights, self.expected_returns)
        assert abs(portfolio_return - target_return) < 1e-3
    
    def test_optimize_portfolio_with_constraints(self):
        """Test optimization with additional constraints"""
        constraints = {
            'min_weights': [0.1, 0.0, 0.0, 0.0],  # Minimum 10% in first asset
            'max_weights': [0.5, 1.0, 1.0, 1.0]   # Maximum 50% in first asset
        }
        
        weights = self.optimizer.optimize_portfolio(
            self.expected_returns,
            self.volatilities,
            constraints=constraints
        )
        
        # Check constraints are satisfied
        assert weights[0] >= 0.1 - 1e-6  # Min constraint
        assert weights[0] <= 0.5 + 1e-6  # Max constraint
        assert abs(np.sum(weights) - 1.0) < 1e-6
    
    def test_efficient_frontier(self):
        """Test efficient frontier generation"""
        # Create a simple covariance matrix
        n_assets = len(self.expected_returns)
        covariance_matrix = np.diag(np.array(self.volatilities)**2)
        
        risks, returns = self.optimizer.efficient_frontier(
            self.expected_returns,
            covariance_matrix,
            num_points=10
        )
        
        # Check dimensions
        assert len(risks) <= 10  # May be fewer if optimization fails
        assert len(returns) <= 10
        assert len(risks) == len(returns)
        
        # Check that risks and returns are positive
        assert np.all(risks > 0)
        assert np.all(returns > 0)
    
    def test_risk_parity_optimization(self):
        """Test risk parity optimization"""
        # Create a simple covariance matrix
        covariance_matrix = np.array([
            [0.04, 0.01, 0.02, 0.015],
            [0.01, 0.09, 0.03, 0.02],
            [0.02, 0.03, 0.16, 0.04],
            [0.015, 0.02, 0.04, 0.25]
        ])
        
        weights = self.optimizer.risk_parity_optimization(covariance_matrix)
        
        # Check that weights sum to 1
        assert abs(np.sum(weights) - 1.0) < 1e-6
        
        # Check that all weights are positive (risk parity constraint)
        assert np.all(weights > 0)
        
        # Check correct number of weights
        assert len(weights) == covariance_matrix.shape[0]

if __name__ == "__main__":
    pytest.main([__file__])
