"""
Unit tests for Monte Carlo simulation module
"""

import pytest
import numpy as np
from src.monte_carlo import MonteCarloSimulator

class TestMonteCarloSimulator:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.simulator = MonteCarloSimulator(random_seed=42)
    
    def test_run_simulation_basic(self):
        """Test basic Monte Carlo simulation"""
        results = self.simulator.run_simulation(
            num_simulations=100,
            expected_return=0.10,
            volatility=0.20,
            time_horizon=1,
            initial_value=100.0
        )
        
        # Check structure
        assert 'paths' in results
        assert 'final_values' in results
        assert 'statistics' in results
        assert 'risk_metrics' in results
        
        # Check dimensions
        assert results['paths'].shape == (100, 13)  # 12 months + initial
        assert len(results['final_values']) == 100
        
        # Check statistics
        stats = results['statistics']
        assert
