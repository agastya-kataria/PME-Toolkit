"""
Unit tests for visualization module
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.visualization import DashboardVisualizer

class TestDashboardVisualizer:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.visualizer = DashboardVisualizer()
    
    def test_create_allocation_chart(self):
        """Test allocation chart creation"""
        allocations = {
            'Private Equity': 0.4,
            'Private Debt': 0.3,
            'Real Estate': 0.2,
            'Infrastructure': 0.1
        }
        
        fig = self.visualizer.create_allocation_chart(allocations)
        
        # Check that it returns a plotly figure
        assert isinstance(fig, go.Figure)
        
        # Check that it has data
        assert len(fig.data) > 0
        
        # Check that it's a pie chart
        assert isinstance(fig.data[0], go.Pie)
    
    def test_create_cashflow_chart(self):
        """Test cash flow chart creation"""
        dates = pd.date_range('2020-01-01', periods=20, freq='Q')
        cashflows = pd.DataFrame({
            'Date': dates,
            'Contributions': np.random.uniform(10, 50, len(dates)),
            'Distributions': np.random.uniform(5, 40, len(dates))
        })
        
        fig = self.visualizer.create_cashflow_chart(cashflows)
        
        # Check that it returns a plotly figure
        assert isinstance(fig, go.Figure)
        
        # Check that it has multiple traces (contributions, distributions, cumulative)
        assert len(fig.data) >= 4
    
    def test_create_performance_chart(self):
        """Test performance chart creation"""
        dates = pd.date_range('2020-01-01', periods=24, freq='M')
        nav_values = 100 * np.cumprod(1 + np.random.normal(0.01, 0.05, len(dates)))
        
        fig = self.visualizer.create_performance_chart(dates, nav_values)
        
        # Check that it returns a plotly figure
        assert isinstance(fig, go.Figure)
        
        # Check that it has at least 2 traces (portfolio and benchmark)
        assert len(fig.data) >= 2
    
    def test_create_monte_carlo_chart(self):
        """Test Monte Carlo chart creation"""
        # Mock Monte Carlo results
        num_sims = 100
        time_steps = 61  # 5 years monthly + initial
        
        results = {
            'paths': np.random.randn(num_sims, time_steps).cumsum(axis=1) + 100,
            'final_values': np.random.normal(120, 30, num_sims)
        }
        
        fig = self.visualizer.create_monte_carlo_chart(results)
        
        # Check that it returns a plotly figure
        assert isinstance(fig, go.Figure)
        
        # Check that it has multiple traces
        assert len(fig.data) > 1
    
    def test_create_risk_decomposition_chart(self):
        """Test risk decomposition chart creation"""
        fig = self.visualizer.create_risk_decomposition_chart()
        
        # Check that it returns a plotly figure
        assert isinstance(fig, go.Figure)
        
        # Check that it has data
        assert len(fig.data) > 0
        
        # Check that it's a bar chart
        assert isinstance(fig.data[0], go.Bar)
    
    def test_create_correlation_heatmap(self):
        """Test correlation heatmap creation"""
        correlation_matrix = np.array([
            [1.0, 0.3, 0.2, 0.1],
            [0.3, 1.0, 0.4, 0.2],
            [0.2, 0.4, 1.0, 0.3],
            [0.1, 0.2, 0.3, 1.0]
        ])
        asset_names = ['PE', 'PD', 'RE', 'Infra']
        
        fig = self.visualizer.create_correlation_heatmap(correlation_matrix, asset_names)
        
        # Check that it returns a plotly figure
        assert isinstance(fig, go.Figure)
        
        # Check that it has data
        assert len(fig.data) > 0
        
        # Check that it's a heatmap
        assert isinstance(fig.data[0], go.Heatmap)

if __name__ == "__main__":
    pytest.main([__file__])
