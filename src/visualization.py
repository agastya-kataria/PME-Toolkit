"""
Visualization Module
Creates interactive charts and dashboards for private markets analysis
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import streamlit as st

class DashboardVisualizer:
    """
    Creates interactive visualizations for private markets analysis
    """
    
    def __init__(self):
        self.color_palette = {
            'Private Equity': '#1f77b4',
            'Private Debt': '#ff7f0e', 
            'Real Estate': '#2ca02c',
            'Infrastructure': '#d62728',
            'Public Markets': '#9467bd'
        }
    
    def create_allocation_chart(self, allocations: Dict[str, float]) -> go.Figure:
        """Create pie chart for asset allocation"""
        
        labels = list(allocations.keys())
        values = list(allocations.values())
        colors = [self.color_palette.get(label, '#636EFA') for label in labels]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            font=dict(size=12),
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_cashflow_chart(self, cashflows: pd.DataFrame) -> go.Figure:
        """Create cash flow waterfall chart"""
        
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
        cumulative_contributions = cashflows['Contributions'].cumsum()
        cumulative_distributions = cashflows['Distributions'].cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=cashflows['Date'],
                y=cumulative_contributions,
                name='Cumulative Contributions',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=cashflows['Date'],
                y=cumulative_distributions,
                name='Cumulative Distributions',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            title="Cash Flow Analysis",
            showlegend=True
        )
        
        return fig
    
    def create_performance_chart(self, dates: pd.DatetimeIndex, 
                               nav_values: np.ndarray) -> go.Figure:
        """Create performance chart with benchmarks"""
        
        # Generate benchmark data
        benchmark_returns = np.random.normal(0.08/12, 0.12/np.sqrt(12), len(dates))
        benchmark_values = nav_values[0] * np.cumprod(1 + benchmark_returns)
        
        fig = go.Figure()
        
        # Portfolio performance
        fig.add_trace(go.Scatter(
            x=dates,
            y=nav_values,
            name='Portfolio NAV',
            line=dict(color='blue', width=3)
        ))
        
        # Benchmark performance
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_values,
            name='Public Market Benchmark',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Value ($M)",
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def create_monte_carlo_chart(self, results: Dict) -> go.Figure:
        """Create Monte Carlo simulation visualization"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Simulation Paths (Sample)', 'Final Value Distribution'),
            column_widths=[0.6, 0.4]
        )
        
        # Sample paths (show only first 100 for clarity)
        paths = results['paths'][:100]
        time_steps = np.arange(paths.shape[1])
        
        for i in range(min(50, len(paths))):  # Show max 50 paths
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
        
        # Add mean path
        mean_path = np.mean(results['paths'], axis=0)
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=mean_path,
                mode='lines',
                line=dict(width=3, color='red'),
                name='Mean Path'
            ),
            row=1, col=1
        )
        
        # Final value distribution
        fig.add_trace(
            go.Histogram(
                x=results['final_values'],
                nbinsx=50,
                name='Final Values',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Add percentile lines
        percentiles = [5, 25, 50, 75, 95]
        colors = ['red', 'orange', 'green', 'orange', 'red']
        
        for p, color in zip(percentiles, colors):
            value = np.percentile(results['final_values'], p)
            fig.add_vline(
                x=value,
                line=dict(color=color, width=2, dash='dash'),
                annotation_text=f"{p}th percentile",
                row=1, col=2
            )
        
        fig.update_layout(
            title="Monte Carlo Simulation Results",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_risk_decomposition_chart(self) -> go.Figure:
        """Create risk decomposition chart"""
        
        # Sample risk decomposition data
        risk_factors = ['Market Risk', 'Credit Risk', 'Liquidity Risk', 'Operational Risk', 'Model Risk']
        contributions = [0.45, 0.25, 0.15, 0.10, 0.05]
        
        fig = go.Figure(data=[
            go.Bar(
                x=risk_factors,
                y=contributions,
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            )
        ])
        
        fig.update_layout(
            title="Risk Decomposition",
            xaxis_title="Risk Factor",
            yaxis_title="Risk Contribution",
            yaxis=dict(tickformat='.0%'),
            height=400
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix: np.ndarray,
                                 asset_names: List[str]) -> go.Figure:
        """Create correlation heatmap"""
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=asset_names,
            y=asset_names,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Asset Correlation Matrix",
            height=500
        )
        
        return fig
    
    def create_efficient_frontier(self, risks: np.ndarray, 
                                returns: np.ndarray,
                                optimal_portfolio: Optional[Dict] = None) -> go.Figure:
        """Create efficient frontier chart"""
        
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=risks,
            y=returns,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=3)
        ))
        
        # Optimal portfolio point
        if optimal_portfolio:
            fig.add_trace(go.Scatter(
                x=[optimal_portfolio['risk']],
                y=[optimal_portfolio['return']],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(color='red', size=12, symbol='star')
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
    
    def create_drawdown_chart(self, dates: pd.DatetimeIndex,
                            returns: np.ndarray) -> go.Figure:
        """Create drawdown chart"""
        
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cumulative Returns', 'Drawdown'),
            vertical_spacing=0.1
        )
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cumulative_returns,
                name='Cumulative Returns',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown,
                name='Drawdown',
                fill='tonexty',
                line=dict(color='red'),
                fillcolor='rgba(255,0,0,0.3)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Performance and Drawdown Analysis",
            height=600,
            showlegend=True
        )
        
        return fig
