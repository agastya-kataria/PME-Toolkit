"""
Advanced UI Components with Real-time Interactivity
Professional-grade user interface elements
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import io
from datetime import datetime, timedelta

class AdvancedUIComponents:
    """
    Professional UI components with real-time feedback
    """
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def create_file_uploader_section(self) -> Dict[str, Any]:
        """Create professional file upload section"""
        
        st.markdown("### 📁 Data Upload")
        st.markdown("Upload your fund data in CSV or Excel format")
        
        # Create tabs for different upload types
        upload_tab1, upload_tab2, upload_tab3 = st.tabs([
            "💼 Fund Cashflows", 
            "📊 Benchmark Data", 
            "🌍 Macro Data"
        ])
        
        uploaded_files = {}
        
        with upload_tab1:
            st.markdown("**Expected columns:** Date, Drawdowns, Distributions, NAV")
            fund_file = st.file_uploader(
                "Choose fund cashflow file",
                type=['csv', 'xlsx'],
                key='fund_upload',
                help="Upload historical fund cashflows with dates, contributions, distributions, and NAV values"
            )
            
            if fund_file:
                # Show preview
                try:
                    if fund_file.name.endswith('.csv'):
                        preview_df = pd.read_csv(fund_file).head()
                    else:
                        preview_df = pd.read_excel(fund_file).head()
                    
                    st.success("✅ File uploaded successfully!")
                    st.markdown("**Data Preview:**")
                    st.dataframe(preview_df, use_container_width=True)
                    
                    uploaded_files['fund_data'] = fund_file
                    
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        with upload_tab2:
            st.markdown("**Expected columns:** Date, Price/Returns")
            benchmark_file = st.file_uploader(
                "Choose benchmark data file",
                type=['csv', 'xlsx'],
                key='benchmark_upload',
                help="Upload benchmark index data for comparison"
            )
            
            if benchmark_file:
                uploaded_files['benchmark_data'] = benchmark_file
        
        with upload_tab3:
            st.markdown("**Expected columns:** Date, various macro indicators")
            macro_file = st.file_uploader(
                "Choose macro data file",
                type=['csv', 'xlsx'],
                key='macro_upload',
                help="Upload macroeconomic data for enhanced analysis"
            )
            
            if macro_file:
                uploaded_files['macro_data'] = macro_file
        
        return uploaded_files
    
    def create_interactive_parameter_panel(self) -> Dict[str, Any]:
        """Create interactive parameter adjustment panel"""
        
        st.markdown("### ⚙️ Analysis Parameters")
        
        # Create expandable sections
        with st.expander("🎯 Optimization Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                risk_aversion = st.slider(
                    "Risk Aversion",
                    min_value=1.0, max_value=10.0, value=3.0, step=0.5,
                    help="Higher values prefer lower risk portfolios"
                )
                
                rebalance_freq = st.selectbox(
                    "Rebalancing Frequency",
                    ["Monthly", "Quarterly", "Semi-Annual", "Annual"],
                    index=1,
                    help="How often to rebalance the portfolio"
                )
            
            with col2:
                lookback_window = st.slider(
                    "Lookback Window (Months)",
                    min_value=12, max_value=60, value=36,
                    help="Historical data window for optimization"
                )
                
                confidence_level = st.slider(
                    "Confidence Level",
                    min_value=0.90, max_value=0.99, value=0.95, step=0.01,
                    format="%.2f",
                    help="Confidence level for risk metrics"
                )
        
        with st.expander("🧪 Stress Testing", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                market_shock = st.slider(
                    "Market Shock (%)",
                    min_value=-50, max_value=0, value=-20,
                    help="Market return shock for stress testing"
                )
                
                liquidity_shock = st.slider(
                    "Liquidity Shock (%)",
                    min_value=0, max_value=50, value=20,
                    help="Increase in illiquidity for stress testing"
                )
            
            with col2:
                correlation_shock = st.slider(
                    "Correlation Shock",
                    min_value=0.0, max_value=1.0, value=0.3, step=0.1,
                    help="Increase in asset correlations during stress"
                )
                
                volatility_shock = st.slider(
                    "Volatility Shock (%)",
                    min_value=0, max_value=100, value=50,
                    help="Increase in volatility during stress"
                )
        
        with st.expander("🤖 Machine Learning Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                ml_model_type = st.selectbox(
                    "ML Model Type",
                    ["Ensemble", "Random Forest", "XGBoost", "Neural Network"],
                    help="Type of ML model for return prediction"
                )
                
                feature_importance_threshold = st.slider(
                    "Feature Importance Threshold",
                    min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                    format="%.2f",
                    help="Minimum feature importance to include"
                )
            
            with col2:
                prediction_horizon = st.selectbox(
                    "Prediction Horizon",
                    ["1 Month", "3 Months", "6 Months", "12 Months"],
                    index=1,
                    help="How far ahead to predict returns"
                )
                
                uncertainty_estimation = st.checkbox(
                    "Include Uncertainty Estimation",
                    value=True,
                    help="Estimate prediction uncertainty"
                )
        
        return {
            'risk_aversion': risk_aversion,
            'rebalance_freq': rebalance_freq,
            'lookback_window': lookback_window,
            'confidence_level': confidence_level,
            'market_shock': market_shock / 100,
            'liquidity_shock': liquidity_shock / 100,
            'correlation_shock': correlation_shock,
            'volatility_shock': volatility_shock / 100,
            'ml_model_type': ml_model_type.lower().replace(' ', '_'),
            'feature_importance_threshold': feature_importance_threshold,
            'prediction_horizon': prediction_horizon,
            'uncertainty_estimation': uncertainty_estimation
        }
    
    def create_real_time_dashboard(self, data: Dict[str, Any]) -> None:
        """Create real-time updating dashboard"""
        
        # Key metrics row
        st.markdown("### 📊 Real-Time Portfolio Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Portfolio Value",
                f"${data.get('portfolio_value', 0):,.0f}",
                delta=f"{data.get('value_change', 0):+.1%}",
                help="Current portfolio value with daily change"
            )
        
        with col2:
            st.metric(
                "YTD Return",
                f"{data.get('ytd_return', 0):.1%}",
                delta=f"{data.get('ytd_change', 0):+.1%}",
                help="Year-to-date return performance"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{data.get('sharpe_ratio', 0):.2f}",
                delta=f"{data.get('sharpe_change', 0):+.2f}",
                help="Risk-adjusted return measure"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{data.get('max_drawdown', 0):.1%}",
                delta=f"{data.get('drawdown_change', 0):+.1%}",
                delta_color="inverse",
                help="Maximum peak-to-trough decline"
            )
        
        with col5:
            st.metric(
                "VaR (95%)",
                f"{data.get('var_95', 0):.1%}",
                delta=f"{data.get('var_change', 0):+.1%}",
                delta_color="inverse",
                help="Value at Risk at 95% confidence"
            )
    
    def create_interactive_charts(self, data: Dict[str, Any]) -> None:
        """Create interactive charts with advanced features"""
        
        # Performance chart with benchmarks
        fig_performance = self._create_performance_chart_advanced(data)
        st.plotly_chart(fig_performance, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
        })
        
        # Risk-return scatter with regime overlay
        fig_risk_return = self._create_risk_return_scatter(data)
        st.plotly_chart(fig_risk_return, use_container_width=True)
        
        # Rolling metrics with regime detection
        fig_rolling = self._create_rolling_metrics_chart(data)
        st.plotly_chart(fig_rolling, use_container_width=True)
    
    def _create_performance_chart_advanced(self, data: Dict[str, Any]) -> go.Figure:
        """Create advanced performance chart with multiple features"""
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Portfolio Performance vs Benchmarks', 'Rolling Sharpe Ratio', 'Drawdown Analysis'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25],
            specs=[[{"secondary_y": True}], [{}], [{}]]
        )
        
        # Sample data for demonstration
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        # Portfolio performance
        portfolio_returns = np.random.normal(0.0008, 0.02, len(dates))
        portfolio_cumulative = (1 + portfolio_returns).cumprod() * 100
        
        # Benchmark performance
        benchmark_returns = np.random.normal(0.0006, 0.015, len(dates))
        benchmark_cumulative = (1 + benchmark_returns).cumprod() * 100
        
        # Main performance chart
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=portfolio_cumulative,
                name='Portfolio',
                line=dict(color='#1f77b4', width=3),
                hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=benchmark_cumulative,
                name='S&P 500',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                hovertemplate='<b>S&P 500</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add volume/activity indicator on secondary y-axis
        activity = np.random.uniform(0.5, 2.0, len(dates))
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=activity,
                name='Activity Level',
                line=dict(color='rgba(128,128,128,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                yaxis='y2',
                hovertemplate='<b>Activity</b><br>Date: %{x}<br>Level: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Rolling Sharpe ratio
        rolling_sharpe = pd.Series(portfolio_returns).rolling(252).mean() / pd.Series(portfolio_returns).rolling(252).std() * np.sqrt(252)
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_sharpe,
                name='Rolling Sharpe (1Y)',
                line=dict(color='#2ca02c', width=2),
                hovertemplate='<b>Sharpe Ratio</b><br>Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add Sharpe threshold line
        fig.add_hline(y=1.0, line_dash="dot", line_color="red", row=2, col=1,
                     annotation_text="Good Performance Threshold")
        
        # Drawdown analysis
        running_max = pd.Series(portfolio_cumulative).expanding().max()
        drawdown = (pd.Series(portfolio_cumulative) - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown,
                name='Drawdown',
                fill='tonexty',
                line=dict(color='#d62728'),
                fillcolor='rgba(214, 39, 40, 0.3)',
                hovertemplate='<b>Drawdown</b><br>Date: %{x}<br>Drawdown: %{y:.1%}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Comprehensive Performance Analysis",
            height=800,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Update x-axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        # Update y-axes
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Activity Level", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat='.1%', row=3, col=1)
        
        return fig
    
    def _create_risk_return_scatter(self, data: Dict[str, Any]) -> go.Figure:
        """Create risk-return scatter plot with regime overlay"""
        
        # Sample data for asset classes
        assets = ['Private Equity', 'Private Debt', 'Real Estate', 'Infrastructure', 'Public Equity']
        returns = [0.14, 0.10, 0.12, 0.11, 0.10]
        risks = [0.25, 0.15, 0.20, 0.18, 0.16]
        sharpe_ratios = [(r - 0.03) / v for r, v in zip(returns, risks)]
        
        # Create regime-based coloring
        regimes = ['Bull Market', 'Normal', 'Bear Market', 'Normal', 'Volatile']
        regime_colors = {'Bull Market': '#2ca02c', 'Normal': '#1f77b4', 'Bear Market': '#d62728', 'Volatile': '#ff7f0e'}
        colors = [regime_colors[regime] for regime in regimes]
        
        fig = go.Figure()
        
        # Add asset points
        fig.add_trace(go.Scatter(
            x=risks,
            y=returns,
            mode='markers+text',
            marker=dict(
                size=[s*20 + 10 for s in sharpe_ratios],  # Size based on Sharpe ratio
                color=colors,
                opacity=0.7,
                line=dict(width=2, color='white')
            ),
            text=assets,
            textposition="top center",
            textfont=dict(size=10, color='black'),
            hovertemplate='<b>%{text}</b><br>Risk: %{x:.1%}<br>Return: %{y:.1%}<br>Sharpe: %{marker.size}<extra></extra>',
            name='Asset Classes'
        ))
        
        # Add efficient frontier
        frontier_risks = np.linspace(0.10, 0.30, 50)
        frontier_returns = []
        
        for risk in frontier_risks:
            # Simple efficient frontier approximation
            max_return = 0.03 + (risk - 0.10) * 0.5 + 0.02 * np.sqrt(risk)
            frontier_returns.append(max_return)
        
        fig.add_trace(go.Scatter(
            x=frontier_risks,
            y=frontier_returns,
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='Efficient Frontier',
            hovertemplate='<b>Efficient Frontier</b><br>Risk: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>'
        ))
        
        # Add current portfolio point
        fig.add_trace(go.Scatter(
            x=[0.22],
            y=[0.13],
            mode='markers',
            marker=dict(
                size=20,
                color='gold',
                symbol='star',
                line=dict(width=3, color='black')
            ),
            name='Current Portfolio',
            hovertemplate='<b>Current Portfolio</b><br>Risk: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Risk-Return Analysis with Market Regimes",
            xaxis_title="Risk (Volatility)",
            yaxis_title="Expected Return",
            xaxis=dict(tickformat='.1%'),
            yaxis=dict(tickformat='.1%'),
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def _create_rolling_metrics_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create rolling metrics with regime detection"""
        
        # Sample data
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='M')
        np.random.seed(42)
        
        # Generate regime-aware metrics
        regimes = np.random.choice([0, 1, 2], size=len(dates), p=[0.4, 0.4, 0.2])  # Bull, Normal, Bear
        regime_names = ['Bull Market', 'Normal Market', 'Bear Market']
        regime_colors = ['#2ca02c', '#1f77b4', '#d62728']
        
        # Rolling Sharpe with regime influence
        base_sharpe = np.random.normal(1.2, 0.3, len(dates))
        regime_adjustment = np.array([0.5, 0.0, -0.8])[regimes]
        rolling_sharpe = base_sharpe + regime_adjustment
        
        # Rolling volatility
        base_vol = np.random.normal(0.15, 0.03, len(dates))
        vol_adjustment = np.array([-0.02, 0.0, 0.05])[regimes]
        rolling_vol = np.maximum(base_vol + vol_adjustment, 0.05)
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Market Regime Detection', 'Rolling Sharpe Ratio', 'Rolling Volatility'),
            vertical_spacing=0.1
        )
        
        # Market regime visualization
        for i, (regime_name, color) in enumerate(zip(regime_names, regime_colors)):
            regime_mask = regimes == i
            if np.any(regime_mask):
                fig.add_trace(
                    go.Scatter(
                        x=dates[regime_mask],
                        y=np.ones(np.sum(regime_mask)) * i,
                        mode='markers',
                        marker=dict(color=color, size=8),
                        name=regime_name,
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_sharpe,
                mode='lines',
                line=dict(color='#1f77b4', width=2),
                name='Rolling Sharpe',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add Sharpe benchmarks
        fig.add_hline(y=1.0, line_dash="dot", line_color="green", row=2, col=1)
        fig.add_hline(y=0.5, line_dash="dot", line_color="orange", row=2, col=1)
        
        # Rolling volatility
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_vol,
                mode='lines',
                line=dict(color='#d62728', width=2),
                fill='tonexty',
                fillcolor='rgba(214, 39, 40, 0.1)',
                name='Rolling Volatility',
                showlegend=False
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title="Rolling Performance Metrics with Regime Detection",
            height=700,
            template='plotly_white'
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Regime", row=1, col=1, tickvals=[0, 1, 2], ticktext=regime_names)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Volatility", tickformat='.1%', row=3, col=1)
        
        return fig
    
    def create_export_section(self, data: Dict[str, Any]) -> None:
        """Create professional export functionality"""
        
        st.markdown("### 📤 Export & Reporting")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export to Excel", use_container_width=True):
                excel_buffer = self._create_excel_report(data)
                st.download_button(
                    label="Download Excel Report",
                    data=excel_buffer,
                    file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                pdf_buffer = self._create_pdf_report(data)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
        
        with col3:
            if st.button("📈 Export Charts", use_container_width=True):
                charts_zip = self._create_charts_export(data)
                st.download_button(
                    label="Download Charts Package",
                    data=charts_zip,
                    file_name=f"portfolio_charts_{datetime.now().strftime('%Y%m%d')}.zip",
                    mime="application/zip"
                )
    
    def _create_excel_report(self, data: Dict[str, Any]) -> bytes:
        """Create comprehensive Excel report"""
        
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Portfolio Value', 'YTD Return', 'Sharpe Ratio', 'Max Drawdown', 'VaR (95%)'],
                'Value': ['$1,250,000', '12.4%', '1.85', '-8.2%', '-3.1%'],
                'Benchmark': ['$1,180,000', '10.1%', '1.62', '-12.1%', '-4.2%'],
                'Difference': ['+$70,000', '+2.3%', '+0.23', '+3.9%', '+1.1%']
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Performance data
            dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='M')
            performance_data = {
                'Date': dates,
                'Portfolio_Value': np.random.uniform(1000000, 1500000, len(dates)),
                'Benchmark_Value': np.random.uniform(950000, 1400000, len(dates)),
                'Portfolio_Return': np.random.normal(0.01, 0.03, len(dates)),
                'Benchmark_Return': np.random.normal(0.008, 0.025, len(dates))
            }
            performance_df = pd.DataFrame(performance_data)
            performance_df.to_excel(writer, sheet_name='Performance', index=False)
            
            # Risk metrics
            risk_data = {
                'Asset_Class': ['Private Equity', 'Private Debt', 'Real Estate', 'Infrastructure'],
                'Weight': [0.35, 0.25, 0.30, 0.10],
                'Expected_Return': [0.14, 0.10, 0.12, 0.11],
                'Volatility': [0.25, 0.15, 0.20, 0.18],
                'Sharpe_Ratio': [0.44, 0.47, 0.45, 0.44]
            }
            risk_df = pd.DataFrame(risk_data)
            risk_df.to_excel(writer, sheet_name='Risk_Analysis', index=False)
        
        output.seek(0)
        return output.getvalue()
    
    def _create_pdf_report(self, data: Dict[str, Any]) -> bytes:
        """Create PDF report (placeholder - would use reportlab in production)"""
        
        # This is a placeholder - in production, you'd use reportlab or similar
        report_content = f"""
        PORTFOLIO ANALYSIS REPORT
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        EXECUTIVE SUMMARY
        ================
        Portfolio Value: $1,250,000
        YTD Return: 12.4%
        Sharpe Ratio: 1.85
        Maximum Drawdown: -8.2%
        
        PERFORMANCE ANALYSIS
        ===================
        The portfolio has outperformed the benchmark by 2.3% year-to-date,
        demonstrating strong risk-adjusted returns with a Sharpe ratio of 1.85.
        
        RISK ANALYSIS
        =============
        Current portfolio exhibits well-diversified risk exposure across
        private markets asset classes with controlled maximum drawdown.
        
        RECOMMENDATIONS
        ===============
        1. Maintain current allocation strategy
        2. Monitor liquidity requirements
        3. Consider rebalancing in Q2 2024
        """
        
        return report_content.encode('utf-8')
    
    def _create_charts_export(self, data: Dict[str, Any]) -> bytes:
        """Create charts export package (placeholder)"""
        
        # This would create a ZIP file with all charts in production
        import zipfile
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('performance_chart.html', '<html>Performance Chart</html>')
            zip_file.writestr('risk_analysis.html', '<html>Risk Analysis Chart</html>')
            zip_file.writestr('allocation_chart.html', '<html>Allocation Chart</html>')
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    def create_scenario_comparison_table(self, scenarios: Dict[str, Dict]) -> None:
        """Create interactive scenario comparison table"""
        
        st.markdown("### 🎭 Scenario Analysis Comparison")
        
        # Convert scenarios to DataFrame
        scenario_data = []
        for scenario_name, metrics in scenarios.items():
            scenario_data.append({
                'Scenario': scenario_name,
                'Expected Return': f"{metrics.get('expected_return', 0):.2%}",
                'Volatility': f"{metrics.get('volatility', 0):.2%}",
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                'VaR (95%)': f"{metrics.get('var_95', 0):.2%}",
                'Probability of Loss': f"{metrics.get('prob_loss', 0):.1%}"
            })
        
        scenario_df = pd.DataFrame(scenario_data)
        
        # Style the dataframe
        styled_df = scenario_df.style.apply(self._highlight_best_scenario, axis=0)
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Add interpretation
        st.markdown("""
        **Interpretation Guide:**
        - 🟢 **Green**: Best performing metric
        - 🟡 **Yellow**: Moderate performance  
        - 🔴 **Red**: Concerning performance
        """)
    
    def _highlight_best_scenario(self, s):
        """Highlight best performing scenarios"""
        
        # This is a simplified version - in production you'd have more sophisticated highlighting
        return ['background-color: lightgreen' if 'Base Case' in str(val) else '' for val in s]

