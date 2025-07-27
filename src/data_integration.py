"""
Professional Data Integration Module
Real data sources, CSV upload, API integrations
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from typing import Dict, List, Optional, Tuple, Any
import io
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataIntegrationEngine:
    """
    Professional data integration with multiple sources
    """
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'json']
        self.api_cache = {}
        
    def upload_csv_data(self, uploaded_file) -> pd.DataFrame:
        """Process uploaded CSV file with validation"""
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Unsupported file format")
            
            # Validate and clean
            df = self._validate_and_clean_data(df)
            
            return df
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return pd.DataFrame()
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean uploaded data"""
        
        # Try to identify date column
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            date_col = date_columns[0]
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col).sort_index()
            except:
                pass
        
        # Convert numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    # Remove common formatting
                    df[col] = df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        # Remove rows with all NaN
        df = df.dropna(how='all')
        
        return df
    
    def fetch_benchmark_data(self, benchmark: str = '^GSPC', 
                           start_date: str = '2015-01-01',
                           end_date: str = None) -> pd.DataFrame:
        """Fetch benchmark data from Yahoo Finance"""
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        cache_key = f"{benchmark}_{start_date}_{end_date}"
        
        if cache_key in self.api_cache:
            return self.api_cache[cache_key]
        
        try:
            # Fetch data
            ticker = yf.Ticker(benchmark)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                st.warning(f"No data found for benchmark {benchmark}")
                return pd.DataFrame()
            
            # Calculate returns
            data['Returns'] = data['Close'].pct_change()
            data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()
            
            # Cache the result
            self.api_cache[cache_key] = data
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching benchmark data: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_benchmarks(self, benchmarks: Dict[str, str],
                                start_date: str = '2015-01-01') -> pd.DataFrame:
        """Fetch multiple benchmark indices"""
        
        benchmark_data = {}
        
        for name, ticker in benchmarks.items():
            data = self.fetch_benchmark_data(ticker, start_date)
            if not data.empty:
                benchmark_data[name] = data['Returns']
        
        if benchmark_data:
            return pd.DataFrame(benchmark_data).fillna(0)
        else:
            return pd.DataFrame()
    
    def fetch_macro_data(self) -> pd.DataFrame:
        """Fetch macroeconomic data"""
        
        # Common macro indicators
        macro_tickers = {
            'VIX': '^VIX',  # Volatility Index
            'DXY': 'DX-Y.NYB',  # Dollar Index
            'TNX': '^TNX',  # 10-Year Treasury
            'GOLD': 'GC=F'  # Gold Futures
        }
        
        macro_data = {}
        
        for name, ticker in macro_tickers.items():
            try:
                data = yf.Ticker(ticker).history(period='5y')
                if not data.empty:
                    macro_data[name] = data['Close'].pct_change()
            except:
                continue
        
        if macro_data:
            return pd.DataFrame(macro_data).fillna(0)
        else:
            return pd.DataFrame()
    
    def create_sample_fund_data(self, fund_type: str = 'PE') -> Dict[str, pd.DataFrame]:
        """Create realistic sample fund data based on actual patterns"""
        
        np.random.seed(42)  # For reproducibility
        
        if fund_type == 'PE':
            return self._create_pe_fund_data()
        elif fund_type == 'RE':
            return self._create_re_fund_data()
        elif fund_type == 'PD':
            return self._create_pd_fund_data()
        else:
            return self._create_pe_fund_data()
    
    def _create_pe_fund_data(self) -> Dict[str, pd.DataFrame]:
        """Create realistic PE fund data with J-curve"""
        
        # Fund details
        fund_info = {
            'Fund Name': 'Blackstone Capital Partners VIII',
            'Vintage Year': 2018,
            'Fund Size': 26000,  # $26B
            'Strategy': 'Large Buyout',
            'Geography': 'North America',
            'Status': 'Active'
        }
        
        # Create quarterly cash flows (typical 8-year fund life)
        dates = pd.date_range(start='2018-01-01', periods=32, freq='Q')
        
        # Realistic drawdown pattern (J-curve)
        drawdown_pattern = np.array([
            0.15, 0.25, 0.20, 0.15, 0.12, 0.08, 0.05, 0.00,  # Years 1-2
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # Years 3-4
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # Years 5-6
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00   # Years 7-8
        ])
        
        # Realistic distribution pattern
        distribution_pattern = np.array([
            0.00, 0.00, 0.00, 0.00, 0.02, 0.05, 0.08, 0.12,  # Years 1-2
            0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.25,  # Years 3-4
            0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03,  # Years 5-6
            0.02, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00   # Years 7-8
        ])
        
        # Add some randomness
        drawdown_noise = np.random.normal(1.0, 0.15, len(drawdown_pattern))
        distribution_noise = np.random.normal(1.0, 0.20, len(distribution_pattern))
        
        drawdowns = drawdown_pattern * drawdown_noise * fund_info['Fund Size']
        distributions = distribution_pattern * distribution_noise * fund_info['Fund Size']
        
        # Ensure non-negative and realistic totals
        drawdowns = np.maximum(drawdowns, 0)
        distributions = np.maximum(distributions, 0)
        
        # Scale to realistic totals
        total_drawdowns = np.sum(drawdowns)
        if total_drawdowns > fund_info['Fund Size']:
            drawdowns = drawdowns * (fund_info['Fund Size'] * 0.95) / total_drawdowns
        
        # NAV calculation (simplified)
        nav_values = []
        cumulative_invested = 0
        cumulative_distributed = 0
        
        for i in range(len(dates)):
            cumulative_invested += drawdowns[i]
            cumulative_distributed += distributions[i]
            
            # Unrealized value appreciation (varies by vintage and market conditions)
            if i < 8:  # First 2 years - minimal appreciation
                appreciation_factor = 1.0 + np.random.normal(0.02, 0.05)
            elif i < 16:  # Years 3-4 - moderate appreciation
                appreciation_factor = 1.0 + np.random.normal(0.08, 0.10)
            else:  # Later years - higher appreciation
                appreciation_factor = 1.0 + np.random.normal(0.12, 0.15)
            
            unrealized_value = (cumulative_invested - cumulative_distributed) * appreciation_factor
            nav = max(0, unrealized_value)
            nav_values.append(nav)
        
        # Create cash flow DataFrame
        cashflows = pd.DataFrame({
            'Date': dates,
            'Drawdowns': drawdowns,
            'Distributions': distributions,
            'Net_Cashflow': drawdowns - distributions,
            'Cumulative_Drawdowns': np.cumsum(drawdowns),
            'Cumulative_Distributions': np.cumsum(distributions),
            'NAV': nav_values
        })
        
        # Calculate performance metrics
        total_invested = np.sum(drawdowns)
        total_distributed = np.sum(distributions)
        final_nav = nav_values[-1]
        
        # IRR calculation (simplified)
        irr = self._calculate_irr(cashflows['Net_Cashflow'].values, final_nav)
        
        # Multiple calculation
        total_value = total_distributed + final_nav
        multiple = total_value / total_invested if total_invested > 0 else 0
        
        performance_metrics = pd.DataFrame({
            'Metric': ['IRR', 'Multiple', 'DPI', 'RVPI', 'TVPI'],
            'Value': [
                f"{irr:.1%}",
                f"{multiple:.2f}x",
                f"{total_distributed/total_invested:.2f}x" if total_invested > 0 else "0.00x",
                f"{final_nav/total_invested:.2f}x" if total_invested > 0 else "0.00x",
                f"{multiple:.2f}x"
            ]
        })
        
        return {
            'fund_info': fund_info,
            'cashflows': cashflows,
            'performance_metrics': performance_metrics
        }
    
    def _create_re_fund_data(self) -> Dict[str, pd.DataFrame]:
        """Create realistic Real Estate fund data"""
        
        fund_info = {
            'Fund Name': 'Brookfield Real Estate Partners V',
            'Vintage Year': 2019,
            'Fund Size': 15000,  # $15B
            'Strategy': 'Core Plus',
            'Geography': 'Global',
            'Status': 'Active'
        }
        
        # RE funds typically have more steady cash flows
        dates = pd.date_range(start='2019-01-01', periods=28, freq='Q')
        
        # More gradual drawdown pattern
        drawdown_pattern = np.array([
            0.12, 0.15, 0.18, 0.20, 0.15, 0.10, 0.05, 0.03,  # Years 1-2
            0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # Years 3-4
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # Years 5-6
            0.00, 0.00, 0.00, 0.00                            # Year 7
        ])
        
        # More steady distribution pattern (income + capital gains)
        distribution_pattern = np.array([
            0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.11,  # Years 1-2
            0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.22,  # Years 3-4
            0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04,  # Years 5-6
            0.02, 0.01, 0.01, 0.00                            # Year 7
        ])
        
        # Process similar to PE but with different patterns
        drawdown_noise = np.random.normal(1.0, 0.10, len(drawdown_pattern))
        distribution_noise = np.random.normal(1.0, 0.15, len(distribution_pattern))
        
        drawdowns = drawdown_pattern * drawdown_noise * fund_info['Fund Size']
        distributions = distribution_pattern * distribution_noise * fund_info['Fund Size']
        
        drawdowns = np.maximum(drawdowns, 0)
        distributions = np.maximum(distributions, 0)
        
        # Create similar structure as PE fund
        nav_values = []
        cumulative_invested = 0
        cumulative_distributed = 0
        
        for i in range(len(dates)):
            cumulative_invested += drawdowns[i]
            cumulative_distributed += distributions[i]
            
            # RE appreciation (typically more stable)
            appreciation_factor = 1.0 + np.random.normal(0.06, 0.08)
            unrealized_value = (cumulative_invested - cumulative_distributed) * appreciation_factor
            nav = max(0, unrealized_value)
            nav_values.append(nav)
        
        cashflows = pd.DataFrame({
            'Date': dates,
            'Drawdowns': drawdowns,
            'Distributions': distributions,
            'Net_Cashflow': drawdowns - distributions,
            'Cumulative_Drawdowns': np.cumsum(drawdowns),
            'Cumulative_Distributions': np.cumsum(distributions),
            'NAV': nav_values
        })
        
        # Calculate metrics
        total_invested = np.sum(drawdowns)
        total_distributed = np.sum(distributions)
        final_nav = nav_values[-1]
        
        irr = self._calculate_irr(cashflows['Net_Cashflow'].values, final_nav)
        multiple = (total_distributed + final_nav) / total_invested if total_invested > 0 else 0
        
        performance_metrics = pd.DataFrame({
            'Metric': ['IRR', 'Multiple', 'DPI', 'RVPI', 'TVPI'],
            'Value': [
                f"{irr:.1%}",
                f"{multiple:.2f}x",
                f"{total_distributed/total_invested:.2f}x" if total_invested > 0 else "0.00x",
                f"{final_nav/total_invested:.2f}x" if total_invested > 0 else "0.00x",
                f"{multiple:.2f}x"
            ]
        })
        
        return {
            'fund_info': fund_info,
            'cashflows': cashflows,
            'performance_metrics': performance_metrics
        }
    
    def _create_pd_fund_data(self) -> Dict[str, pd.DataFrame]:
        """Create realistic Private Debt fund data"""
        
        fund_info = {
            'Fund Name': 'Apollo Strategic Fund IV',
            'Vintage Year': 2020,
            'Fund Size': 8000,  # $8B
            'Strategy': 'Direct Lending',
            'Geography': 'North America',
            'Status': 'Active'
        }
        
        # PD funds have more predictable cash flows
        dates = pd.date_range(start='2020-01-01', periods=24, freq='Q')
        
        # Faster deployment, steady income
        drawdown_pattern = np.array([
            0.20, 0.25, 0.25, 0.20, 0.08, 0.02, 0.00, 0.00,  # Years 1-2
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # Years 3-4
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00   # Years 5-6
        ])
        
        # Steady distributions (interest + principal repayments)
        distribution_pattern = np.array([
            0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14,  # Years 1-2
            0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.24, 0.22,  # Years 3-4
            0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06   # Years 5-6
        ])
        
        # Less volatility in PD
        drawdown_noise = np.random.normal(1.0, 0.05, len(drawdown_pattern))
        distribution_noise = np.random.normal(1.0, 0.08, len(distribution_pattern))
        
        drawdowns = drawdown_pattern * drawdown_noise * fund_info['Fund Size']
        distributions = distribution_pattern * distribution_noise * fund_info['Fund Size']
        
        drawdowns = np.maximum(drawdowns, 0)
        distributions = np.maximum(distributions, 0)
        
        # NAV calculation (more stable for debt)
        nav_values = []
        cumulative_invested = 0
        cumulative_distributed = 0
        
        for i in range(len(dates)):
            cumulative_invested += drawdowns[i]
            cumulative_distributed += distributions[i]
            
            # Debt appreciation (minimal, mostly principal repayment)
            appreciation_factor = 1.0 + np.random.normal(0.01, 0.02)
            unrealized_value = (cumulative_invested - cumulative_distributed) * appreciation_factor
            nav = max(0, unrealized_value)
            nav_values.append(nav)
        
        cashflows = pd.DataFrame({
            'Date': dates,
            'Drawdowns': drawdowns,
            'Distributions': distributions,
            'Net_Cashflow': drawdowns - distributions,
            'Cumulative_Drawdowns': np.cumsum(drawdowns),
            'Cumulative_Distributions': np.cumsum(distributions),
            'NAV': nav_values
        })
        
        # Calculate metrics
        total_invested = np.sum(drawdowns)
        total_distributed = np.sum(distributions)
        final_nav = nav_values[-1]
        
        irr = self._calculate_irr(cashflows['Net_Cashflow'].values, final_nav)
        multiple = (total_distributed + final_nav) / total_invested if total_invested > 0 else 0
        
        performance_metrics = pd.DataFrame({
            'Metric': ['IRR', 'Multiple', 'DPI', 'RVPI', 'TVPI'],
            'Value': [
                f"{irr:.1%}",
                f"{multiple:.2f}x",
                f"{total_distributed/total_invested:.2f}x" if total_invested > 0 else "0.00x",
                f"{final_nav/total_invested:.2f}x" if total_invested > 0 else "0.00x",
                f"{multiple:.2f}x"
            ]
        })
        
        return {
            'fund_info': fund_info,
            'cashflows': cashflows,
            'performance_metrics': performance_metrics
        }
    
    def _calculate_irr(self, cashflows: np.ndarray, final_value: float) -> float:
        """Calculate IRR using Newton-Raphson method"""
        
        # Combine cashflows with final value
        all_cashflows = np.append(cashflows, final_value)
        
        # Initial guess
        irr = 0.1
        
        for _ in range(100):  # Max iterations
            # Calculate NPV and its derivative
            npv = 0
            npv_derivative = 0
            
            for i, cf in enumerate(all_cashflows):
                npv += cf / ((1 + irr) ** (i / 4))  # Quarterly periods
                npv_derivative -= (i / 4) * cf / ((1 + irr) ** (i / 4 + 1))
            
            # Newton-Raphson update
            if abs(npv_derivative) < 1e-10:
                break
                
            irr_new = irr - npv / npv_derivative
            
            if abs(irr_new - irr) < 1e-8:
                break
                
            irr = irr_new
        
        return irr

class BenchmarkEngine:
    """
    Professional benchmark comparison engine
    """
    
    def __init__(self):
        self.data_engine = DataIntegrationEngine()
        
        # Standard benchmarks for private markets
        self.benchmarks = {
            'Public Equity': {
                'S&P 500': '^GSPC',
                'MSCI World': 'URTH',
                'Russell 2000': '^RUT',
                'NASDAQ': '^IXIC'
            },
            'Fixed Income': {
                '10Y Treasury': '^TNX',
                'Investment Grade': 'LQD',
                'High Yield': 'HYG',
                'TIPS': 'SCHP'
            },
            'Alternatives': {
                'REITs': 'VNQ',
                'Commodities': 'DJP',
                'Gold': 'GLD',
                'Infrastructure': 'IGF'
            }
        }
    
    def calculate_pme_analysis(self, fund_cashflows: pd.DataFrame, 
                             benchmark_ticker: str = '^GSPC') -> Dict[str, Any]:
        """
        Calculate comprehensive PME analysis
        """
        
        # Get benchmark data
        start_date = fund_cashflows['Date'].min().strftime('%Y-%m-%d')
        end_date = fund_cashflows['Date'].max().strftime('%Y-%m-%d')
        
        benchmark_data = self.data_engine.fetch_benchmark_data(
            benchmark_ticker, start_date, end_date
        )
        
        if benchmark_data.empty:
            return {'error': 'Could not fetch benchmark data'}
        
        # Align dates
        benchmark_returns = benchmark_data['Returns'].reindex(
            fund_cashflows['Date'], method='nearest'
        ).fillna(0)
        
        # PME calculation
        pme_contributions = 0
        pme_distributions = 0
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        for i, row in fund_cashflows.iterrows():
            date = row['Date']
            contribution = row['Drawdowns']
            distribution = row['Distributions']
            
            if date in benchmark_cumulative.index:
                # Scale by benchmark performance from that date to end
                scaling_factor = benchmark_cumulative.iloc[-1] / benchmark_cumulative.loc[date]
                
                pme_contributions += contribution * scaling_factor
                pme_distributions += distribution * scaling_factor
        
        # Add final NAV
        final_nav = fund_cashflows['NAV'].iloc[-1]
        pme_distributions += final_nav
        
        # PME ratio
        pme_ratio = pme_distributions / pme_contributions if pme_contributions > 0 else 0
        
        # Additional metrics
        fund_irr = self.data_engine._calculate_irr(
            fund_cashflows['Net_Cashflow'].values,
            fund_cashflows['NAV'].iloc[-1]
        )
        
        benchmark_irr = (benchmark_cumulative.iloc[-1] ** (4 / len(benchmark_cumulative)) - 1)
        
        # Alpha calculation
        alpha = fund_irr - benchmark_irr
        
        return {
            'pme_ratio': pme_ratio,
            'fund_irr': fund_irr,
            'benchmark_irr': benchmark_irr,
            'alpha': alpha,
            'pme_contributions': pme_contributions,
            'pme_distributions': pme_distributions,
            'benchmark_name': benchmark_ticker,
            'analysis_period': f"{start_date} to {end_date}"
        }
    
    def stress_test_analysis(self, fund_cashflows: pd.DataFrame,
                           scenarios: Dict[str, Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive stress testing
        """
        
        if scenarios is None:
            scenarios = {
                'Base Case': {'return_shock': 0.0, 'timing_shock': 0.0},
                'Market Crash': {'return_shock': -0.30, 'timing_shock': 0.0},
                'Delayed Exits': {'return_shock': 0.0, 'timing_shock': 0.25},
                'Combined Stress': {'return_shock': -0.20, 'timing_shock': 0.20}
            }
        
        results = {}
        
        for scenario_name, params in scenarios.items():
            # Apply shocks
            stressed_cashflows = fund_cashflows.copy()
            
            # Return shock (affects distributions and NAV)
            return_shock = params.get('return_shock', 0.0)
            stressed_cashflows['Distributions'] *= (1 + return_shock)
            stressed_cashflows['NAV'] *= (1 + return_shock)
            
            # Timing shock (delays distributions)
            timing_shock = params.get('timing_shock', 0.0)
            if timing_shock > 0:
                # Shift distributions later
                shift_periods = int(len(stressed_cashflows) * timing_shock)
                distributions = stressed_cashflows['Distributions'].values
                distributions = np.concatenate([
                    np.zeros(shift_periods),
                    distributions[:-shift_periods]
                ])
                stressed_cashflows['Distributions'] = distributions
            
            # Recalculate metrics
            stressed_irr = self.data_engine._calculate_irr(
                stressed_cashflows['Drawdowns'].values - stressed_cashflows['Distributions'].values,
                stressed_cashflows['NAV'].iloc[-1]
            )
            
            total_invested = stressed_cashflows['Drawdowns'].sum()
            total_value = stressed_cashflows['Distributions'].sum() + stressed_cashflows['NAV'].iloc[-1]
            stressed_multiple = total_value / total_invested if total_invested > 0 else 0
            
            results[scenario_name] = {
                'irr': stressed_irr,
                'multiple': stressed_multiple,
                'total_value': total_value,
                'return_shock': return_shock,
                'timing_shock': timing_shock
            }
        
        return results
