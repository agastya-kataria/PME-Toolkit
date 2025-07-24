# PM-Analyzer: Private Markets Portfolio Analyzer

A comprehensive Python-based tool for analyzing private equity, private debt, and real estate investments. This open-source framework provides institutional-grade analytics for private markets portfolios.

## 🚀 Features

### Core Analytics
- **Return Calculations**: Money-weighted IRR (MWIR), Time-weighted returns (TWIR), Public Market Equivalent (PME)
- **Risk Metrics**: VaR, CVaR, Maximum Drawdown, Sharpe/Sortino ratios
- **Performance Attribution**: Detailed breakdown of portfolio performance drivers

### Advanced Modeling
- **Monte Carlo Simulation**: Stochastic NAV forecasting with customizable parameters
- **Cash Flow Modeling**: J-curve modeling for private equity investments
- **Scenario Analysis**: Stress testing under various market conditions

### Portfolio Optimization
- **Mean-Variance Optimization**: With illiquidity constraints and penalties
- **Risk Parity**: Equal risk contribution portfolios
- **Black-Litterman**: Bayesian approach incorporating investor views

### Interactive Dashboards
- **Real-time Visualization**: Interactive charts using Plotly/Streamlit
- **Performance Tracking**: NAV progression and benchmark comparisons
- **Risk Decomposition**: Visual breakdown of portfolio risk factors

## 📊 Quick Start

### Installation

\`\`\`bash
git clone https://github.com/your-username/pm-analyzer.git
cd pm-analyzer
pip install -r requirements.txt
\`\`\`

### Running the Application

\`\`\`bash
streamlit run main.py
\`\`\`

### Basic Usage

\`\`\`python
from src.data_ingestion import DataIngestion
from src.return_analytics import ReturnAnalytics
from src.monte_carlo import MonteCarloSimulator

# Initialize components
data_ingestion = DataIngestion()
analytics = ReturnAnalytics()
simulator = MonteCarloSimulator()

# Load sample data
sample_data = data_ingestion.generate_sample_data()

# Calculate returns
nav_values = sample_data['nav_data']['NAV'].values
dates = sample_data['nav_data']['Date']
mwir = analytics.calculate_mwir(nav_values, dates)

# Run Monte Carlo simulation
results = simulator.run_simulation(
    num_simulations=1000,
    expected_return=0.12,
    volatility=0.25,
    time_horizon=5
)

print(f"Money-Weighted IRR: {mwir:.2%}")
print(f"Expected Final Value: ${results['statistics']['mean']:.1f}M")
\`\`\`

## 🏗️ Architecture

### Project Structure
\`\`\`
pm-analyzer/
├── main.py                 # Streamlit application entry point
├── src/
│   ├── data_ingestion.py   # Data loading and preprocessing
│   ├── return_analytics.py # Return and risk calculations
│   ├── monte_carlo.py      # Stochastic modeling
│   ├── portfolio_optimizer.py # Optimization algorithms
│   └── visualization.py    # Interactive charts
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── examples/              # Jupyter notebooks
\`\`\`

### Key Components

1. **Data Ingestion**: Flexible data loading from CSV, APIs, and databases
2. **Return Analytics**: Comprehensive return and risk metric calculations
3. **Monte Carlo Engine**: Advanced stochastic modeling capabilities
4. **Portfolio Optimizer**: Multiple optimization methodologies
5. **Visualization**: Interactive dashboard components

## 📈 Use Cases

### Investment Analysis
- Evaluate private market fund performance
- Compare returns across asset classes
- Assess risk-adjusted performance metrics

### Portfolio Construction
- Optimize allocations across private markets
- Incorporate illiquidity constraints
- Balance risk and return objectives

### Risk Management
- Stress test portfolios under various scenarios
- Monitor concentration and correlation risks
- Track liquidity and cash flow requirements

### Reporting & Communication
- Generate institutional-quality reports
- Create interactive presentations
- Benchmark against public markets

## 🔧 Advanced Features

### Custom Analytics
\`\`\`python
# Custom return calculation
def custom_metric(returns, benchmark):
    excess_returns = returns - benchmark
    return np.mean(excess_returns) / np.std(excess_returns)
\`\`\`

### API Integration
\`\`\`python
# Load data from external APIs
data = data_ingestion.load_api_data(
    api_url="https://api.example.com/nav-data",
    headers={"Authorization": "Bearer your-token"}
)
\`\`\`

### Batch Processing
\`\`\`python
# Process multiple portfolios
portfolios = ['Fund_A', 'Fund_B', 'Fund_C']
results = {}

for portfolio in portfolios:
    data = load_portfolio_data(portfolio)
    results[portfolio] = analytics.calculate_all_metrics(data)
\`\`\`

## 🧪 Testing

Run the test suite:
\`\`\`bash
pytest tests/ -v
\`\`\`

## 📚 Documentation

Detailed documentation is available in the \`docs/\` directory:
- [API Reference](docs/api_reference.md)
- [User Guide](docs/user_guide.md)
- [Examples](examples/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
\`\`\`bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/
\`\`\`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by institutional private markets analytics platforms
- Built with modern Python data science stack
- Designed for extensibility and real-world application

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- GitHub Issues: [Create an issue](https://github.com/your-username/pm-analyzer/issues)
- Email: your-email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/your-profile)

---

**PM-Analyzer** - Bringing institutional-grade private markets analytics to the open-source community.
\`\`\`
\`\`\`

```python file="tests/test_return_analytics.py"
"""
Unit tests for return analytics module
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.return_analytics import ReturnAnalytics

class TestReturnAnalytics:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analytics = ReturnAnalytics()
        
        # Sample data
        self.dates = pd.date_range(start='2020-01-01', periods=12, freq='M')
        self.returns = np.array([0.01, -0.02, 0.03, 0.015, -0.01, 0.02, 
                                0.025, -0.005, 0.01, 0.02, -0.015, 0.03])
        self.nav_values = 100 * np.cumprod(1 + self.returns)
    
    def test_calculate_twir(self):
        """Test time-weighted return calculation"""
        twir = self.analytics.calculate_twir(self.returns)
        
        # Manual calculation for verification
        expected_twir = np.prod(1 + self.returns) - 1
        
        assert abs(twir - expected_twir) &lt; 1e-10
        assert isinstance(twir, float)
    
    def test_calculate_mwir(self):
        """Test money-weighted IRR calculation"""
        mwir = self.analytics.calculate_mwir(self.nav_values, self.dates)
        
        assert isinstance(mwir, (float, np.floating))
        assert not np.isnan(mwir)
        assert -1 &lt; mwir &lt; 5  # Reasonable range for IRR
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        sharpe = self.analytics.calculate_sharpe_ratio(self.returns)
        
        expected_excess_return = np.mean(self.returns) - self.analytics.risk_free_rate / 12
        expected_sharpe = expected_excess_return / np.std(self.returns)
        
        assert abs(sharpe - expected_sharpe) &lt; 1e-10
    
    def test_calculate_var(self):
        """Test Value at Risk calculation"""
        var_95 = self.analytics.calculate_var(self.returns, 0.95)
        var_99 = self.analytics.calculate_var(self.returns, 0.99)
        
        # VaR should be negative (loss)
        assert var_95 &lt; 0
        assert var_99 &lt; 0
        
        # 99% VaR should be more extreme than 95% VaR
        assert var_99 &lt;= var_95
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation"""
        max_dd = self.analytics.calculate_max_drawdown(self.returns)
        
        # Drawdown should be negative or zero
        assert max_dd &lt;= 0
        assert isinstance(max_dd, (float, np.floating))
    
    def test_empty_returns(self):
        """Test handling of empty returns array"""
        empty_returns = np.array([])
        
        twir = self.analytics.calculate_twir(empty_returns)
        assert twir == 0.0
        
        sharpe = self.analytics.calculate_sharpe_ratio(empty_returns)
        assert sharpe == 0.0
    
    def test_single_return(self):
        """Test handling of single return value"""
        single_return = np.array([0.05])
        
        twir = self.analytics.calculate_twir(single_return)
        assert abs(twir - 0.05) &lt; 1e-10
        
        # Sharpe ratio should be 0 when std is 0
        sharpe = self.analytics.calculate_sharpe_ratio(single_return)
        assert sharpe == 0.0

if __name__ == "__main__":
    pytest.main([__file__])
