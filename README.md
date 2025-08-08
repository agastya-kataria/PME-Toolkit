# 📊 PM-Analyzer: Private Markets Portfolio Analyzer

A comprehensive Python-based tool for analyzing private equity, private debt, and real estate investments. This open-source framework provides institutional-grade analytics for private markets portfolios.

---

## 🚀 Features

### Core Analytics
- **Return Calculations**: Money-weighted IRR (MWIR), Time-weighted returns (TWIR), Public Market Equivalent (PME)
- **Risk Metrics**: VaR, CVaR, Maximum Drawdown, Sharpe & Sortino ratios
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
- **Real-time Visualization**: Interactive charts using Plotly / Streamlit
- **Performance Tracking**: NAV progression and benchmark comparisons
- **Risk Decomposition**: Visual breakdown of portfolio risk factors

---

## 📦 Quick Start

### Installation

```bash
git clone https://github.com/your-username/pm-analyzer.git
cd pm-analyzer
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run main.py
```

---

## 🧠 Basic Usage

```python
from src.data_ingestion import DataIngestion
from src.return_analytics import ReturnAnalytics
from src.monte_carlo import MonteCarloSimulator

# Initialize components
data_ingestion = DataIngestion()
analytics = ReturnAnalytics()
simulator = MonteCarloSimulator()

# Load sample data
sample_data = data_ingestion.generate_sample_data()

# Calculate MWIR
nav_values = sample_data['nav_data']['NAV'].values
dates = sample_data['nav_data']['Date']
mwir = analytics.calculate_mwir(nav_values, dates)

# Run Monte Carlo Simulation
results = simulator.run_simulation(
    num_simulations=1000,
    expected_return=0.12,
    volatility=0.25,
    time_horizon=5
)

print(f"Money-Weighted IRR: {mwir:.2%}")
print(f"Expected Final Value: ${results['statistics']['mean']:.1f}M")
```

---

## 🏗️ Architecture

### Project Structure

```text
pm-analyzer/
├── main.py                 # Streamlit app entry point
├── src/
│   ├── data_ingestion.py   # Data loading and preprocessing
│   ├── return_analytics.py # Return and risk calculations
│   ├── monte_carlo.py      # Stochastic modeling
│   ├── portfolio_optimizer.py # Optimization algorithms
│   └── visualization.py    # Interactive charts and dashboards
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── examples/               # Jupyter notebooks and demos
```

---

## 📈 Use Cases

### Investment Analysis
- Evaluate private fund performance
- Compare returns across asset classes
- Assess risk-adjusted metrics

### Portfolio Construction
- Optimize allocations across private markets
- Balance illiquidity and return objectives

### Risk Management
- Stress-test portfolios
- Track liquidity and concentration risks

### Reporting & Communication
- Generate reports and interactive visuals
- Benchmark against public markets

---
## 🖼️ Graphics and Screenshots

Here are graphics for each section of the app:

### 📌 Dashboard
![Dashboard Screenshot](screenshots/dashboard.png)

### 💼 Portfolio
![Portfolio Screenshot](screenshots/portfolio.png)

### 📈 Analytics
![Analytics Screenshot](screenshots/analytics.png)

### 📄 Reports
![Reports Screenshot](screenshots/reports.png)

### ⚙️ Settings
![Settings Screenshot](screenshots/settings.png)

---
## 🔧 Advanced Features

### Custom Analytics

```python
def custom_metric(returns, benchmark):
    excess_returns = returns - benchmark
    return np.mean(excess_returns) / np.std(excess_returns)
```

### API Integration

```python
data = data_ingestion.load_api_data(
    api_url="https://api.example.com/nav-data",
    headers={"Authorization": "Bearer your-token"}
)
```

### Batch Processing

```python
portfolios = ['Fund_A', 'Fund_B', 'Fund_C']
results = {}

for portfolio in portfolios:
    data = load_portfolio_data(portfolio)
    results[portfolio] = analytics.calculate_all_metrics(data)
```

---

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

---

## 📚 Documentation

Documentation is available in the `docs/` folder and includes:
- API Reference
- User Guide
- Example Notebooks

---

## 🤝 Contributing

We welcome contributions! Please read the `CONTRIBUTING.md` guide.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Format code
black src/ tests/

# Run lint checks
flake8 src/ tests/
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by institutional-grade private markets analytics platforms
- Built with modern Python data science stack
- Designed for extensibility and real-world use cases

---

## 📞 Contact

- **GitHub Issues**: [Submit here](https://github.com/agatya-kataria/pm-analyzer/issues)
- **Email**: agastyakataria176@gmail.com
- **LinkedIn**: [https://www.linkedin.com/in/agastyakataria176](https://www.linkedin.com/in/agastyakataria176)

> PM-Analyzer – Bringing institutional-grade private markets analytics to the open-source community.
