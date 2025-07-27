"""
Data Ingestion Module
Handles data import from various sources including CSV, APIs, and databases
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import json

class DataIngestion:
    """
    Handles data ingestion from multiple sources for private markets analysis
    """
    
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'api']
        
    def load_csv_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            return pd.read_csv(file_path, parse_dates=True)
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")
    
    def load_api_data(self, api_url: str, headers: Optional[Dict] = None) -> Dict:
        """Load data from API endpoint"""
        try:
            response = requests.get(api_url, headers=headers or {})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ValueError(f"Error loading API data: {str(e)}")
    
    def generate_sample_data(self) -> Dict:
        """Generate sample private markets data for demonstration"""
        
        # Sample allocations
        allocations = {
            'Private Equity': 0.35,
            'Private Debt': 0.25,
            'Real Estate': 0.30,
            'Infrastructure': 0.10
        }
        
        # Sample cash flows
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='Q')
        cashflows = pd.DataFrame({
            'Date': dates,
            'Contributions': np.random.normal(50, 15, len(dates)),
            'Distributions': np.random.normal(30, 20, len(dates)),
            'Net_Cash_Flow': lambda x: x['Contributions'] - x['Distributions']
        })
        cashflows['Net_Cash_Flow'] = cashflows['Contributions'] - cashflows['Distributions']
        
        # Sample NAV data
        nav_data = pd.DataFrame({
            'Date': dates,
            'NAV': np.cumsum(np.random.normal(10, 5, len(dates))) + 1000,
            'Unrealized_Gains': np.random.normal(20, 10, len(dates))
        })
        
        return {
            'allocations': allocations,
            'cashflows': cashflows,
            'nav_data': nav_data
        }
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data quality and completeness"""
        required_columns = ['Date', 'Amount']
        
        if not all(col in data.columns for col in required_columns):
            return False
            
        if data.isnull().sum().sum() > len(data) * 0.1:  # More than 10% missing
            return False
            
        return True
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle missing values
        data = data.fillna(method='forward').fillna(0)
        
        # Ensure proper date formatting
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date')
        
        return data
