"""
Unit tests for data ingestion module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_ingestion import DataIngestion

class TestDataIngestion:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.ingestion = DataIngestion()
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        sample_data = self.ingestion.generate_sample_data()
        
        # Check structure
        assert 'allocations' in sample_data
        assert 'cashflows' in sample_data
        assert 'nav_data' in sample_data
        
        # Check allocations
        allocations = sample_data['allocations']
        assert isinstance(allocations, dict)
        assert abs(sum(allocations.values()) - 1.0) < 1e-6  # Should sum to 1
        
        # Check cashflows DataFrame
        cashflows = sample_data['cashflows']
        assert isinstance(cashflows, pd.DataFrame)
        required_columns = ['Date', 'Contributions', 'Distributions', 'Net_Cash_Flow']
        assert all(col in cashflows.columns for col in required_columns)
        
        # Check NAV data
        nav_data = sample_data['nav_data']
        assert isinstance(nav_data, pd.DataFrame)
        assert 'Date' in nav_data.columns
        assert 'NAV' in nav_data.columns
    
    def test_validate_data_valid(self):
        """Test data validation with valid data"""
        valid_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10, freq='M'),
            'Amount': np.random.randn(10),
            'Other_Column': np.random.randn(10)
        })
        
        assert self.ingestion.validate_data(valid_data) == True
    
    def test_validate_data_missing_columns(self):
        """Test data validation with missing required columns"""
        invalid_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10, freq='M'),
            'Wrong_Column': np.random.randn(10)
        })
        
        assert self.ingestion.validate_data(invalid_data) == False
    
    def test_validate_data_too_many_nulls(self):
        """Test data validation with too many null values"""
        data_with_nulls = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10, freq='M'),
            'Amount': [np.nan] * 8 + [1.0, 2.0]  # 80% nulls
        })
        
        assert self.ingestion.validate_data(data_with_nulls) == False
    
    def test_clean_data(self):
        """Test data cleaning functionality"""
        dirty_data = pd.DataFrame({
            'Date': ['2020-01-01', '2020-02-01', '2020-02-01', '2020-03-01'],  # Duplicate
            'Amount': [100, np.nan, 200, 300],  # Missing value
            'Other': [1, 2, 2, 4]
        })
        
        cleaned_data = self.ingestion.clean_data(dirty_data)
        
        # Check duplicates removed
        assert len(cleaned_data) == 3
        
        # Check missing values handled
        assert not cleaned_data['Amount'].isnull().any()
        
        # Check date formatting
        assert pd.api.types.is_datetime64_any_dtype(cleaned_data['Date'])
        
        # Check sorting by date
        assert cleaned_data['Date'].is_monotonic_increasing

if __name__ == "__main__":
    pytest.main([__file__])
