"""
Monte Carlo Simulation Module
Implements stochastic modeling for private markets forecasting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings

class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for private markets analysis
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed:
            np.random.seed(random_seed)
    
    def run_simulation(self, num_simulations: int = 1000, 
                      expected_return: float = 0.12,
                      volatility: float = 0.25,
                      time_horizon: int = 5,
                      initial_value: float = 100.0) -> Dict:
        """
        Run Monte Carlo simulation for NAV projections
        
        Args:
            num_simulations: Number of simulation paths
            expected_return: Expected annual return
            volatility: Annual volatility
            time_horizon: Investment horizon in years
            initial_value: Initial investment value
        
        Returns:
            Dictionary containing simulation results
        """
        
        # Time parameters
        dt = 1/12  # Monthly time steps
        n_steps = int(time_horizon / dt)
        
        # Initialize results arrays
        paths = np.zeros((num_simulations, n_steps + 1))
        paths[:, 0] = initial_value
        
        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (num_simulations, n_steps))
        
        # Simulate paths using geometric Brownian motion
        for i in range(n_steps):
            drift = (expected_return - 0.5 * volatility**2) * dt
            diffusion = volatility * np.sqrt(dt) * random_shocks[:, i]
            paths[:, i + 1] = paths[:, i] * np.exp(drift + diffusion)
        
        # Calculate final values and statistics
        final_values = paths[:, -1]
        
        results = {
            'paths': paths,
            'final_values': final_values,
            'statistics': {
                'mean': np.mean(final_values),
                'median': np.median(final_values),
                'std': np.std(final_values),
                'min': np.min(final_values),
                'max': np.max(final_values),
                'percentiles': {
                    '5th': np.percentile(final_values, 5),
                    '25th': np.percentile(final_values, 25),
                    '75th': np.percentile(final_values, 75),
                    '95th': np.percentile(final_values, 95)
                }
            },
            'risk_metrics': {
                'prob_loss': np.mean(final_values < initial_value),
                'expected_shortfall_5': np.mean(final_values[final_values <= np.percentile(final_values, 5)]),
                'var_95': np.percentile(final_values, 5) - initial_value
            }
        }
        
        return results
    
    def simulate_cashflows(self, num_simulations: int = 1000,
                          commitment: float = 100.0,
                          drawdown_schedule: List[float] = None,
                          distribution_schedule: List[float] = None) -> Dict:
        """
        Simulate cash flow patterns for private equity investments
        
        Args:
            num_simulations: Number of simulation paths
            commitment: Total commitment amount
            drawdown_schedule: Expected drawdown pattern (as % of commitment)
            distribution_schedule: Expected distribution pattern
        
        Returns:
            Dictionary containing cash flow simulation results
        """
        
        if drawdown_schedule is None:
            # Default J-curve pattern for PE
            drawdown_schedule = [0.2, 0.3, 0.25, 0.15, 0.1, 0.0, 0.0, 0.0]
        
        if distribution_schedule is None:
            # Default distribution pattern
            distribution_schedule = [0.0, 0.0, 0.1, 0.2, 0.3, 0.25, 0.15, 0.0]
        
        n_periods = max(len(drawdown_schedule), len(distribution_schedule))
        
        # Pad schedules to same length
        drawdown_schedule += [0.0] * (n_periods - len(drawdown_schedule))
        distribution_schedule += [0.0] * (n_periods - len(distribution_schedule))
        
        # Initialize results
        drawdowns = np.zeros((num_simulations, n_periods))
        distributions = np.zeros((num_simulations, n_periods))
        
        for sim in range(num_simulations):
            # Add randomness to schedules
            drawdown_noise = np.random.normal(1.0, 0.2, n_periods)
            distribution_noise = np.random.normal(1.0, 0.3, n_periods)
            
            # Apply noise and ensure non-negative
            sim_drawdowns = np.maximum(0, np.array(drawdown_schedule) * drawdown_noise)
            sim_distributions = np.maximum(0, np.array(distribution_schedule) * distribution_noise)
            
            # Normalize drawdowns to not exceed commitment
            if np.sum(sim_drawdowns) > 1.0:
                sim_drawdowns = sim_drawdowns / np.sum(sim_drawdowns)
            
            drawdowns[sim, :] = sim_drawdowns * commitment
            distributions[sim, :] = sim_distributions * commitment
        
        return {
            'drawdowns': drawdowns,
            'distributions': distributions,
            'net_cashflows': distributions - drawdowns,
            'cumulative_drawdowns': np.cumsum(drawdowns, axis=1),
            'cumulative_distributions': np.cumsum(distributions, axis=1)
        }
    
    def scenario_analysis(self, base_case: Dict, scenarios: Dict[str, Dict]) -> Dict:
        """
        Perform scenario analysis with different parameter sets
        
        Args:
            base_case: Base case parameters
            scenarios: Dictionary of scenario parameters
        
        Returns:
            Dictionary containing results for each scenario
        """
        
        results = {'base_case': self.run_simulation(**base_case)}
        
        for scenario_name, params in scenarios.items():
            # Merge base case with scenario-specific parameters
            scenario_params = {**base_case, **params}
            results[scenario_name] = self.run_simulation(**scenario_params)
        
        return results
    
    def sensitivity_analysis(self, base_params: Dict, 
                           sensitivity_params: Dict[str, List[float]]) -> Dict:
        """
        Perform sensitivity analysis by varying individual parameters
        
        Args:
            base_params: Base case parameters
            sensitivity_params: Parameters to vary and their ranges
        
        Returns:
            Dictionary containing sensitivity results
        """
        
        results = {}
        
        for param_name, param_values in sensitivity_params.items():
            param_results = []
            
            for value in param_values:
                test_params = base_params.copy()
                test_params[param_name] = value
                
                sim_result = self.run_simulation(**test_params)
                param_results.append({
                    'parameter_value': value,
                    'mean_final_value': sim_result['statistics']['mean'],
                    'prob_loss': sim_result['risk_metrics']['prob_loss'],
                    'var_95': sim_result['risk_metrics']['var_95']
                })
            
            results[param_name] = param_results
        
        return results
