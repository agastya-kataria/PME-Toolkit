"""
Machine Learning Models for Private Markets Analysis
Advanced ML capabilities for return prediction, risk modeling, and factor analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ReturnPredictionModel:
    """
    Advanced ML model for predicting private markets returns
    Uses ensemble methods and feature engineering
    """
    
    def __init__(self, model_type='ensemble'):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_fitted = False
        
    def create_features(self, data: pd.DataFrame, macro_data: pd.DataFrame = None) -> pd.DataFrame:
        """Create comprehensive feature set for ML models"""
        features = pd.DataFrame(index=data.index)
        
        # Lagged returns (momentum factors)
        for lag in [1, 3, 6, 12]:
            if f'return_lag_{lag}' not in data.columns:
                features[f'return_lag_{lag}'] = data['returns'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            features[f'return_mean_{window}m'] = data['returns'].rolling(window).mean()
            features[f'return_std_{window}m'] = data['returns'].rolling(window).std()
            features[f'return_skew_{window}m'] = data['returns'].rolling(window).skew()
            features[f'return_kurt_{window}m'] = data['returns'].rolling(window).kurt()
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['returns'])
        features['bollinger_position'] = self._calculate_bollinger_position(data['returns'])
        
        # Volatility regime
        features['vol_regime'] = self._identify_volatility_regime(data['returns'])
        
        # Market cycle indicators
        features['market_cycle'] = self._identify_market_cycle(data['returns'])
        
        # Macro features if available
        if macro_data is not None:
            features = features.join(macro_data, how='left')
        
        # Forward fill and drop NaN
        features = features.fillna(method='ffill').dropna()
        
        return features
    
    def _calculate_rsi(self, returns: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = returns.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_position(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        
        position = (returns - lower_band) / (upper_band - lower_band)
        return position.clip(0, 1)
    
    def _identify_volatility_regime(self, returns: pd.Series, window: int = 12) -> pd.Series:
        """Identify high/low volatility regimes"""
        rolling_vol = returns.rolling(window).std()
        vol_median = rolling_vol.median()
        return (rolling_vol > vol_median).astype(int)
    
    def _identify_market_cycle(self, returns: pd.Series, window: int = 24) -> pd.Series:
        """Identify market cycle phase"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        # Simple cycle identification: 0=recovery, 1=expansion, 2=peak, 3=contraction
        cycle = pd.Series(0, index=returns.index)
        cycle[drawdown > -0.05] = 1  # Expansion
        cycle[drawdown > -0.02] = 2  # Peak
        cycle[drawdown < -0.10] = 3  # Contraction
        
        return cycle
    
    def fit(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2):
        """Fit ensemble of ML models"""
        
        # Split data chronologically
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        X_train_std = self.scalers['standard'].fit_transform(X_train)
        X_train_rob = self.scalers['robust'].fit_transform(X_train)
        X_val_std = self.scalers['standard'].transform(X_val)
        X_val_rob = self.scalers['robust'].transform(X_val)
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1
            ),
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        
        # Fit models
        model_scores = {}
        
        # Tree-based models (use original features)
        for name in ['random_forest', 'xgboost', 'lightgbm']:
            self.models[name].fit(X_train, y_train)
            y_pred = self.models[name].predict(X_val)
            model_scores[name] = r2_score(y_val, y_pred)
            
            # Store feature importance
            if hasattr(self.models[name], 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X.columns, self.models[name].feature_importances_))
        
        # Linear models (use scaled features)
        for name in ['ridge', 'elastic_net']:
            self.models[name].fit(X_train_std, y_train)
            y_pred = self.models[name].predict(X_val_std)
            model_scores[name] = r2_score(y_val, y_pred)
        
        # Calculate ensemble weights based on validation performance
        total_score = sum(max(0, score) for score in model_scores.values())
        self.ensemble_weights = {
            name: max(0, score) / total_score if total_score > 0 else 1/len(model_scores)
            for name, score in model_scores.items()
        }
        
        self.is_fitted = True
        self.validation_scores = model_scores
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = {}
        
        # Tree-based models
        for name in ['random_forest', 'xgboost', 'lightgbm']:
            predictions[name] = self.models[name].predict(X)
        
        # Linear models (scaled)
        X_std = self.scalers['standard'].transform(X)
        for name in ['ridge', 'elastic_net']:
            predictions[name] = self.models[name].predict(X_std)
        
        # Ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += self.ensemble_weights[name] * pred
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, X: pd.DataFrame, n_bootstrap: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimation using bootstrap"""
        predictions = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_idx = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X.iloc[bootstrap_idx]
            
            pred = self.predict(X_bootstrap)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get aggregated feature importance"""
        if not self.feature_importance:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame(self.feature_importance).fillna(0)
        importance_df['mean_importance'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('mean_importance', ascending=False)
        
        return importance_df

class RiskFactorModel:
    """
    Factor model for risk decomposition and attribution
    Uses PCA and factor analysis
    """
    
    def __init__(self, n_factors: int = 5):
        self.n_factors = n_factors
        self.factor_loadings = None
        self.factor_returns = None
        self.specific_risk = None
        self.explained_variance = None
        self.is_fitted = False
    
    def fit(self, returns_data: pd.DataFrame):
        """Fit factor model using PCA"""
        from sklearn.decomposition import PCA
        
        # Standardize returns
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns_data.fillna(0))
        
        # Fit PCA
        pca = PCA(n_components=self.n_factors)
        factor_returns = pca.fit_transform(returns_scaled)
        
        # Store results
        self.factor_loadings = pd.DataFrame(
            pca.components_.T,
            index=returns_data.columns,
            columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )
        
        self.factor_returns = pd.DataFrame(
            factor_returns,
            index=returns_data.index,
            columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )
        
        self.explained_variance = pca.explained_variance_ratio_
        
        # Calculate specific risk (residual variance)
        reconstructed = factor_returns @ pca.components_
        residuals = returns_scaled - reconstructed
        self.specific_risk = pd.Series(
            np.var(residuals, axis=0),
            index=returns_data.columns
        )
        
        self.scaler = scaler
        self.pca = pca
        self.is_fitted = True
        
        return self
    
    def decompose_risk(self, portfolio_weights: np.ndarray) -> Dict[str, float]:
        """Decompose portfolio risk into factor and specific components"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Factor risk contribution
        factor_exposure = self.factor_loadings.T @ portfolio_weights
        factor_cov = np.cov(self.factor_returns.T)
        factor_risk = factor_exposure.T @ factor_cov @ factor_exposure
        
        # Specific risk contribution
        specific_risk_contrib = portfolio_weights.T @ np.diag(self.specific_risk) @ portfolio_weights
        
        total_risk = factor_risk + specific_risk_contrib
        
        return {
            'total_risk': float(total_risk),
            'factor_risk': float(factor_risk),
            'specific_risk': float(specific_risk_contrib),
            'factor_risk_pct': float(factor_risk / total_risk),
            'specific_risk_pct': float(specific_risk_contrib / total_risk)
        }
    
    def get_factor_exposures(self, portfolio_weights: np.ndarray) -> pd.Series:
        """Get portfolio's factor exposures"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        exposures = self.factor_loadings.T @ portfolio_weights
        return pd.Series(exposures, index=self.factor_loadings.columns)

class RegimeDetectionModel:
    """
    Hidden Markov Model for market regime detection
    Identifies bull/bear markets and volatility regimes
    """
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.regime_probs = None
        self.regime_params = None
        self.is_fitted = False
    
    def fit(self, returns: pd.Series):
        """Fit HMM for regime detection"""
        try:
            from hmmlearn import hmm
            
            # Prepare data (returns and volatility)
            returns_vol = np.column_stack([
                returns.values,
                returns.rolling(20).std().fillna(returns.std()).values
            ])
            
            # Fit Gaussian HMM
            model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                random_state=42
            )
            
            model.fit(returns_vol)
            
            # Predict regimes
            regime_probs = model.predict_proba(returns_vol)
            regimes = model.predict(returns_vol)
            
            self.regime_probs = pd.DataFrame(
                regime_probs,
                index=returns.index,
                columns=[f'Regime_{i}' for i in range(self.n_regimes)]
            )
            
            self.regimes = pd.Series(regimes, index=returns.index)
            self.model = model
            self.is_fitted = True
            
            # Calculate regime statistics
            self.regime_stats = {}
            for i in range(self.n_regimes):
                regime_returns = returns[regimes == i]
                self.regime_stats[f'Regime_{i}'] = {
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'frequency': len(regime_returns) / len(returns),
                    'avg_duration': self._calculate_avg_duration(regimes, i)
                }
            
        except ImportError:
            # Fallback to simple volatility-based regime detection
            self._simple_regime_detection(returns)
        
        return self
    
    def _simple_regime_detection(self, returns: pd.Series):
        """Simple regime detection based on volatility and returns"""
        rolling_vol = returns.rolling(20).std()
        rolling_ret = returns.rolling(20).mean()
        
        vol_threshold = rolling_vol.quantile(0.67)
        ret_threshold = rolling_ret.quantile(0.33)
        
        regimes = np.zeros(len(returns))
        regimes[(rolling_ret > ret_threshold) & (rolling_vol < vol_threshold)] = 0  # Bull/Low Vol
        regimes[(rolling_ret <= ret_threshold) & (rolling_vol < vol_threshold)] = 1  # Bear/Low Vol
        regimes[rolling_vol >= vol_threshold] = 2  # High Vol
        
        self.regimes = pd.Series(regimes, index=returns.index)
        self.is_fitted = True
    
    def _calculate_avg_duration(self, regimes: np.ndarray, regime: int) -> float:
        """Calculate average duration of regime"""
        regime_mask = (regimes == regime)
        durations = []
        current_duration = 0
        
        for is_regime in regime_mask:
            if is_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def predict_regime(self, current_data: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict current regime and probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self, 'model'):
            regime_probs = self.model.predict_proba(current_data.reshape(1, -1))[0]
            regime = np.argmax(regime_probs)
        else:
            # Simple fallback
            regime = int(self.regimes.iloc[-1])
            regime_probs = np.zeros(self.n_regimes)
            regime_probs[regime] = 1.0
        
        return regime, regime_probs

class MLPortfolioOptimizer:
    """
    ML-enhanced portfolio optimizer
    Uses predicted returns and risk models for optimization
    """
    
    def __init__(self):
        self.return_model = ReturnPredictionModel()
        self.risk_model = RiskFactorModel()
        self.regime_model = RegimeDetectionModel()
        self.is_fitted = False
    
    def fit(self, returns_data: pd.DataFrame, features_data: pd.DataFrame = None):
        """Fit all ML models"""
        
        # Prepare target variable (next period returns)
        target_returns = returns_data.shift(-1).dropna()
        
        if features_data is None:
            # Create features from returns data
            features_data = pd.DataFrame()
            for asset in returns_data.columns:
                asset_features = self.return_model.create_features(
                    pd.DataFrame({'returns': returns_data[asset]})
                )
                for col in asset_features.columns:
                    features_data[f"{asset}_{col}"] = asset_features[col]
        
        # Align data
        common_index = target_returns.index.intersection(features_data.index)
        target_aligned = target_returns.loc[common_index]
        features_aligned = features_data.loc[common_index]
        
        # Fit models for each asset
        self.asset_models = {}
        for asset in returns_data.columns:
            asset_target = target_aligned[asset]
            asset_features = features_aligned[[col for col in features_aligned.columns if asset in col]]
            
            if len(asset_features.columns) > 0:
                model = ReturnPredictionModel()
                model.fit(asset_features, asset_target)
                self.asset_models[asset] = model
        
        # Fit risk factor model
        self.risk_model.fit(returns_data)
        
        # Fit regime model (use first asset as proxy)
        first_asset = returns_data.columns[0]
        self.regime_model.fit(returns_data[first_asset])
        
        self.is_fitted = True
        return self
    
    def predict_returns(self, features_data: pd.DataFrame) -> pd.DataFrame:
        """Predict returns for all assets"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted first")
        
        predictions = {}
        uncertainties = {}
        
        for asset, model in self.asset_models.items():
            asset_features = features_data[[col for col in features_data.columns if asset in col]]
            if len(asset_features.columns) > 0:
                pred, uncertainty = model.predict_with_uncertainty(asset_features)
                predictions[asset] = pred
                uncertainties[asset] = uncertainty
        
        pred_df = pd.DataFrame(predictions, index=features_data.index)
        uncertainty_df = pd.DataFrame(uncertainties, index=features_data.index)
        
        return pred_df, uncertainty_df
    
    def optimize_portfolio_ml(self, features_data: pd.DataFrame, 
                             risk_aversion: float = 3.0) -> Dict[str, Any]:
        """Optimize portfolio using ML predictions"""
        
        # Get return predictions
        pred_returns, pred_uncertainty = self.predict_returns(features_data)
        
        # Use latest predictions
        expected_returns = pred_returns.iloc[-1].values
        return_uncertainty = pred_uncertainty.iloc[-1].values
        
        # Adjust for prediction uncertainty
        risk_adjusted_returns = expected_returns - risk_aversion * return_uncertainty
        
        # Get current regime
        current_regime, regime_probs = self.regime_model.predict_regime(
            np.array([expected_returns.mean(), expected_returns.std()])
        )
        
        # Regime-based risk adjustment
        regime_multiplier = 1.0
        if current_regime == 2:  # High volatility regime
            regime_multiplier = 1.5
        elif current_regime == 0:  # Bull market
            regime_multiplier = 0.8
        
        # Simple mean-variance optimization with ML inputs
        n_assets = len(expected_returns)
        
        # Create covariance matrix using factor model
        factor_cov = np.cov(self.risk_model.factor_returns.T)
        loadings = self.risk_model.factor_loadings.values
        
        # Factor covariance contribution
        factor_contrib = loadings @ factor_cov @ loadings.T
        
        # Add specific risk
        specific_risk_matrix = np.diag(self.risk_model.specific_risk.values)
        
        # Total covariance matrix
        cov_matrix = factor_contrib + specific_risk_matrix
        cov_matrix *= regime_multiplier  # Adjust for regime
        
        # Optimize using quadratic programming
        try:
            import cvxpy as cp
            
            weights = cp.Variable(n_assets)
            
            # Objective: maximize utility (return - risk penalty)
            portfolio_return = risk_adjusted_returns.T @ weights
            portfolio_risk = cp.quad_form(weights, cov_matrix)
            utility = portfolio_return - 0.5 * risk_aversion * portfolio_risk
            
            # Constraints
            constraints = [
                cp.sum(weights) == 1,  # Fully invested
                weights >= 0,  # Long only
                weights <= 0.4  # Max 40% in any asset
            ]
            
            # Solve
            problem = cp.Problem(cp.Maximize(utility), constraints)
            problem.solve()
            
            if weights.value is not None:
                optimal_weights = weights.value
            else:
                # Fallback to equal weights
                optimal_weights = np.ones(n_assets) / n_assets
                
        except ImportError:
            # Fallback to equal weights if cvxpy not available
            optimal_weights = np.ones(n_assets) / n_assets
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_risk = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
        
        # Risk decomposition
        risk_decomp = self.risk_model.decompose_risk(optimal_weights)
        factor_exposures = self.risk_model.get_factor_exposures(optimal_weights)
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk,
            'sharpe_ratio': portfolio_return / portfolio_risk if portfolio_risk > 0 else 0,
            'current_regime': current_regime,
            'regime_probabilities': regime_probs,
            'risk_decomposition': risk_decomp,
            'factor_exposures': factor_exposures,
            'return_predictions': pred_returns.iloc[-1],
            'prediction_uncertainty': pred_uncertainty.iloc[-1]
        }
