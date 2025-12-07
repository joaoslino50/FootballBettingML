import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import joblib
import json
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, log_loss, classification_report, 
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV

from sklearn.base import clone

try:
    from uq360.algorithms.blackbox_metamodel import BlackboxMetamodelClassification
    UQ360_AVAILABLE = True
except ImportError:
    UQ360_AVAILABLE = False
    print("Warning: UQ360 not available, using bootstrap uncertainty estimation")


class UncertaintyQuantifier:
    """Wrapper for uncertainty quantification using UQ360 or bootstrap methods."""
    
    def __init__(self, base_model, n_bootstrap: int = 10, confidence_level: float = 0.9):
        self.base_model = base_model
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.bootstrap_models = []
        self.uq_model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit uncertainty quantification models."""
        if UQ360_AVAILABLE:
            try:
                self.uq_model = BlackboxMetamodelClassification(
                    base_model=self.base_model,
                    meta_model=RandomForestClassifier(n_estimators=50, random_state=42)
                )
                self.uq_model.fit(X, y)
            except Exception as e:
                print(f"  UQ360 fitting failed, using bootstrap: {e}")
                self._fit_bootstrap(X, y)
        else:
            self._fit_bootstrap(X, y)
        
        return self
    
    def _fit_bootstrap(self, X: np.ndarray, y: np.ndarray):
        """Fit bootstrap models for uncertainty estimation."""
        n_samples = X.shape[0]
        self.bootstrap_models = []
        
        for i in range(self.n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            try:
                model_copy = clone(self.base_model)
            except Exception:
                model_copy = RandomForestClassifier(n_estimators=100, random_state=42+i)
            model_copy.fit(X_boot, y_boot)
            self.bootstrap_models.append(model_copy)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.base_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.base_model.predict_proba(X)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Dict:
        """Predict with uncertainty intervals."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        if self.uq_model is not None and UQ360_AVAILABLE:
            try:
                _, uq_output = self.uq_model.predict(X)
                uncertainty = uq_output.get('uncertainty', None)
                if uncertainty is not None:
                    alpha = (1 - self.confidence_level) / 2
                    lower = np.clip(probabilities - uncertainty * 1.96, 0, 1)
                    upper = np.clip(probabilities + uncertainty * 1.96, 0, 1)
                    return {
                        'predictions': predictions,
                        'probabilities': probabilities,
                        'uncertainty': uncertainty,
                        'confidence_interval_lower': lower,
                        'confidence_interval_upper': upper
                    }
            except Exception:
                pass
        
        if self.bootstrap_models:
            all_probs = np.array([m.predict_proba(X) for m in self.bootstrap_models])
            alpha = (1 - self.confidence_level) / 2
            lower = np.percentile(all_probs, alpha * 100, axis=0)
            upper = np.percentile(all_probs, (1 - alpha) * 100, axis=0)
            uncertainty = np.std(all_probs, axis=0)
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'uncertainty': uncertainty,
                'confidence_interval_lower': lower,
                'confidence_interval_upper': upper
            }
        
        uncertainty = np.full_like(probabilities, 0.1)
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'uncertainty': uncertainty,
            'confidence_interval_lower': np.clip(probabilities - 0.1, 0, 1),
            'confidence_interval_upper': np.clip(probabilities + 0.1, 0, 1)
        }
    
    @property
    def classes_(self):
        return self.base_model.classes_
    
    def get_params(self, deep=True):
        return self.base_model.get_params(deep=deep)


class BettingModelTrainer:
    def __init__(self, data: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, fast_mode: bool = False):
        self.data = data.copy()
        self.test_size = test_size
        self.val_size = val_size
        self.fast_mode = fast_mode

        if 'MatchDate' in self.data.columns:
            self.data['MatchDate'] = pd.to_datetime(self.data['MatchDate'])
            self.data = self.data.sort_values('MatchDate').reset_index(drop=True)
        
        self.models = {}
        self.uq_models = {}
        self.scalers = {}
        self.feature_names = {}
        self.evaluation_results = {}    
        self.exclude_features = [
            'MatchDate', 'HomeTeam', 'AwayTeam', 'Division',
            'FTResult', 'FTHome', 'FTAway', 'TotalGoals', 'Over25',
            'GoalDifference', 'HighScoring'
        ]
    
    def prepare_targets(self) -> Dict[str, pd.Series]:
        targets = {}
        
        if 'FTResult' in self.data.columns:
            targets['match_result'] = self.data['FTResult'].copy()
        
        if 'TotalGoals' in self.data.columns:
            targets['over_under_25'] = (self.data['TotalGoals'] > 2.5).astype(int)
        elif 'FTHome' in self.data.columns and 'FTAway' in self.data.columns:
            targets['over_under_25'] = ((self.data['FTHome'] + self.data['FTAway']) > 2.5).astype(int)
        
        if 'FTHome' in self.data.columns and 'FTAway' in self.data.columns:
            targets['btts'] = (
                (self.data['FTHome'] > 0) & (self.data['FTAway'] > 0)
            ).astype(int)
        
        if 'FTAway' in self.data.columns:
            targets['clean_sheet_home'] = (self.data['FTAway'] == 0).astype(int)
        
        if 'FTHome' in self.data.columns:
            targets['clean_sheet_away'] = (self.data['FTHome'] == 0).astype(int)
        
        return targets
    
    def get_feature_columns(self) -> List[str]: 
        all_cols = self.data.columns.tolist()
        feature_cols = [
            col for col in all_cols 
            if col not in self.exclude_features
        ]
        
        numeric_cols = self.data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        return numeric_cols
    
    def temporal_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n_samples = len(self.data)
        test_start = int(n_samples * (1 - self.test_size))
        val_start = int(n_samples * (1 - self.test_size - self.val_size))
        
        train_df = self.data.iloc[:val_start].copy()
        val_df = self.data.iloc[val_start:test_start].copy()
        test_df = self.data.iloc[test_start:].copy()
        
        return train_df, val_df, test_df
    
    def prepare_features(self, df: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
        feature_cols = self.get_feature_columns()
        X = df[feature_cols].copy()
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]
        
        X = X.fillna(0)
        
        X = X.replace([np.inf, -np.inf], 0)
        
        X_array = X.values.astype(float)
        
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)
        else:
            X_scaled = scaler.transform(X_array)
        return X_scaled, scaler
    
    def train_match_result_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        print("Training Match Result model...")
        
        X_train, scaler = self.prepare_features(train_df)
        X_val, _ = self.prepare_features(val_df, scaler=scaler)
        
        y_train = train_df['FTResult'].values
        y_val = val_df['FTResult'].values
        
        n_estimators = 50 if self.fast_mode else 200
        max_depth = 10 if self.fast_mode else 15
        cv_folds = 2 if self.fast_mode else 3
        n_bootstrap = 5 if self.fast_mode else 10
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        calibrated_model = CalibratedClassifierCV(
            model, method='isotonic', cv=cv_folds
        )
        calibrated_model.fit(X_train, y_train)
        
        print("  Training uncertainty quantification model...")
        uq_model = UncertaintyQuantifier(calibrated_model, n_bootstrap=n_bootstrap)
        uq_model.fit(X_train, y_train)
        
        y_pred = calibrated_model.predict(X_val)
        y_proba = calibrated_model.predict_proba(X_val)
        
        uq_results = uq_model.predict_with_uncertainty(X_val)
        avg_uncertainty = np.mean(uq_results['uncertainty'])
        
        accuracy = accuracy_score(y_val, y_pred)
        logloss = log_loss(y_val, y_proba, labels=calibrated_model.classes_)
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Log Loss: {logloss:.4f}")
        print(f"  Avg Uncertainty: {avg_uncertainty:.4f}")
        
        self.models['match_result'] = calibrated_model
        self.uq_models['match_result'] = uq_model
        self.scalers['match_result'] = scaler
        self.feature_names['match_result'] = self.get_feature_columns()
        
        return {
            'model': calibrated_model,
            'uq_model': uq_model,
            'scaler': scaler,
            'accuracy': accuracy,
            'log_loss': logloss,
            'avg_uncertainty': avg_uncertainty,
            'classification_report': classification_report(y_val, y_pred)
        }
    
    def train_binary_model(self, market_name: str, target_col: str, 
                          train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        print(f"Training {market_name} model...")
        
        X_train, scaler = self.prepare_features(train_df)
        X_val, _ = self.prepare_features(val_df, scaler=scaler)
        
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        
        if len(np.unique(y_train)) < 2:
            print(f"  Warning: Insufficient class diversity in training set for {market_name}")
            return None
        
        n_estimators = 50 if self.fast_mode else 200
        max_depth = 3 if self.fast_mode else 5
        learning_rate = 0.1 if self.fast_mode else 0.05
        cv_folds = 2 if self.fast_mode else 3
        n_bootstrap = 5 if self.fast_mode else 10
        
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            subsample=0.8
        )
        
        model.fit(X_train, y_train)
        
        calibrated_model = CalibratedClassifierCV(
            model, method='isotonic', cv=cv_folds
        )
        calibrated_model.fit(X_train, y_train)
        
        print("  Training uncertainty quantification model...")
        uq_model = UncertaintyQuantifier(calibrated_model, n_bootstrap=n_bootstrap)
        uq_model.fit(X_train, y_train)
        
        y_pred = calibrated_model.predict(X_val)
        y_proba = calibrated_model.predict_proba(X_val)[:, 1]
        
        uq_results = uq_model.predict_with_uncertainty(X_val)
        avg_uncertainty = np.mean(uq_results['uncertainty'][:, 1]) if uq_results['uncertainty'].ndim > 1 else np.mean(uq_results['uncertainty'])
        
        accuracy = accuracy_score(y_val, y_pred)
        logloss = log_loss(y_val, y_proba)
        
        try:
            roc_auc = roc_auc_score(y_val, y_proba)
        except ValueError:
            roc_auc = None
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Log Loss: {logloss:.4f}")
        print(f"  Avg Uncertainty: {avg_uncertainty:.4f}")
        if roc_auc:
            print(f"  ROC AUC: {roc_auc:.4f}")
        
        self.models[market_name] = calibrated_model
        self.uq_models[market_name] = uq_model
        self.scalers[market_name] = scaler
        self.feature_names[market_name] = self.get_feature_columns()
        
        return {
            'model': calibrated_model,
            'uq_model': uq_model,
            'scaler': scaler,
            'accuracy': accuracy,
            'log_loss': logloss,
            'avg_uncertainty': avg_uncertainty,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_val, y_pred)
        }
    
    def train_all_models(self) -> Dict:
        print("Training ML Models for Betting Markets")
        
        targets = self.prepare_targets()
        
        train_df, val_df, test_df = self.temporal_split()
        
        print(f"\nData Split:")
        print(f"  Train: {len(train_df)} matches ({train_df['MatchDate'].min()} to {train_df['MatchDate'].max()})")
        print(f"  Validation: {len(val_df)} maqtches ({val_df['MatchDate'].min()} to {val_df['MatchDate'].max()})")
        print(f"  Test: {len(test_df)} matches ({test_df['MatchDate'].min()} to {test_df['MatchDate'].max()})")
        
        results = {}
        
        if 'match_result' in targets and 'FTResult' in train_df.columns:
            train_df['FTResult'] = targets['match_result'].loc[train_df.index]
            val_df['FTResult'] = targets['match_result'].loc[val_df.index]
            results['match_result'] = self.train_match_result_model(train_df, val_df)
        
        if 'over_under_25' in targets:
            train_df['over_under_25'] = targets['over_under_25'].loc[train_df.index]
            val_df['over_under_25'] = targets['over_under_25'].loc[val_df.index]
            results['over_under_25'] = self.train_binary_model(
                'over_under_25', 'over_under_25', train_df, val_df
            )
        
        if 'btts' in targets: 
            train_df['btts'] = targets['btts'].loc[train_df.index]
            val_df['btts'] = targets['btts'].loc[val_df.index]
            results['btts'] = self.train_binary_model( 
                'btts', 'btts', train_df, val_df
            )
        
        if 'clean_sheet_home' in targets:
            train_df['clean_sheet_home'] = targets['clean_sheet_home'].loc[train_df.index]
            val_df['clean_sheet_home'] = targets['clean_sheet_home'].loc[val_df.index]
            results['clean_sheet_home'] = self.train_binary_model(
                'clean_sheet_home', 'clean_sheet_home', train_df, val_df
            )
        
        if 'clean_sheet_away' in targets:
            train_df['clean_sheet_away'] = targets['clean_sheet_away'].loc[train_df.index]
            val_df['clean_sheet_away'] = targets['clean_sheet_away'].loc[val_df.index]
            results['clean_sheet_away'] = self.train_binary_model(
                'clean_sheet_away', 'clean_sheet_away', train_df, val_df
            )
        
        self.evaluation_results = results
        
        print("Model Training Complete!")
        
        return results
    
    def evaluate_on_test(self) -> Dict:
        print("Evaluating Models on Test Set")
        
        _, _, test_df = self.temporal_split()
        targets = self.prepare_targets()
        test_results = {}
        
        for market_name, model in self.models.items():
            print(f"\nEvaluating {market_name}...")
            
            X_test, _ = self.prepare_features(test_df, scaler=self.scalers[market_name])
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            if market_name == 'match_result':
                y_true = test_df['FTResult'].values
                accuracy = accuracy_score(y_true, y_pred)
                logloss = log_loss(y_true, y_proba, labels=model.classes_)
                print(f"  Test Accuracy: {accuracy:.4f}")
                print(f"  Test Log Loss: {logloss:.4f}")
            else:
                target_col = market_name
                if target_col not in test_df.columns and target_col in targets:
                    test_df[target_col] = targets[target_col].loc[test_df.index]
                
                if target_col in test_df.columns:
                    y_true = test_df[target_col].values
                    y_proba_binary = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
                    
                    accuracy = accuracy_score(y_true, y_pred)
                    logloss = log_loss(y_true, y_proba_binary)
                    
                    try:
                        roc_auc = roc_auc_score(y_true, y_proba_binary)
                    except ValueError:
                        roc_auc = None
                    
                    print(f"  Test Accuracy: {accuracy:.4f}")
                    print(f"  Test Log Loss: {logloss:.4f}")
                    if roc_auc:
                        print(f"  Test ROC AUC: {roc_auc:.4f}")
                else:
                    print(f"  Warning: Target column '{target_col}' not found in test set")
            
            test_results[market_name] = {
                'predictions': y_pred,
                'probabilities': y_proba,
                'accuracy': accuracy if 'accuracy' in locals() else None,
                'log_loss': logloss if 'logloss' in locals() else None,
                'roc_auc': roc_auc if 'roc_auc' in locals() else None
            }
        
        return test_results
    
    def save_models(self, output_dir: str = 'models'):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for market_name, model in self.models.items():
            model_path = output_path / f"{market_name}_model_{timestamp}.joblib"
            scaler_path = output_path / f"{market_name}_scaler_{timestamp}.joblib"
            features_path = output_path / f"{market_name}_features_{timestamp}.txt"
            uq_model_path = output_path / f"{market_name}_uq_model_{timestamp}.joblib"
            
            joblib.dump(model, model_path)
            joblib.dump(self.scalers[market_name], scaler_path)
            
            if market_name in self.uq_models:
                joblib.dump(self.uq_models[market_name], uq_model_path)
            
            with open(features_path, 'w') as f:
                f.write('\n'.join(self.feature_names[market_name]))
            
            print(f"Saved {market_name} model to {model_path}")
        
        metadata = {
            'timestamp': timestamp,
            'models': list(self.models.keys()),
            'has_uq_models': list(self.uq_models.keys()),
            'evaluation_results': self.evaluation_results
        }
        
        metadata_path = output_path / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\nSaved metadata to {metadata_path}")
    
    def predict(self, X: pd.DataFrame, market_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if market_name not in self.models:
            raise ValueError(f"Model {market_name} not found")
        
        X_prepared, _ = self.prepare_features(X, scaler=self.scalers[market_name])
        
        predictions = self.models[market_name].predict(X_prepared)
        probabilities = self.models[market_name].predict_proba(X_prepared)
        
        return predictions, probabilities
    
    def predict_with_uncertainty(self, X: pd.DataFrame, market_name: str) -> Dict:
        """Predict with uncertainty intervals using UQ360."""
        if market_name not in self.models:
            raise ValueError(f"Model {market_name} not found")
        
        X_prepared, _ = self.prepare_features(X, scaler=self.scalers[market_name])
        
        if market_name in self.uq_models:
            return self.uq_models[market_name].predict_with_uncertainty(X_prepared)
        else:
            predictions = self.models[market_name].predict(X_prepared)
            probabilities = self.models[market_name].predict_proba(X_prepared)
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'uncertainty': np.full_like(probabilities, 0.1),
                'confidence_interval_lower': np.clip(probabilities - 0.1, 0, 1),
                'confidence_interval_upper': np.clip(probabilities + 0.1, 0, 1)
            }


def main(sample_size: Optional[int] = None, fast_mode: bool = False, use_sparql: bool = False):   
    from feature_engineering import FeatureEngineer
    from data_preprocessing import DataPreprocessor
    
    if use_sparql:
        print("=" * 60)
        print("SPARQL-BASED FEATURE EXTRACTION MODE")
        print("=" * 60)
        print("Extracting features from Knowledge Graph via SPARQL queries")
        print("as specified in the project proposal.")
        print("=" * 60 + "\n")
        
        from sparql_feature_extraction import load_data_to_kg_and_extract_features
        
        featured_data = load_data_to_kg_and_extract_features(
            data_path='data/raw/matches.csv',
            limit=sample_size
        )
        
        if len(featured_data) == 0:
            print("Error: No data extracted from SPARQL. Falling back to pandas.")
            use_sparql = False
    
    if not use_sparql:
        print("Loading data...")
        preprocessor = DataPreprocessor('data/raw/matches.csv')
        clean_data = preprocessor.preprocess()
        
        if sample_size and sample_size < len(clean_data):
            print(f"\nTRAINING ON SAMPLE: {sample_size} matches (out of {len(clean_data)} total)")
            print("   This is for quick iteration and evaluation.\n")
            clean_data = clean_data.sample(n=sample_size, random_state=42).sort_values('MatchDate').reset_index(drop=True)
        
        print("Engineering features...")
        engineer = FeatureEngineer(clean_data)
        featured_data = engineer.create_all_features(include_goal_features=True)
        
        print(f"Created {len(engineer.features_created)} features")
    
    print(f"Dataset shape: {featured_data.shape}")
    
    if fast_mode:
        print("\nFAST MODE: Using smaller models for quick iteration")
    
    trainer = BettingModelTrainer(featured_data, fast_mode=fast_mode)
    trainer.train_all_models()
    
    test_results = trainer.evaluate_on_test()
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY (with UQ360 Uncertainty)")
    print("=" * 60)
    for market_name, results in trainer.evaluation_results.items():
        if results:
            print(f"\n{market_name.upper().replace('_', ' ')}:")
            print(f"  Validation Accuracy: {results.get('accuracy', 'N/A'):.4f}" if results.get('accuracy') else "  Validation Accuracy: N/A")
            print(f"  Validation Log Loss: {results.get('log_loss', 'N/A'):.4f}" if results.get('log_loss') else "  Validation Log Loss: N/A")
            if results.get('avg_uncertainty'):
                print(f"  Avg Uncertainty: {results['avg_uncertainty']:.4f}")
            if results.get('roc_auc'):
                print(f"  Validation ROC AUC: {results['roc_auc']:.4f}")
    
    trainer.save_models()
    
    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    if sample_size:
        print(f"\nRemember: Models were trained on {sample_size} sample matches.")
        print("   For production, train on full dataset without sample_size parameter.")
    if use_sparql:
        print("\nFeatures were extracted from Knowledge Graph via SPARQL")
    print("=" * 60)


if __name__ == '__main__':
    import sys
    
    sample_size = None
    fast_mode = False
    use_sparql = False
    
    if len(sys.argv) > 1:
        if '--sample' in sys.argv or '-s' in sys.argv:
            idx = sys.argv.index('--sample') if '--sample' in sys.argv else sys.argv.index('-s')
            if idx + 1 < len(sys.argv):
                try:
                    sample_size = int(sys.argv[idx + 1])
                except ValueError:
                    print("Error: Sample size must be an integer")
                    sys.exit(1)
        
        if '--fast' in sys.argv or '-f' in sys.argv:
            fast_mode = True
        
        if '--sparql' in sys.argv:
            use_sparql = True
            print("Using SPARQL-based feature extraction from Knowledge Graph")
    
    if sample_size is None and '--full' not in sys.argv:
        print("=" * 60)
        print("QUICK EVALUATION MODE")
        print("=" * 60)
        print("Training on 5000 sample matches with fast models.")
        print("Use --full to train on complete dataset.")
        print("Use --sample N to specify custom sample size.")
        print("Use --sparql to extract features via SPARQL from Knowledge Graph.")
        print("=" * 60 + "\n")
        sample_size = 5000
        fast_mode = True
    
    main(sample_size=sample_size, fast_mode=fast_mode, use_sparql=use_sparql)

