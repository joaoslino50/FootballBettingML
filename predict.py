import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
import json

from model_training import BettingModelTrainer, UncertaintyQuantifier


class BettingPredictor:

    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.uq_models = {}
        self.scalers = {}
        self.feature_names = {}
        self.metadata = None
        
        self._load_models()
    
    def _load_models(self):
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory {self.models_dir} not found")
        
        model_files = list(self.models_dir.glob("*_model_*.joblib"))
        model_files = [f for f in model_files if '_uq_model_' not in f.name]
        if not model_files:
            raise FileNotFoundError("No model files found")
        
        timestamps = set()
        for f in model_files:
            parts = f.stem.split('_model_')
            if len(parts) == 2:
                timestamps.add(parts[1])
        
        if not timestamps:
            raise ValueError("Could not parse model timestamps")
        
        latest_timestamp = sorted(timestamps)[-1]
        self.latest_timestamp = latest_timestamp
        
        metadata_file = self.models_dir / f"metadata_{latest_timestamp}.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        for market_name in self.metadata.get('models', []):
            model_file = self.models_dir / f"{market_name}_model_{latest_timestamp}.joblib"
            scaler_file = self.models_dir / f"{market_name}_scaler_{latest_timestamp}.joblib"
            features_file = self.models_dir / f"{market_name}_features_{latest_timestamp}.txt"
            uq_model_file = self.models_dir / f"{market_name}_uq_model_{latest_timestamp}.joblib"
            
            if model_file.exists() and scaler_file.exists():
                self.models[market_name] = joblib.load(model_file)
                self.scalers[market_name] = joblib.load(scaler_file)
                
                if uq_model_file.exists():
                    try:
                        self.uq_models[market_name] = joblib.load(uq_model_file)
                    except (AttributeError, ModuleNotFoundError) as e:
                        pass
                
                if features_file.exists():
                    with open(features_file, 'r') as f:
                        self.feature_names[market_name] = [line.strip() for line in f.readlines()]
        
        uq_count = len(self.uq_models)
        print(f"Loaded {len(self.models)} models, {uq_count} UQ models (timestamp: {latest_timestamp})")
    
    def predict_match(self, match_data: pd.DataFrame, 
                     home_team: str, away_team: str,
                     include_uncertainty: bool = True) -> Dict:
        predictions = {}
        
        for market_name, model in self.models.items():
            feature_cols = self.feature_names.get(market_name, [])
            X = match_data[feature_cols].copy()
            
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            X_scaled = self.scalers[market_name].transform(X)
            
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            
            uncertainty_data = None
            if include_uncertainty and market_name in self.uq_models:
                uq_result = self.uq_models[market_name].predict_with_uncertainty(X_scaled)
                uncertainty_data = {
                    'uncertainty': uq_result['uncertainty'][0],
                    'confidence_interval_lower': uq_result['confidence_interval_lower'][0],
                    'confidence_interval_upper': uq_result['confidence_interval_upper'][0]
                }
            
            if market_name == 'match_result':
                class_names = model.classes_
                pred_result = {
                    'prediction': pred,
                    'probabilities': {
                        class_names[i]: float(proba[i]) 
                        for i in range(len(class_names))
                    },
                    'home_team': home_team,
                    'away_team': away_team
                }
                
                if uncertainty_data:
                    pred_result['confidence_intervals'] = {
                        class_names[i]: [
                            float(uncertainty_data['confidence_interval_lower'][i]),
                            float(uncertainty_data['confidence_interval_upper'][i])
                        ]
                        for i in range(len(class_names))
                    }
                    pred_result['uncertainty'] = {
                        class_names[i]: float(uncertainty_data['uncertainty'][i])
                        for i in range(len(class_names))
                    }
                
                predictions[market_name] = pred_result
            else:
                prob_positive = float(proba[1]) if len(proba) > 1 else float(proba[0])
                prob_positive = max(0.0, min(1.0, prob_positive))
                
                pred_result = {
                    'prediction': bool(pred),
                    'probability': prob_positive,
                    'home_team': home_team,
                    'away_team': away_team
                }
                
                if uncertainty_data:
                    idx = 1 if len(proba) > 1 else 0
                    lower = float(uncertainty_data['confidence_interval_lower'][idx]) if hasattr(uncertainty_data['confidence_interval_lower'], '__len__') else float(uncertainty_data['confidence_interval_lower'])
                    upper = float(uncertainty_data['confidence_interval_upper'][idx]) if hasattr(uncertainty_data['confidence_interval_upper'], '__len__') else float(uncertainty_data['confidence_interval_upper'])
                    uncert = float(uncertainty_data['uncertainty'][idx]) if hasattr(uncertainty_data['uncertainty'], '__len__') else float(uncertainty_data['uncertainty'])
                    
                    pred_result['confidence_interval'] = [lower, upper]
                    pred_result['uncertainty'] = uncert
                
                predictions[market_name] = pred_result
        
        return predictions
    
    def format_predictions(self, predictions: Dict) -> str:
        output = []
        output.append("=" * 60)
        output.append("BETTING PREDICTIONS (with Uncertainty Quantification)")
        output.append("=" * 60)
        
        if 'match_result' in predictions:
            mr = predictions['match_result']
            output.append(f"\nMatch Result: {mr['home_team']} vs {mr['away_team']}")
            output.append(f"  Predicted: {mr['prediction']}")
            output.append("  Probabilities (with 90% confidence intervals):")
            for outcome, prob in mr['probabilities'].items():
                if 'confidence_intervals' in mr and outcome in mr['confidence_intervals']:
                    ci = mr['confidence_intervals'][outcome]
                    output.append(f"    {outcome}: {prob:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]")
                else:
                    output.append(f"    {outcome}: {prob:.1%}")
        
        if 'over_under_25' in predictions:
            ou = predictions['over_under_25']
            pred_text = "Over 2.5" if ou['prediction'] else "Under 2.5"
            output.append(f"\nOver/Under 2.5 Goals:")
            output.append(f"  Predicted: {pred_text}")
            if 'confidence_interval' in ou:
                output.append(f"  Probability: {ou['probability']:.1%} [{ou['confidence_interval'][0]:.1%}, {ou['confidence_interval'][1]:.1%}]")
            else:
                output.append(f"  Probability: {ou['probability']:.1%}")
        
        if 'btts' in predictions:
            btts = predictions['btts']
            pred_text = "Yes" if btts['prediction'] else "No"
            output.append(f"\nBoth Teams To Score:")
            output.append(f"  Predicted: {pred_text}")
            if 'confidence_interval' in btts:
                output.append(f"  Probability: {btts['probability']:.1%} [{btts['confidence_interval'][0]:.1%}, {btts['confidence_interval'][1]:.1%}]")
            else:
                output.append(f"  Probability: {btts['probability']:.1%}")
        
        if 'clean_sheet_home' in predictions:
            cs = predictions['clean_sheet_home']
            pred_text = "Yes" if cs['prediction'] else "No"
            output.append(f"\nHome Team Clean Sheet:")
            output.append(f"  Predicted: {pred_text}")
            if 'confidence_interval' in cs:
                output.append(f"  Probability: {cs['probability']:.1%} [{cs['confidence_interval'][0]:.1%}, {cs['confidence_interval'][1]:.1%}]")
            else:
                output.append(f"  Probability: {cs['probability']:.1%}")
        
        if 'clean_sheet_away' in predictions:
            cs = predictions['clean_sheet_away']
            pred_text = "Yes" if cs['prediction'] else "No"
            output.append(f"\nAway Team Clean Sheet:")
            output.append(f"  Predicted: {pred_text}")
            if 'confidence_interval' in cs:
                output.append(f"  Probability: {cs['probability']:.1%} [{cs['confidence_interval'][0]:.1%}, {cs['confidence_interval'][1]:.1%}]")
            else:
                output.append(f"  Probability: {cs['probability']:.1%}")
        
        output.append("=" * 60)
        return "\n".join(output)


def main():
    from feature_engineering import FeatureEngineer
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor('data/raw/matches.csv')
    clean_data = preprocessor.preprocess()
    
    engineer = FeatureEngineer(clean_data)
    featured_data = engineer.create_all_features(include_goal_features=True)
    
    try:
        predictor = BettingPredictor()
        
        recent_match = featured_data.iloc[-1:].copy()
        home_team = recent_match['HomeTeam'].iloc[0]
        away_team = recent_match['AwayTeam'].iloc[0]
        
        predictions = predictor.predict_match(recent_match, home_team, away_team)
        
        print(predictor.format_predictions(predictions))
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train models first using: python model_training.py")


if __name__ == '__main__':
    main()

