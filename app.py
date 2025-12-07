from pathlib import Path
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer

def main():    
    data_path = 'data/raw/matches.csv'
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    preprocessor = DataPreprocessor(data_path)
    clean_data = preprocessor.preprocess()
    
    print(f"Loaded raw data: {len(preprocessor.raw_data)} matches")
    print(f"Cleaned data: {len(clean_data)} matches")
    
    summary = preprocessor.get_data_summary()
    print(f"\nData Summary:")
    print(f"  - Total matches: {summary['total_matches']}")
    print(f"  - Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
    print(f"  - Leagues: {len(summary['leagues'])}")
    print(f"  - Total teams: {summary['total_teams']}")
    print(f"  - Avg goals per match: {summary['avg_goals_per_match']:.2f}")
    
    engineer = FeatureEngineer(clean_data)
    featured_data = engineer.create_all_features(include_goal_features=True)
    
    print(f"\nCreated {len(engineer.features_created)} new features")
    
    feature_categories = {
        'ELO Features': engineer.get_feature_list('elo'),
        'Form Features': engineer.get_feature_list('form'),
        'Odds Features': engineer.get_feature_list('odds'),
        'Temporal Features': engineer.get_feature_list('temporal'),
        'Head-to-Head Features': engineer.get_feature_list('h2h'),
        'Goal Efficiency Features': engineer.get_feature_list('goal_efficiency'),
        'Goal Features': engineer.get_feature_list('goal')
    }
    
    print(f"\nFeature Breakdown:")
    for category, features in feature_categories.items():
        print(f"{category}: {len(features)} features")
        if features:
            print(f"    All features: {', '.join(features)}")
    
    print(f"Final dataset shape: {featured_data.shape}")
    print(f"Rows: {featured_data.shape[0]}")
    print(f"Columns: {featured_data.shape[1]}")
    
if __name__ == '__main__':
    main()

