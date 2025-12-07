import pandas as pd
from pathlib import Path

class DataPreprocessor:
    ESSENTIAL_FEATURES = [
        'Division', 'MatchDate', 'HomeTeam', 'AwayTeam',
        'HomeElo', 'AwayElo', 
        'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
        'FTHome', 'FTAway', 'FTResult',
        'OddHome', 'OddDraw', 'OddAway'
    ]
    
    def __init__(self, data_path: str = 'data/raw/matches.csv'):
        self.data_path = Path(data_path)
        self.raw_data = None
        self.clean_data = None
        
    def load_data(self) -> pd.DataFrame:
        self.raw_data = pd.read_csv(
            self.data_path,
            parse_dates=['MatchDate'],
            low_memory=False
        )
        return self.raw_data
    
    def filter_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df['MatchDate'] >= '2015-01-01'].copy()
    
    def require_essential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        essential_df = df[self.ESSENTIAL_FEATURES]
        valid_rows = essential_df.notna().all(axis=1)
        cleaned_df = df[valid_rows].copy()
        return cleaned_df
    
    def preprocess(self) -> pd.DataFrame:
        if self.raw_data is None:
            self.load_data()
        
        df = self.raw_data.copy()
        
        df = self.filter_by_date(df)
        df = self.require_essential_features(df)
        
        self.clean_data = df
        
        return df
    
    def save_clean_data(self, output_path: str = 'data/processed/matches_clean.parquet'):
        if self.clean_data is None:
            raise ValueError("No clean data to save")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.clean_data.to_parquet(output_path, index=False)
    
    def get_data_summary(self) -> dict:
        if self.clean_data is None:
            raise ValueError("No clean data available")
        
        df = self.clean_data
        
        summary = {
            'total_matches': len(df),
            'date_range': (df['MatchDate'].min(), df['MatchDate'].max()),
            'leagues': df['Division'].unique().tolist(),
            'total_teams': len(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())),
            'avg_home_elo': df['HomeElo'].mean(),
            'avg_away_elo': df['AwayElo'].mean(),
            'result_distribution': df['FTResult'].value_counts().to_dict(),
            'avg_goals_per_match': (df['FTHome'] + df['FTAway']).mean(),
        }
        
        return summary

def main():
    preprocessor = DataPreprocessor('data/raw/matches.csv')
    preprocessor.preprocess()
    preprocessor.save_clean_data('data/processed/matches_clean.parquet')

if __name__ == '__main__':
    main()

