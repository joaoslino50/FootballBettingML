import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.features_created = []
        
    def create_elo_features(self) -> pd.DataFrame:
        self.df['EloDiff'] = self.df['HomeElo'] - self.df['AwayElo']
        self.df['EloRatio'] = self.df['HomeElo'] / self.df['AwayElo']
        self.df['HomeEloAdvantage'] = (self.df['EloDiff'] > 0).astype(int)
        self.df['HomeElo_Std'] = (self.df['HomeElo'] - self.df['HomeElo'].mean()) / self.df['HomeElo'].std()
        self.df['AwayElo_Std'] = (self.df['AwayElo'] - self.df['AwayElo'].mean()) / self.df['AwayElo'].std()
        
        created = ['EloDiff', 'EloRatio', 'HomeEloAdvantage', 'HomeElo_Std', 'AwayElo_Std']
        self.features_created.extend(created)
        
        return self.df
    
    def create_form_features(self) -> pd.DataFrame:
        self.df['Form3Diff'] = self.df['Form3Home'] - self.df['Form3Away']
        self.df['Form5Diff'] = self.df['Form5Home'] - self.df['Form5Away']
        self.df['TotalForm3'] = self.df['Form3Home'] + self.df['Form3Away']
        self.df['TotalForm5'] = self.df['Form5Home'] + self.df['Form5Away']
        
        epsilon = 1e-8
        self.df['Form3Ratio'] = self.df['Form3Home'] / (self.df['Form3Away'] + epsilon)
        
        self.df['HomeFormAdvantage'] = (self.df['Form5Diff'] > 0).astype(int)
        
        created = ['Form3Diff', 'Form5Diff', 'TotalForm3', 'TotalForm5', 
                   'Form3Ratio', 'HomeFormAdvantage']
        self.features_created.extend(created)
        
        return self.df
    
    def create_odds_features(self) -> pd.DataFrame:
        self.df['ImpliedProbHome'] = 1 / self.df['OddHome']
        self.df['ImpliedProbDraw'] = 1 / self.df['OddDraw']
        self.df['ImpliedProbAway'] = 1 / self.df['OddAway']
        
        self.df['Overround'] = (
            self.df['ImpliedProbHome'] + 
            self.df['ImpliedProbDraw'] + 
            self.df['ImpliedProbAway']
        )
        
        self.df['BookmakerMargin'] = self.df['Overround'] - 1.0
        
        self.df['TrueProbHome'] = self.df['ImpliedProbHome'] / self.df['Overround']
        self.df['TrueProbDraw'] = self.df['ImpliedProbDraw'] / self.df['Overround']
        self.df['TrueProbAway'] = self.df['ImpliedProbAway'] / self.df['Overround']
        
        self.df['HomeFavorite'] = (
            (self.df['OddHome'] < self.df['OddDraw']) & 
            (self.df['OddHome'] < self.df['OddAway'])
        ).astype(int)
        
        self.df['AwayFavorite'] = (
            (self.df['OddAway'] < self.df['OddDraw']) & 
            (self.df['OddAway'] < self.df['OddHome'])
        ).astype(int)
        
        self.df['OddsRatio'] = self.df['OddAway'] / self.df['OddHome']
        
        created = ['ImpliedProbHome', 'ImpliedProbDraw', 'ImpliedProbAway',
                   'Overround', 'BookmakerMargin', 'TrueProbHome', 'TrueProbDraw', 
                   'TrueProbAway', 'HomeFavorite', 'AwayFavorite', 'OddsRatio']
        self.features_created.extend(created)
        
        return self.df
    
    def create_temporal_features(self) -> pd.DataFrame:
        self.df['MatchDate'] = pd.to_datetime(self.df['MatchDate'])
        
        self.df['DayOfWeek'] = self.df['MatchDate'].dt.dayofweek
        self.df['IsWeekend'] = self.df['DayOfWeek'].isin([5, 6]).astype(int)
        self.df['Month'] = self.df['MatchDate'].dt.month
        
        conditions = [
            self.df['Month'].isin([8, 9, 10]),
            self.df['Month'].isin([11, 12, 1, 2]),
            self.df['Month'].isin([3, 4, 5])
        ]
        self.df['SeasonPhase'] = np.select(conditions, ['Early', 'Mid', 'Late'], default='Other')
        
        season_dummies = pd.get_dummies(self.df['SeasonPhase'], prefix='Season')
        self.df = pd.concat([self.df, season_dummies], axis=1)
        
        created = ['DayOfWeek', 'IsWeekend', 'Month', 'SeasonPhase'] + list(season_dummies.columns)
        self.features_created.extend(created)
        
        return self.df
    
    def create_goal_features(self) -> pd.DataFrame:
        self.df['TotalGoals'] = self.df['FTHome'] + self.df['FTAway']
        self.df['Over25'] = (self.df['TotalGoals'] > 2.5).astype(int)
        self.df['GoalDifference'] = self.df['FTHome'] - self.df['FTAway']
        self.df['HighScoring'] = (self.df['TotalGoals'] >= 4).astype(int)
        
        created = ['TotalGoals', 'Over25', 'GoalDifference', 'HighScoring']
        self.features_created.extend(created)
        
        return self.df
    
    def create_head_to_head_features(self) -> pd.DataFrame:
        self.df['MatchDate'] = pd.to_datetime(self.df['MatchDate'])
        df_sorted = self.df.sort_values('MatchDate').reset_index(drop=True)
        
        h2h_features = {
            'H2H_TotalMatches': np.zeros(len(df_sorted), dtype=int),
            'H2H_HomeWins': np.zeros(len(df_sorted), dtype=int),
            'H2H_Draws': np.zeros(len(df_sorted), dtype=int),
            'H2H_AwayWins': np.zeros(len(df_sorted), dtype=int),
            'H2H_HomeWinRate': np.zeros(len(df_sorted), dtype=float),
            'H2H_AvgGoalsHome': np.zeros(len(df_sorted), dtype=float),
            'H2H_AvgGoalsAway': np.zeros(len(df_sorted), dtype=float),
            'H2H_AvgTotalGoals': np.zeros(len(df_sorted), dtype=float),
            'H2H_Last3HomeWins': np.zeros(len(df_sorted), dtype=int),
            'H2H_Last3Draws': np.zeros(len(df_sorted), dtype=int),
            'H2H_Last3AwayWins': np.zeros(len(df_sorted), dtype=int),
            'H2H_Last5HomeWins': np.zeros(len(df_sorted), dtype=int),
            'H2H_Last5Draws': np.zeros(len(df_sorted), dtype=int),
            'H2H_Last5AwayWins': np.zeros(len(df_sorted), dtype=int),
        }
        
        for idx, row in df_sorted.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            match_date = row['MatchDate']
            
            previous_matches = df_sorted[
                (df_sorted['MatchDate'] < match_date) &
                (
                    ((df_sorted['HomeTeam'] == home_team) & (df_sorted['AwayTeam'] == away_team)) |
                    ((df_sorted['HomeTeam'] == away_team) & (df_sorted['AwayTeam'] == home_team))
                )
            ].copy()
            
            if len(previous_matches) > 0:
                home_as_home = previous_matches[
                    (previous_matches['HomeTeam'] == home_team) & 
                    (previous_matches['AwayTeam'] == away_team)
                ]
                
                home_as_away = previous_matches[
                    (previous_matches['HomeTeam'] == away_team) & 
                    (previous_matches['AwayTeam'] == home_team)
                ]
                
                if len(home_as_home) > 0:
                    h2h_features['H2H_HomeWins'][idx] += (home_as_home['FTResult'] == 'H').sum()
                    h2h_features['H2H_Draws'][idx] += (home_as_home['FTResult'] == 'D').sum()
                    h2h_features['H2H_AwayWins'][idx] += (home_as_home['FTResult'] == 'A').sum()
                    
                    h2h_features['H2H_AvgGoalsHome'][idx] += home_as_home['FTHome'].sum()
                    h2h_features['H2H_AvgGoalsAway'][idx] += home_as_home['FTAway'].sum()
                
                if len(home_as_away) > 0:
                    h2h_features['H2H_HomeWins'][idx] += (home_as_away['FTResult'] == 'A').sum()
                    h2h_features['H2H_Draws'][idx] += (home_as_away['FTResult'] == 'D').sum()
                    h2h_features['H2H_AwayWins'][idx] += (home_as_away['FTResult'] == 'H').sum()
                    
                    h2h_features['H2H_AvgGoalsHome'][idx] += home_as_away['FTAway'].sum()
                    h2h_features['H2H_AvgGoalsAway'][idx] += home_as_away['FTHome'].sum()
                
                total_matches = len(previous_matches)
                h2h_features['H2H_TotalMatches'][idx] = total_matches
                
                if total_matches > 0:
                    h2h_features['H2H_HomeWinRate'][idx] = h2h_features['H2H_HomeWins'][idx] / total_matches
                    h2h_features['H2H_AvgGoalsHome'][idx] /= total_matches
                    h2h_features['H2H_AvgGoalsAway'][idx] /= total_matches
                    h2h_features['H2H_AvgTotalGoals'][idx] = (
                        h2h_features['H2H_AvgGoalsHome'][idx] + 
                        h2h_features['H2H_AvgGoalsAway'][idx]
                    )
                
                recent_matches = previous_matches.nlargest(5, 'MatchDate')
                
                for match_pos, (recent_idx, recent_row) in enumerate(recent_matches.iterrows(), 1):
                    if recent_row['HomeTeam'] == home_team:
                        result = recent_row['FTResult']
                    else:
                        result = 'H' if recent_row['FTResult'] == 'A' else ('A' if recent_row['FTResult'] == 'H' else 'D')
                    
                    if result == 'H':
                        h2h_features['H2H_Last5HomeWins'][idx] += 1
                        if match_pos <= 3:
                            h2h_features['H2H_Last3HomeWins'][idx] += 1
                    elif result == 'D':
                        h2h_features['H2H_Last5Draws'][idx] += 1
                        if match_pos <= 3:
                            h2h_features['H2H_Last3Draws'][idx] += 1
                    else:
                        h2h_features['H2H_Last5AwayWins'][idx] += 1
                        if match_pos <= 3:
                            h2h_features['H2H_Last3AwayWins'][idx] += 1
        
        for feature_name, feature_values in h2h_features.items():
            df_sorted[feature_name] = feature_values
        
        df_sorted['H2H_HomeAdvantage'] = (
            (df_sorted['H2H_HomeWinRate'] > 0.5).astype(int)
        )
        df_sorted['H2H_GoalDiff'] = (
            df_sorted['H2H_AvgGoalsHome'] - df_sorted['H2H_AvgGoalsAway']
        )
        df_sorted['H2H_Last3HomeWinRate'] = np.where(
            df_sorted['H2H_TotalMatches'] >= 3,
            df_sorted['H2H_Last3HomeWins'] / 3.0,
            0.0
        )
        df_sorted['H2H_Last5HomeWinRate'] = np.where(
            df_sorted['H2H_TotalMatches'] >= 5,
            df_sorted['H2H_Last5HomeWins'] / 5.0,
            0.0
        )
        
        self.df = df_sorted
        
        created = [
            'H2H_TotalMatches', 'H2H_HomeWins', 'H2H_Draws', 'H2H_AwayWins',
            'H2H_HomeWinRate', 'H2H_AvgGoalsHome', 'H2H_AvgGoalsAway', 
            'H2H_AvgTotalGoals', 'H2H_Last3HomeWins', 'H2H_Last3Draws',
            'H2H_Last3AwayWins', 'H2H_Last5HomeWins', 'H2H_Last5Draws',
            'H2H_Last5AwayWins', 'H2H_HomeAdvantage', 'H2H_GoalDiff',
            'H2H_Last3HomeWinRate', 'H2H_Last5HomeWinRate'
        ]
        self.features_created.extend(created)
        
        return self.df
    
    def create_goal_efficiency_features(self) -> pd.DataFrame:
        self.df['MatchDate'] = pd.to_datetime(self.df['MatchDate'])
        df_sorted = self.df.sort_values('MatchDate').reset_index(drop=True)
        
        efficiency_features = {
            'HomeTeam_GoalsScoredPerGame': np.zeros(len(df_sorted), dtype=float),
            'HomeTeam_GoalsConcededPerGame': np.zeros(len(df_sorted), dtype=float),
            'HomeTeam_GoalsScoredHome': np.zeros(len(df_sorted), dtype=float),
            'HomeTeam_GoalsConcededHome': np.zeros(len(df_sorted), dtype=float),
            'HomeTeam_CleanSheetRate': np.zeros(len(df_sorted), dtype=float),
            'HomeTeam_BTTSRate': np.zeros(len(df_sorted), dtype=float),
            'HomeTeam_AvgTotalGoals': np.zeros(len(df_sorted), dtype=float),
            'HomeTeam_MatchesPlayed': np.zeros(len(df_sorted), dtype=int),
            
            'AwayTeam_GoalsScoredPerGame': np.zeros(len(df_sorted), dtype=float),
            'AwayTeam_GoalsConcededPerGame': np.zeros(len(df_sorted), dtype=float),
            'AwayTeam_GoalsScoredAway': np.zeros(len(df_sorted), dtype=float),
            'AwayTeam_GoalsConcededAway': np.zeros(len(df_sorted), dtype=float),
            'AwayTeam_CleanSheetRate': np.zeros(len(df_sorted), dtype=float),
            'AwayTeam_BTTSRate': np.zeros(len(df_sorted), dtype=float),
            'AwayTeam_AvgTotalGoals': np.zeros(len(df_sorted), dtype=float),
            'AwayTeam_MatchesPlayed': np.zeros(len(df_sorted), dtype=int),
            
            'HomeTeam_RecentGoalsScored': np.zeros(len(df_sorted), dtype=float),
            'HomeTeam_RecentGoalsConceded': np.zeros(len(df_sorted), dtype=float),
            'AwayTeam_RecentGoalsScored': np.zeros(len(df_sorted), dtype=float),
            'AwayTeam_RecentGoalsConceded': np.zeros(len(df_sorted), dtype=float),
        }
        
        for idx, row in df_sorted.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            match_date = row['MatchDate']
            
            home_previous = df_sorted[
                (df_sorted['MatchDate'] < match_date) &
                ((df_sorted['HomeTeam'] == home_team) | (df_sorted['AwayTeam'] == home_team))
            ]
            
            if len(home_previous) > 0:
                home_as_home = home_previous[home_previous['HomeTeam'] == home_team]
                home_as_away = home_previous[home_previous['AwayTeam'] == home_team]
                
                total_goals_scored = home_as_home['FTHome'].sum() + home_as_away['FTAway'].sum()
                total_goals_conceded = home_as_home['FTAway'].sum() + home_as_away['FTHome'].sum()
                total_matches = len(home_previous)
                
                efficiency_features['HomeTeam_MatchesPlayed'][idx] = total_matches
                efficiency_features['HomeTeam_GoalsScoredPerGame'][idx] = total_goals_scored / total_matches
                efficiency_features['HomeTeam_GoalsConcededPerGame'][idx] = total_goals_conceded / total_matches
                efficiency_features['HomeTeam_AvgTotalGoals'][idx] = (total_goals_scored + total_goals_conceded) / total_matches
                
                if len(home_as_home) > 0:
                    efficiency_features['HomeTeam_GoalsScoredHome'][idx] = home_as_home['FTHome'].sum() / len(home_as_home)
                    efficiency_features['HomeTeam_GoalsConcededHome'][idx] = home_as_home['FTAway'].sum() / len(home_as_home)
                
                clean_sheets = (home_as_home['FTAway'] == 0).sum() + (home_as_away['FTHome'] == 0).sum()
                efficiency_features['HomeTeam_CleanSheetRate'][idx] = clean_sheets / total_matches
                
                btts_count = (
                    ((home_as_home['FTHome'] > 0) & (home_as_home['FTAway'] > 0)).sum() +
                    ((home_as_away['FTHome'] > 0) & (home_as_away['FTAway'] > 0)).sum()
                )
                efficiency_features['HomeTeam_BTTSRate'][idx] = btts_count / total_matches
                
                recent_home = home_previous.tail(5)
                if len(recent_home) > 0:
                    recent_goals_scored = 0
                    recent_goals_conceded = 0
                    for _, recent_row in recent_home.iterrows():
                        if recent_row['HomeTeam'] == home_team:
                            recent_goals_scored += recent_row['FTHome']
                            recent_goals_conceded += recent_row['FTAway']
                        else:
                            recent_goals_scored += recent_row['FTAway']
                            recent_goals_conceded += recent_row['FTHome']
                    
                    efficiency_features['HomeTeam_RecentGoalsScored'][idx] = recent_goals_scored / len(recent_home)
                    efficiency_features['HomeTeam_RecentGoalsConceded'][idx] = recent_goals_conceded / len(recent_home)
            
            away_previous = df_sorted[
                (df_sorted['MatchDate'] < match_date) &
                ((df_sorted['HomeTeam'] == away_team) | (df_sorted['AwayTeam'] == away_team))
            ]
            
            if len(away_previous) > 0:
                away_as_home = away_previous[away_previous['HomeTeam'] == away_team]
                away_as_away = away_previous[away_previous['AwayTeam'] == away_team]
                
                total_goals_scored = away_as_home['FTHome'].sum() + away_as_away['FTAway'].sum()
                total_goals_conceded = away_as_home['FTAway'].sum() + away_as_away['FTHome'].sum()
                total_matches = len(away_previous)
                
                efficiency_features['AwayTeam_MatchesPlayed'][idx] = total_matches
                efficiency_features['AwayTeam_GoalsScoredPerGame'][idx] = total_goals_scored / total_matches
                efficiency_features['AwayTeam_GoalsConcededPerGame'][idx] = total_goals_conceded / total_matches
                efficiency_features['AwayTeam_AvgTotalGoals'][idx] = (total_goals_scored + total_goals_conceded) / total_matches
                
                if len(away_as_away) > 0:
                    efficiency_features['AwayTeam_GoalsScoredAway'][idx] = away_as_away['FTAway'].sum() / len(away_as_away)
                    efficiency_features['AwayTeam_GoalsConcededAway'][idx] = away_as_away['FTHome'].sum() / len(away_as_away)
                
                clean_sheets = (away_as_home['FTAway'] == 0).sum() + (away_as_away['FTHome'] == 0).sum()
                efficiency_features['AwayTeam_CleanSheetRate'][idx] = clean_sheets / total_matches
                
                btts_count = (
                    ((away_as_home['FTHome'] > 0) & (away_as_home['FTAway'] > 0)).sum() +
                    ((away_as_away['FTHome'] > 0) & (away_as_away['FTAway'] > 0)).sum()
                )
                efficiency_features['AwayTeam_BTTSRate'][idx] = btts_count / total_matches
                
                recent_away = away_previous.tail(5)
                if len(recent_away) > 0:
                    recent_goals_scored = 0
                    recent_goals_conceded = 0
                    for _, recent_row in recent_away.iterrows():
                        if recent_row['HomeTeam'] == away_team:
                            recent_goals_scored += recent_row['FTHome']
                            recent_goals_conceded += recent_row['FTAway']
                        else:
                            recent_goals_scored += recent_row['FTAway']
                            recent_goals_conceded += recent_row['FTHome']
                    
                    efficiency_features['AwayTeam_RecentGoalsScored'][idx] = recent_goals_scored / len(recent_away)
                    efficiency_features['AwayTeam_RecentGoalsConceded'][idx] = recent_goals_conceded / len(recent_away)
        
        for feature_name, feature_values in efficiency_features.items():
            df_sorted[feature_name] = feature_values
        
        df_sorted['GoalEfficiency_ScoringDiff'] = (
            df_sorted['HomeTeam_GoalsScoredPerGame'] - df_sorted['AwayTeam_GoalsScoredPerGame']
        )
        df_sorted['GoalEfficiency_ConcedingDiff'] = (
            df_sorted['HomeTeam_GoalsConcededPerGame'] - df_sorted['AwayTeam_GoalsConcededPerGame']
        )
        df_sorted['GoalEfficiency_NetGoalDiff'] = (
            df_sorted['GoalEfficiency_ScoringDiff'] - df_sorted['GoalEfficiency_ConcedingDiff']
        )
        df_sorted['GoalEfficiency_ExpectedTotalGoals'] = (
            df_sorted['HomeTeam_GoalsScoredHome'] + df_sorted['HomeTeam_GoalsConcededHome'] +
            df_sorted['AwayTeam_GoalsScoredAway'] + df_sorted['AwayTeam_GoalsConcededAway']
        )
        df_sorted['GoalEfficiency_BTTSProbability'] = (
            df_sorted['HomeTeam_BTTSRate'] * df_sorted['AwayTeam_BTTSRate']
        )
        df_sorted['GoalEfficiency_CleanSheetProbability'] = (
            df_sorted['HomeTeam_CleanSheetRate'] * df_sorted['AwayTeam_CleanSheetRate']
        )
        df_sorted['GoalEfficiency_Over25Probability'] = (
            (df_sorted['HomeTeam_AvgTotalGoals'] + df_sorted['AwayTeam_AvgTotalGoals']) / 2 > 2.5
        ).astype(float)
        
        self.df = df_sorted
        
        created = [
            'HomeTeam_GoalsScoredPerGame', 'HomeTeam_GoalsConcededPerGame',
            'HomeTeam_GoalsScoredHome', 'HomeTeam_GoalsConcededHome',
            'HomeTeam_CleanSheetRate', 'HomeTeam_BTTSRate', 'HomeTeam_AvgTotalGoals',
            'HomeTeam_MatchesPlayed', 'HomeTeam_RecentGoalsScored', 'HomeTeam_RecentGoalsConceded',
            'AwayTeam_GoalsScoredPerGame', 'AwayTeam_GoalsConcededPerGame',
            'AwayTeam_GoalsScoredAway', 'AwayTeam_GoalsConcededAway',
            'AwayTeam_CleanSheetRate', 'AwayTeam_BTTSRate', 'AwayTeam_AvgTotalGoals',
            'AwayTeam_MatchesPlayed', 'AwayTeam_RecentGoalsScored', 'AwayTeam_RecentGoalsConceded',
            'GoalEfficiency_ScoringDiff', 'GoalEfficiency_ConcedingDiff', 'GoalEfficiency_NetGoalDiff',
            'GoalEfficiency_ExpectedTotalGoals', 'GoalEfficiency_BTTSProbability',
            'GoalEfficiency_CleanSheetProbability', 'GoalEfficiency_Over25Probability'
        ]
        self.features_created.extend(created)
        
        return self.df
    
    def create_all_features(self, 
                           include_goal_features: bool = True,
                           include_h2h_features: bool = True,
                           include_goal_efficiency: bool = True) -> pd.DataFrame:
        self.create_elo_features()
        self.create_form_features()
        self.create_odds_features()
        self.create_temporal_features()
        
        if include_h2h_features:
            self.create_head_to_head_features()
        
        if include_goal_efficiency:
            self.create_goal_efficiency_features()
        
        if include_goal_features:
            self.create_goal_features()
        
        return self.df
    
    def get_feature_list(self, category: Optional[str] = None) -> list:
        if category is None:
            return self.features_created
        
        category_patterns = {
            'elo': ['Elo'],
            'form': ['Form'],
            'odds': ['Odd', 'Implied', 'Prob', 'Favorite', 'Overround', 'Margin'],
            'temporal': ['Day', 'Week', 'Month', 'Season'],
            'goal': ['Goal', 'Over', 'High', 'Scoring'],
            'h2h': ['H2H'],
            'goal_efficiency': ['GoalEfficiency', 'GoalsScored', 'GoalsConceded', 'CleanSheet', 'BTTS']
        }
        
        if category in category_patterns:
            patterns = category_patterns[category]
            return [f for f in self.features_created 
                   if any(p in f for p in patterns)]
        
        return []
    
    def save_features(self, output_path: str = 'data/features/matches_with_features.parquet'):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_parquet(output_path, index=False)

def main():
    df = pd.read_parquet('data/processed/matches_clean.parquet')
    engineer = FeatureEngineer(df)
    engineer.create_all_features(include_goal_features=True)
    engineer.save_features('data/features/matches_with_features.parquet')

if __name__ == '__main__':
    main()

