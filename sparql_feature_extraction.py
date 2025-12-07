"""
SPARQL-Based Feature Extraction for Football Betting ML

This module extracts ML features from the RDF Knowledge Graph using SPARQL queries,
as specified in the project proposal. Features are queried from the KG and returned
as pandas DataFrames for model training.

Flow: CSV -> Knowledge Graph -> SPARQL Queries -> Feature DataFrame -> ML Models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from knowledge_graph import FootballKnowledgeGraph, STRIKO, SCHEMA


class SPARQLFeatureExtractor:
    """Extract ML features from Knowledge Graph using SPARQL queries."""
    
    def __init__(self, kg: FootballKnowledgeGraph):
        self.kg = kg
        self.graph = kg.graph
        
    def extract_elo_features(self, match_uris: List[str] = None) -> pd.DataFrame:
        """
        Extract Elo-based features via SPARQL.
        
        Features extracted:
        - HomeElo, AwayElo
        - EloDiff (computed)
        - EloRatio (computed)
        """
        query = """
        SELECT ?match ?homeTeam ?awayTeam ?date ?homeElo ?awayElo
        WHERE {
            ?match a schema:SportsEvent .
            ?match schema:homeTeam ?homeTeam .
            ?match schema:awayTeam ?awayTeam .
            ?match schema:startDate ?date .
            OPTIONAL { ?match striko:homeElo ?homeElo . }
            OPTIONAL { ?match striko:awayElo ?awayElo . }
        }
        ORDER BY ?date
        """
        
        results = []
        for row in self.graph.query(query, initNs={"schema": SCHEMA, "striko": STRIKO}):
            match_uri = str(row.match)
            if match_uris and match_uri not in match_uris:
                continue
                
            home_elo = float(row.homeElo) if row.homeElo else None
            away_elo = float(row.awayElo) if row.awayElo else None
            
            results.append({
                'match_uri': match_uri,
                'HomeTeam': str(row.homeTeam).split('/')[-1].replace('_', ' ').title(),
                'AwayTeam': str(row.awayTeam).split('/')[-1].replace('_', ' ').title(),
                'MatchDate': str(row.date),
                'HomeElo': home_elo,
                'AwayElo': away_elo,
            })
        
        df = pd.DataFrame(results)
        
        if len(df) > 0 and 'HomeElo' in df.columns and 'AwayElo' in df.columns:
            df['EloDiff'] = df['HomeElo'] - df['AwayElo']
            df['EloRatio'] = df['HomeElo'] / df['AwayElo'].replace(0, 1)
            df['HomeEloAdvantage'] = (df['EloDiff'] > 0).astype(int)
            
            df['HomeElo_Std'] = (df['HomeElo'] - df['HomeElo'].mean()) / df['HomeElo'].std()
            df['AwayElo_Std'] = (df['AwayElo'] - df['AwayElo'].mean()) / df['AwayElo'].std()
        
        return df
    
    def extract_form_features(self, match_uris: List[str] = None) -> pd.DataFrame:
        """
        Extract form-based features via SPARQL.
        
        Features extracted:
        - Form3Home, Form5Home (from stored data)
        - Form3Away, Form5Away (from stored data)
        - FormDiff features (computed)
        """
        query = """
        SELECT ?match ?homeTeam ?awayTeam ?date ?form3Home ?form5Home ?form3Away ?form5Away
        WHERE {
            ?match a schema:SportsEvent .
            ?match schema:homeTeam ?homeTeam .
            ?match schema:awayTeam ?awayTeam .
            ?match schema:startDate ?date .
            OPTIONAL { ?match striko:form3Home ?form3Home . }
            OPTIONAL { ?match striko:form5Home ?form5Home . }
            OPTIONAL { ?match striko:form3Away ?form3Away . }
            OPTIONAL { ?match striko:form5Away ?form5Away . }
        }
        ORDER BY ?date
        """
        
        results = []
        for row in self.graph.query(query, initNs={"schema": SCHEMA, "striko": STRIKO}):
            match_uri = str(row.match)
            if match_uris and match_uri not in match_uris:
                continue
                
            results.append({
                'match_uri': match_uri,
                'Form3Home': float(row.form3Home) if row.form3Home else 0.0,
                'Form5Home': float(row.form5Home) if row.form5Home else 0.0,
                'Form3Away': float(row.form3Away) if row.form3Away else 0.0,
                'Form5Away': float(row.form5Away) if row.form5Away else 0.0,
            })
        
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            df['Form3Diff'] = df['Form3Home'] - df['Form3Away']
            df['Form5Diff'] = df['Form5Home'] - df['Form5Away']
            df['TotalForm3'] = df['Form3Home'] + df['Form3Away']
            df['TotalForm5'] = df['Form5Home'] + df['Form5Away']
            epsilon = 1e-8
            df['Form3Ratio'] = df['Form3Home'] / (df['Form3Away'] + epsilon)
            df['HomeFormAdvantage'] = (df['Form5Diff'] > 0).astype(int)
        
        return df
    
    def extract_odds_features(self, match_uris: List[str] = None) -> pd.DataFrame:
        """
        Extract odds-based features via SPARQL.
        
        Features extracted:
        - OddHome, OddDraw, OddAway
        - ImpliedProb features
        - Overround, BookmakerMargin
        """
        query = """
        SELECT ?match ?odds ?homeOdds ?drawOdds ?awayOdds 
               ?impliedProbHome ?impliedProbDraw ?impliedProbAway
        WHERE {
            ?odds a striko:BettingOdds .
            ?odds striko:forMatch ?match .
            OPTIONAL { ?odds striko:homeOdds ?homeOdds . }
            OPTIONAL { ?odds striko:drawOdds ?drawOdds . }
            OPTIONAL { ?odds striko:awayOdds ?awayOdds . }
            OPTIONAL { ?odds striko:impliedProbHome ?impliedProbHome . }
            OPTIONAL { ?odds striko:impliedProbDraw ?impliedProbDraw . }
            OPTIONAL { ?odds striko:impliedProbAway ?impliedProbAway . }
        }
        """
        
        results = []
        for row in self.graph.query(query, initNs={"striko": STRIKO}):
            match_uri = str(row.match)
            if match_uris and match_uri not in match_uris:
                continue
                
            home_odds = float(row.homeOdds) if row.homeOdds else None
            draw_odds = float(row.drawOdds) if row.drawOdds else None
            away_odds = float(row.awayOdds) if row.awayOdds else None
            
            results.append({
                'match_uri': match_uri,
                'OddHome': home_odds,
                'OddDraw': draw_odds,
                'OddAway': away_odds,
                'ImpliedProbHome': float(row.impliedProbHome) if row.impliedProbHome else None,
                'ImpliedProbDraw': float(row.impliedProbDraw) if row.impliedProbDraw else None,
                'ImpliedProbAway': float(row.impliedProbAway) if row.impliedProbAway else None,
            })
        
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            if df['ImpliedProbHome'].isna().all():
                df['ImpliedProbHome'] = 1 / df['OddHome']
                df['ImpliedProbDraw'] = 1 / df['OddDraw']
                df['ImpliedProbAway'] = 1 / df['OddAway']
            
            df['Overround'] = df['ImpliedProbHome'] + df['ImpliedProbDraw'] + df['ImpliedProbAway']
            df['BookmakerMargin'] = df['Overround'] - 1.0
            
            df['TrueProbHome'] = df['ImpliedProbHome'] / df['Overround']
            df['TrueProbDraw'] = df['ImpliedProbDraw'] / df['Overround']
            df['TrueProbAway'] = df['ImpliedProbAway'] / df['Overround']
            
            df['HomeFavorite'] = ((df['OddHome'] < df['OddDraw']) & 
                                   (df['OddHome'] < df['OddAway'])).astype(int)
            df['AwayFavorite'] = ((df['OddAway'] < df['OddDraw']) & 
                                   (df['OddAway'] < df['OddHome'])).astype(int)
            df['OddsRatio'] = df['OddAway'] / df['OddHome']
        
        return df
    
    def extract_match_results(self, match_uris: List[str] = None) -> pd.DataFrame:
        """
        Extract match results and goal data via SPARQL.
        
        Features extracted:
        - FTHome, FTAway, FTResult
        - TotalGoals, Over25, BTTS
        """
        query = """
        SELECT ?match ?homeGoals ?awayGoals ?result ?totalGoals ?over25 ?btts
        WHERE {
            ?match a schema:SportsEvent .
            OPTIONAL { ?match striko:homeGoals ?homeGoals . }
            OPTIONAL { ?match striko:awayGoals ?awayGoals . }
            OPTIONAL { ?match striko:result ?result . }
            OPTIONAL { ?match striko:totalGoals ?totalGoals . }
            OPTIONAL { ?match striko:over25 ?over25 . }
            OPTIONAL { ?match striko:btts ?btts . }
        }
        """
        
        results = []
        for row in self.graph.query(query, initNs={"schema": SCHEMA, "striko": STRIKO}):
            match_uri = str(row.match)
            if match_uris and match_uri not in match_uris:
                continue
                
            results.append({
                'match_uri': match_uri,
                'FTHome': int(row.homeGoals) if row.homeGoals else None,
                'FTAway': int(row.awayGoals) if row.awayGoals else None,
                'FTResult': str(row.result) if row.result else None,
                'TotalGoals': int(row.totalGoals) if row.totalGoals else None,
                'Over25': bool(row.over25) if row.over25 else None,
                'BTTS': bool(row.btts) if row.btts else None,
            })
        
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            if df['TotalGoals'].isna().any():
                df['TotalGoals'] = df['FTHome'] + df['FTAway']
            if df['Over25'].isna().any():
                df['Over25'] = (df['TotalGoals'] > 2.5).astype(int)
            
            df['GoalDifference'] = df['FTHome'] - df['FTAway']
            df['HighScoring'] = (df['TotalGoals'] >= 4).astype(int)
        
        return df
    
    def extract_h2h_features_sparql(self, home_team: str, away_team: str, 
                                     before_date: str) -> Dict:
        """
        Extract head-to-head features via SPARQL for a specific matchup.
        
        This queries historical matches between the two teams.
        """
        home_uri = self.kg._get_team_uri(home_team)
        away_uri = self.kg._get_team_uri(away_team)
        
        query = f"""
        SELECT ?match ?date ?homeGoals ?awayGoals ?result ?matchHome ?matchAway
        WHERE {{
            ?match a schema:SportsEvent .
            ?match schema:startDate ?date .
            ?match schema:homeTeam ?matchHome .
            ?match schema:awayTeam ?matchAway .
            ?match striko:homeGoals ?homeGoals .
            ?match striko:awayGoals ?awayGoals .
            ?match striko:result ?result .
            FILTER (?date < "{before_date}"^^xsd:dateTime)
            FILTER (
                (?matchHome = <{home_uri}> && ?matchAway = <{away_uri}>) ||
                (?matchHome = <{away_uri}> && ?matchAway = <{home_uri}>)
            )
        }}
        ORDER BY DESC(?date)
        LIMIT 10
        """
        
        h2h_matches = []
        for row in self.graph.query(query, initNs={"schema": SCHEMA, "striko": STRIKO, "xsd": XSD}):
            is_home = str(row.matchHome) == str(home_uri)
            h2h_matches.append({
                'date': str(row.date),
                'home_goals': int(row.homeGoals) if is_home else int(row.awayGoals),
                'away_goals': int(row.awayGoals) if is_home else int(row.homeGoals),
                'result': str(row.result),
                'is_home': is_home
            })
        
        features = {
            'H2H_TotalMatches': len(h2h_matches),
            'H2H_HomeWins': 0,
            'H2H_Draws': 0,
            'H2H_AwayWins': 0,
            'H2H_AvgGoalsHome': 0.0,
            'H2H_AvgGoalsAway': 0.0,
            'H2H_AvgTotalGoals': 0.0,
        }
        
        if h2h_matches:
            for match in h2h_matches:
                if match['is_home']:
                    if match['result'] == 'H':
                        features['H2H_HomeWins'] += 1
                    elif match['result'] == 'D':
                        features['H2H_Draws'] += 1
                    else:
                        features['H2H_AwayWins'] += 1
                else:
                    if match['result'] == 'A':
                        features['H2H_HomeWins'] += 1
                    elif match['result'] == 'D':
                        features['H2H_Draws'] += 1
                    else:
                        features['H2H_AwayWins'] += 1
                
                features['H2H_AvgGoalsHome'] += match['home_goals']
                features['H2H_AvgGoalsAway'] += match['away_goals']
            
            n = len(h2h_matches)
            features['H2H_AvgGoalsHome'] /= n
            features['H2H_AvgGoalsAway'] /= n
            features['H2H_AvgTotalGoals'] = features['H2H_AvgGoalsHome'] + features['H2H_AvgGoalsAway']
            features['H2H_HomeWinRate'] = features['H2H_HomeWins'] / n
        else:
            features['H2H_HomeWinRate'] = 0.5
        
        return features
    
    def extract_goal_efficiency_sparql(self, team_name: str, before_date: str) -> Dict:
        """
        Extract goal efficiency features for a team via SPARQL.
        """
        team_uri = self.kg._get_team_uri(team_name)
        
        query_home = f"""
        SELECT (COUNT(?match) as ?totalMatches)
               (SUM(?homeGoals) as ?goalsScored)
               (SUM(?awayGoals) as ?goalsConceded)
               (SUM(IF(?awayGoals = 0, 1, 0)) as ?cleanSheets)
        WHERE {{
            ?match a schema:SportsEvent .
            ?match schema:homeTeam <{team_uri}> .
            ?match schema:startDate ?date .
            ?match striko:homeGoals ?homeGoals .
            ?match striko:awayGoals ?awayGoals .
            FILTER (?date < "{before_date}"^^xsd:dateTime)
        }}
        """
        
        query_away = f"""
        SELECT (COUNT(?match) as ?totalMatches)
               (SUM(?awayGoals) as ?goalsScored)
               (SUM(?homeGoals) as ?goalsConceded)
               (SUM(IF(?homeGoals = 0, 1, 0)) as ?cleanSheets)
        WHERE {{
            ?match a schema:SportsEvent .
            ?match schema:awayTeam <{team_uri}> .
            ?match schema:startDate ?date .
            ?match striko:homeGoals ?homeGoals .
            ?match striko:awayGoals ?awayGoals .
            FILTER (?date < "{before_date}"^^xsd:dateTime)
        }}
        """
        
        features = {
            'GoalsScoredPerGame': 0.0,
            'GoalsConcededPerGame': 0.0,
            'CleanSheetRate': 0.0,
            'MatchesPlayed': 0,
        }
        
        total_matches = 0
        total_scored = 0
        total_conceded = 0
        total_clean_sheets = 0
        
        for row in self.graph.query(query_home, initNs={"schema": SCHEMA, "striko": STRIKO, "xsd": XSD}):
            if row.totalMatches:
                total_matches += int(row.totalMatches)
                total_scored += int(row.goalsScored) if row.goalsScored else 0
                total_conceded += int(row.goalsConceded) if row.goalsConceded else 0
                total_clean_sheets += int(row.cleanSheets) if row.cleanSheets else 0
        
        for row in self.graph.query(query_away, initNs={"schema": SCHEMA, "striko": STRIKO, "xsd": XSD}):
            if row.totalMatches:
                total_matches += int(row.totalMatches)
                total_scored += int(row.goalsScored) if row.goalsScored else 0
                total_conceded += int(row.goalsConceded) if row.goalsConceded else 0
                total_clean_sheets += int(row.cleanSheets) if row.cleanSheets else 0
        
        if total_matches > 0:
            features['MatchesPlayed'] = total_matches
            features['GoalsScoredPerGame'] = total_scored / total_matches
            features['GoalsConcededPerGame'] = total_conceded / total_matches
            features['CleanSheetRate'] = total_clean_sheets / total_matches
        
        return features
    
    def extract_all_features(self) -> pd.DataFrame:
        """
        Extract all features from the Knowledge Graph via SPARQL and merge them.
        
        Returns a DataFrame ready for ML model training.
        """
        print("Extracting features from Knowledge Graph via SPARQL...")
        
        print("  Extracting Elo features...")
        elo_df = self.extract_elo_features()
        
        print("  Extracting form features...")
        form_df = self.extract_form_features()
        
        print("  Extracting odds features...")
        odds_df = self.extract_odds_features()
        
        print("  Extracting match results...")
        results_df = self.extract_match_results()
        
        print("  Merging features...")
        if len(elo_df) == 0:
            print("  Warning: No Elo data found in knowledge graph")
            return pd.DataFrame()
        
        df = elo_df.copy()
        
        if len(form_df) > 0:
            form_cols = [c for c in form_df.columns if c != 'match_uri']
            df = df.merge(form_df[['match_uri'] + form_cols], on='match_uri', how='left')
        
        if len(odds_df) > 0:
            odds_cols = [c for c in odds_df.columns if c != 'match_uri']
            df = df.merge(odds_df[['match_uri'] + odds_cols], on='match_uri', how='left')
        
        if len(results_df) > 0:
            results_cols = [c for c in results_df.columns if c != 'match_uri']
            df = df.merge(results_df[['match_uri'] + results_cols], on='match_uri', how='left')
        
        df['MatchDate'] = pd.to_datetime(df['MatchDate'])
        df = df.sort_values('MatchDate').reset_index(drop=True)
        
        df['DayOfWeek'] = df['MatchDate'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['Month'] = df['MatchDate'].dt.month
        
        print(f"  Extracted {len(df)} matches with {len(df.columns)} features")
        
        return df


def load_data_to_kg_and_extract_features(data_path: str = 'data/raw/matches.csv',
                                          limit: Optional[int] = None) -> pd.DataFrame:
    """
    Full pipeline: Load CSV -> Knowledge Graph -> SPARQL Feature Extraction.
    
    This implements the proposal's data flow:
    CSV -> Knowledge Graph (RDF) -> SPARQL queries -> Feature DataFrame
    """
    from data_preprocessing import DataPreprocessor
    
    print("=" * 60)
    print("SPARQL-BASED FEATURE EXTRACTION PIPELINE")
    print("=" * 60)
    
    print("\n1. Loading and preprocessing data...")
    preprocessor = DataPreprocessor(data_path)
    clean_data = preprocessor.preprocess()
    
    if limit:
        clean_data = clean_data.tail(limit).copy()
        print(f"   Using {len(clean_data)} most recent matches")
    
    print("\n2. Building Knowledge Graph...")
    kg = FootballKnowledgeGraph()
    kg.load_from_dataframe(clean_data)
    
    print("\n3. Extracting features via SPARQL...")
    extractor = SPARQLFeatureExtractor(kg)
    featured_data = extractor.extract_all_features()
    
    print("\n4. Adding form data from original dataset...")
    form_cols = ['Form3Home', 'Form5Home', 'Form3Away', 'Form5Away']
    for col in form_cols:
        if col in clean_data.columns and col not in featured_data.columns:
            date_team_map = clean_data.set_index(['MatchDate', 'HomeTeam', 'AwayTeam'])[col].to_dict()
            featured_data[col] = featured_data.apply(
                lambda row: date_team_map.get((row['MatchDate'], row['HomeTeam'], row['AwayTeam']), 0.0),
                axis=1
            )
    
    print(f"\n✓ Feature extraction complete!")
    print(f"  Dataset shape: {featured_data.shape}")
    
    return featured_data


def main():
    """Demo of SPARQL-based feature extraction."""
    kg_path = Path("data/knowledge_graph/football_kg.ttl")
    
    if kg_path.exists():
        print("Loading existing Knowledge Graph...")
        kg = FootballKnowledgeGraph()
        kg.load(str(kg_path))
        
        extractor = SPARQLFeatureExtractor(kg)
        features = extractor.extract_all_features()
        
        print(f"\nExtracted features shape: {features.shape}")
        print(f"\nFeature columns: {list(features.columns)}")
        
        if len(features) > 0:
            print(f"\nSample data:")
            print(features.head())
    else:
        print("No existing Knowledge Graph found.")
        print("Run: python knowledge_graph.py first, or use load_data_to_kg_and_extract_features()")
        
        print("\nRunning full pipeline with 1000 sample matches...")
        features = load_data_to_kg_and_extract_features(limit=1000)
        
        print(f"\nExtracted features shape: {features.shape}")


if __name__ == '__main__':
    main()

