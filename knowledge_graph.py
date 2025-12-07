import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD, OWL
from rdflib.plugins.sparql import prepareQuery


STRIKO = Namespace("http://striko.football/betting/")
SCHEMA = Namespace("http://schema.org/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")

def create_graph() -> Graph:
    g = Graph()
    g.bind("striko", STRIKO)
    g.bind("schema", SCHEMA)
    g.bind("foaf", FOAF)
    g.bind("rdfs", RDFS)
    return g


class FootballKnowledgeGraph:    
    def __init__(self):
        self.graph = create_graph()
        self.teams = {}
        self.matches = {}
    
    def _get_team_uri(self, team_name: str) -> URIRef:
        if team_name not in self.teams:
            team_id = team_name.lower().replace(" ", "_").replace("'", "")
            self.teams[team_name] = STRIKO[f"team/{team_id}"]
        return self.teams[team_name]
    
    def _get_match_uri(self, home_team: str, away_team: str, match_date: str) -> URIRef:
        match_key = f"{home_team}_{away_team}_{match_date}"
        if match_key not in self.matches:
            match_id = match_key.lower().replace(" ", "_").replace("'", "").replace(":", "-")
            self.matches[match_key] = STRIKO[f"match/{match_id}"]
        return self.matches[match_key]
    
    def add_team(self, team_name: str, division: Optional[str] = None) -> URIRef:
        team_uri = self._get_team_uri(team_name)
        
        self.graph.add((team_uri, RDF.type, SCHEMA.SportsTeam))
        self.graph.add((team_uri, SCHEMA.name, Literal(team_name, datatype=XSD.string)))
        
        if division:
            self.graph.add((team_uri, STRIKO.division, Literal(division, datatype=XSD.string)))
        
        return team_uri
    
    def add_match(self, 
                  home_team: str, 
                  away_team: str, 
                  match_date: str,
                  division: Optional[str] = None,
                  home_goals: Optional[int] = None,
                  away_goals: Optional[int] = None,
                  result: Optional[str] = None,
                  home_elo: Optional[float] = None,
                  away_elo: Optional[float] = None,
                  form3_home: Optional[float] = None,
                  form5_home: Optional[float] = None,
                  form3_away: Optional[float] = None,
                  form5_away: Optional[float] = None) -> URIRef:
        match_uri = self._get_match_uri(home_team, away_team, match_date)
        home_uri = self._get_team_uri(home_team)
        away_uri = self._get_team_uri(away_team)
        
        self.graph.add((match_uri, RDF.type, SCHEMA.SportsEvent))
        self.graph.add((match_uri, SCHEMA.startDate, Literal(match_date, datatype=XSD.dateTime)))
        
        self.graph.add((match_uri, SCHEMA.homeTeam, home_uri))
        self.graph.add((match_uri, SCHEMA.awayTeam, away_uri))
        
        if division:
            self.graph.add((match_uri, STRIKO.division, Literal(division, datatype=XSD.string)))
        
        if home_goals is not None:
            self.graph.add((match_uri, STRIKO.homeGoals, Literal(home_goals, datatype=XSD.integer)))
        
        if away_goals is not None:
            self.graph.add((match_uri, STRIKO.awayGoals, Literal(away_goals, datatype=XSD.integer)))
        
        if result:
            self.graph.add((match_uri, STRIKO.result, Literal(result, datatype=XSD.string)))
        
        if home_elo is not None:
            self.graph.add((match_uri, STRIKO.homeElo, Literal(home_elo, datatype=XSD.float)))
        
        if away_elo is not None:
            self.graph.add((match_uri, STRIKO.awayElo, Literal(away_elo, datatype=XSD.float)))
        
        if form3_home is not None:
            self.graph.add((match_uri, STRIKO.form3Home, Literal(form3_home, datatype=XSD.float)))
        
        if form5_home is not None:
            self.graph.add((match_uri, STRIKO.form5Home, Literal(form5_home, datatype=XSD.float)))
        
        if form3_away is not None:
            self.graph.add((match_uri, STRIKO.form3Away, Literal(form3_away, datatype=XSD.float)))
        
        if form5_away is not None:
            self.graph.add((match_uri, STRIKO.form5Away, Literal(form5_away, datatype=XSD.float)))
        
        if home_goals is not None and away_goals is not None:
            total_goals = home_goals + away_goals
            self.graph.add((match_uri, STRIKO.totalGoals, Literal(total_goals, datatype=XSD.integer)))
            self.graph.add((match_uri, STRIKO.over25, Literal(total_goals > 2.5, datatype=XSD.boolean)))
            self.graph.add((match_uri, STRIKO.btts, Literal(
                (home_goals > 0 and away_goals > 0), datatype=XSD.boolean
            )))
        
        return match_uri
    
    def add_odds(self, 
                 match_uri: URIRef,
                 home_odds: Optional[float] = None,
                 draw_odds: Optional[float] = None,
                 away_odds: Optional[float] = None) -> URIRef:
        odds_uri = STRIKO[f"odds/{match_uri.split('/')[-1]}"]
        
        self.graph.add((odds_uri, RDF.type, STRIKO.BettingOdds))
        self.graph.add((odds_uri, STRIKO.forMatch, match_uri))
        
        if home_odds is not None:
            self.graph.add((odds_uri, STRIKO.homeOdds, Literal(home_odds, datatype=XSD.float)))
            implied_prob = 1.0 / home_odds
            self.graph.add((odds_uri, STRIKO.impliedProbHome, Literal(implied_prob, datatype=XSD.float)))
        
        if draw_odds is not None:
            self.graph.add((odds_uri, STRIKO.drawOdds, Literal(draw_odds, datatype=XSD.float)))
            implied_prob = 1.0 / draw_odds
            self.graph.add((odds_uri, STRIKO.impliedProbDraw, Literal(implied_prob, datatype=XSD.float)))
        
        if away_odds is not None:
            self.graph.add((odds_uri, STRIKO.awayOdds, Literal(away_odds, datatype=XSD.float)))
            implied_prob = 1.0 / away_odds
            self.graph.add((odds_uri, STRIKO.impliedProbAway, Literal(implied_prob, datatype=XSD.float)))
        
        return odds_uri
    
    def add_prediction(self,
                      match_uri: URIRef,
                      market: str,
                      prediction: str,
                      probability: float,
                      model_name: str = "striko_ml",
                      uncertainty_lower: Optional[float] = None,
                      uncertainty_upper: Optional[float] = None,
                      uncertainty_std: Optional[float] = None) -> URIRef:
        pred_uri = STRIKO[f"prediction/{match_uri.split('/')[-1]}_{market}"]
        
        self.graph.add((pred_uri, RDF.type, STRIKO.Prediction))
        self.graph.add((pred_uri, STRIKO.forMatch, match_uri))
        self.graph.add((pred_uri, STRIKO.market, Literal(market, datatype=XSD.string)))
        self.graph.add((pred_uri, STRIKO.predictedOutcome, Literal(prediction, datatype=XSD.string)))
        self.graph.add((pred_uri, STRIKO.probability, Literal(probability, datatype=XSD.float)))
        self.graph.add((pred_uri, STRIKO.modelName, Literal(model_name, datatype=XSD.string)))
        self.graph.add((pred_uri, STRIKO.predictionDate, Literal(
            datetime.now().isoformat(), datatype=XSD.dateTime
        )))
        
        if uncertainty_lower is not None and uncertainty_upper is not None:
            uq_uri = STRIKO[f"uncertainty/{match_uri.split('/')[-1]}_{market}"]
            self.graph.add((uq_uri, RDF.type, STRIKO.UncertaintyInterval))
            self.graph.add((pred_uri, STRIKO.hasUncertaintyInterval, uq_uri))
            self.graph.add((uq_uri, STRIKO.uncertaintyLower, Literal(uncertainty_lower, datatype=XSD.float)))
            self.graph.add((uq_uri, STRIKO.uncertaintyUpper, Literal(uncertainty_upper, datatype=XSD.float)))
            self.graph.add((uq_uri, STRIKO.confidenceLevel, Literal(0.9, datatype=XSD.float)))
            
            if uncertainty_std is not None:
                self.graph.add((uq_uri, STRIKO.uncertaintyStd, Literal(uncertainty_std, datatype=XSD.float)))
        
        return pred_uri
    
    def add_match_predictions(self,
                             match_uri: URIRef,
                             predictions: Dict[str, Dict]) -> List[URIRef]:
        """Add predictions with uncertainty intervals (UQ360) to the knowledge graph."""
        pred_uris = []
        
        for market_name, pred_data in predictions.items():
            if market_name == 'match_result':
                probs = pred_data['probabilities']
                pred_outcome = pred_data['prediction']
                prob = probs.get(pred_outcome, 0.0)
                
                ci_lower = pred_data.get('confidence_intervals', {}).get(pred_outcome, [prob, prob])[0] if 'confidence_intervals' in pred_data else None
                ci_upper = pred_data.get('confidence_intervals', {}).get(pred_outcome, [prob, prob])[1] if 'confidence_intervals' in pred_data else None
                uncert = pred_data.get('uncertainty', {}).get(pred_outcome) if 'uncertainty' in pred_data else None
                
                pred_uri = self.add_prediction(
                    match_uri, market_name, pred_outcome, prob,
                    uncertainty_lower=ci_lower,
                    uncertainty_upper=ci_upper,
                    uncertainty_std=uncert
                )
                pred_uris.append(pred_uri)
                
                for outcome, prob_val in probs.items():
                    outcome_uri = STRIKO[f"prediction/{match_uri.split('/')[-1]}_{market_name}_{outcome}"]
                    self.graph.add((outcome_uri, RDF.type, STRIKO.Prediction))
                    self.graph.add((outcome_uri, STRIKO.forMatch, match_uri))
                    self.graph.add((outcome_uri, STRIKO.market, Literal(market_name, datatype=XSD.string)))
                    self.graph.add((outcome_uri, STRIKO.predictedOutcome, Literal(outcome, datatype=XSD.string)))
                    self.graph.add((outcome_uri, STRIKO.probability, Literal(prob_val, datatype=XSD.float)))
                    self.graph.add((outcome_uri, STRIKO.modelName, Literal("striko_ml", datatype=XSD.string)))
                    
                    if 'confidence_intervals' in pred_data and outcome in pred_data['confidence_intervals']:
                        ci = pred_data['confidence_intervals'][outcome]
                        uq_uri = STRIKO[f"uncertainty/{match_uri.split('/')[-1]}_{market_name}_{outcome}"]
                        self.graph.add((uq_uri, RDF.type, STRIKO.UncertaintyInterval))
                        self.graph.add((outcome_uri, STRIKO.hasUncertaintyInterval, uq_uri))
                        self.graph.add((uq_uri, STRIKO.uncertaintyLower, Literal(ci[0], datatype=XSD.float)))
                        self.graph.add((uq_uri, STRIKO.uncertaintyUpper, Literal(ci[1], datatype=XSD.float)))
                        self.graph.add((uq_uri, STRIKO.confidenceLevel, Literal(0.9, datatype=XSD.float)))
            
            else:
                pred_outcome = str(pred_data['prediction'])
                prob = pred_data.get('probability', 0.0)
                
                ci_lower = pred_data.get('confidence_interval', [None, None])[0] if 'confidence_interval' in pred_data else None
                ci_upper = pred_data.get('confidence_interval', [None, None])[1] if 'confidence_interval' in pred_data else None
                uncert = pred_data.get('uncertainty')
                
                pred_uri = self.add_prediction(
                    match_uri, market_name, pred_outcome, prob,
                    uncertainty_lower=ci_lower,
                    uncertainty_upper=ci_upper,
                    uncertainty_std=uncert
                )
                pred_uris.append(pred_uri)
        
        return pred_uris
    
    def load_from_dataframe(self, df: pd.DataFrame, limit: Optional[int] = None):
        print(f"Loading matches into knowledge graph...")
        
        df_to_load = df.head(limit) if limit else df
        
        for idx, row in df_to_load.iterrows():
            match_date = pd.to_datetime(row['MatchDate']).isoformat()
            
            self.add_team(row['HomeTeam'], row.get('Division'))
            self.add_team(row['AwayTeam'], row.get('Division'))
            
            match_uri = self.add_match(
                home_team=row['HomeTeam'],
                away_team=row['AwayTeam'],
                match_date=match_date,
                division=row.get('Division'),
                home_goals=int(row.get('FTHome', 0)) if pd.notna(row.get('FTHome')) else None,
                away_goals=int(row.get('FTAway', 0)) if pd.notna(row.get('FTAway')) else None,
                result=row.get('FTResult'),
                home_elo=float(row.get('HomeElo')) if pd.notna(row.get('HomeElo')) else None,
                away_elo=float(row.get('AwayElo')) if pd.notna(row.get('AwayElo')) else None,
                form3_home=float(row.get('Form3Home')) if pd.notna(row.get('Form3Home')) else None,
                form5_home=float(row.get('Form5Home')) if pd.notna(row.get('Form5Home')) else None,
                form3_away=float(row.get('Form3Away')) if pd.notna(row.get('Form3Away')) else None,
                form5_away=float(row.get('Form5Away')) if pd.notna(row.get('Form5Away')) else None
            )
            
            if pd.notna(row.get('OddHome')):
                self.add_odds(
                    match_uri,
                    home_odds=float(row.get('OddHome')),
                    draw_odds=float(row.get('OddDraw')) if pd.notna(row.get('OddDraw')) else None,
                    away_odds=float(row.get('OddAway')) if pd.notna(row.get('OddAway')) else None
                )
            
            if (idx + 1) % 1000 == 0:
                print(f"  Loaded {idx + 1} matches...")
        
        print(f"Loaded {len(df_to_load)} matches into knowledge graph")
        print(f"Total triples: {len(self.graph)}")
    
    def query(self, sparql_query: str) -> List[Dict]:
        results = self.graph.query(sparql_query)
        return [dict(row) for row in results]
    
    def get_match_predictions(self, home_team: str, away_team: str, match_date: str) -> Dict:
        match_uri = self._get_match_uri(home_team, away_team, match_date)
        
        query = f"""
        SELECT ?pred ?market ?outcome ?prob ?model
        WHERE {{
            ?pred striko:forMatch <{match_uri}> .
            ?pred striko:market ?market .
            ?pred striko:predictedOutcome ?outcome .
            ?pred striko:probability ?prob .
            ?pred striko:modelName ?model .
        }}
        """
        
        predictions = {}
        for pred, market, outcome, prob, model in self.graph.query(query, initNs={"striko": STRIKO}):
            market_str = str(market)
            if market_str not in predictions:
                predictions[market_str] = {}
            
            predictions[market_str][str(outcome)] = {
                'probability': float(prob),
                'model': str(model)
            }
        
        return predictions
    
    def save(self, filepath: str, format: str = "turtle"):
        self.graph.serialize(destination=filepath, format=format)
        print(f"Saved knowledge graph to {filepath} ({len(self.graph)} triples)")
    
    def load(self, filepath: str, format: str = "turtle"):
        self.graph.parse(filepath, format=format)
        print(f"Loaded knowledge graph from {filepath} ({len(self.graph)} triples)")
    
    def get_stats(self) -> Dict:
        team_count = len(set(self.graph.subjects(RDF.type, SCHEMA.SportsTeam)))
        match_count = len(set(self.graph.subjects(RDF.type, SCHEMA.SportsEvent)))
        prediction_count = len(set(self.graph.subjects(RDF.type, STRIKO.Prediction)))
        total_triples = len(self.graph)
        
        return {
            'teams': team_count,
            'matches': match_count,
            'predictions': prediction_count,
            'total_triples': total_triples
        }


def main():
    from feature_engineering import FeatureEngineer
    from data_preprocessing import DataPreprocessor
    from predict import BettingPredictor
    
    print("=" * 80)
    print("BUILDING RDF KNOWLEDGE GRAPH WITH PREDICTIONS")
    print("=" * 80)
    
    print("\n1. Loading data...")
    preprocessor = DataPreprocessor('data/raw/matches.csv')
    clean_data = preprocessor.preprocess()
    
    print("2. Sampling data for faster knowledge graph creation...")
    sample_size = min(5000, len(clean_data))
    clean_data = clean_data.tail(sample_size).copy()
    print(f"   Using {len(clean_data)} most recent matches")
    
    print("3. Engineering features (skipping slow H2H features for speed)...")
    engineer = FeatureEngineer(clean_data)
    featured_data = engineer.create_all_features(
        include_goal_features=True,
        include_h2h_features=False,
        include_goal_efficiency=False
    )
    
    print("4. Creating knowledge graph...")
    kg = FootballKnowledgeGraph()
    
    print("5. Loading matches into graph...")
    kg.load_from_dataframe(featured_data)
    
    print("\n6. Adding ML predictions with uncertainty intervals...")
    try:
        predictor = BettingPredictor()
        
        prediction_sample = featured_data.tail(100).copy()
        predictions_added = 0
        
        for idx, row in prediction_sample.iterrows():
            try:
                match_data = prediction_sample.loc[[idx]]
                home_team = row['HomeTeam']
                away_team = row['AwayTeam']
                match_date = pd.to_datetime(row['MatchDate']).isoformat()
                
                predictions = predictor.predict_match(match_data, home_team, away_team)
                
                match_uri = kg._get_match_uri(home_team, away_team, match_date)
                kg.add_match_predictions(match_uri, predictions)
                predictions_added += 1
                
            except Exception as e:
                continue
        
        print(f"   Added predictions for {predictions_added} matches (with UQ360 uncertainty intervals)")
        
    except FileNotFoundError:
        print("   ML models not found. Run 'python model_training.py' first.")
        print("   Knowledge graph will be saved without predictions.")
    
    print("\n7. Saving knowledge graph...")
    output_dir = Path("data/knowledge_graph")
    output_dir.mkdir(parents=True, exist_ok=True)
    kg.save(str(output_dir / "football_kg.ttl"), format="turtle")
    
    print("\n8. Knowledge Graph Statistics:")
    stats = kg.get_stats()
    print(f"   Teams: {stats['teams']}")
    print(f"   Matches: {stats['matches']}")
    print(f"   Predictions: {stats['predictions']}")
    print(f"   Total Triples: {stats['total_triples']}")
    
    print("\n" + "=" * 80)
    print("Knowledge graph created successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()