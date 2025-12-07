from knowledge_graph import FootballKnowledgeGraph, STRIKO, SCHEMA
from rdflib import Namespace
from typing import Dict, List


class KnowledgeGraphQueries:
    
    def __init__(self, kg: FootballKnowledgeGraph):
        self.kg = kg
        self.graph = kg.graph
    
    def get_team_matches(self, team_name: str, limit: int = 10) -> list:
        team_uri = self.kg._get_team_uri(team_name)
        
        query = f"""
        SELECT ?match ?date ?homeTeam ?awayTeam ?homeGoals ?awayGoals ?result
        WHERE {{
            ?match schema:homeTeam <{team_uri}> .
            ?match schema:startDate ?date .
            ?match schema:homeTeam ?homeTeam .
            ?match schema:awayTeam ?awayTeam .
            OPTIONAL {{ ?match striko:homeGoals ?homeGoals . }}
            OPTIONAL {{ ?match striko:awayGoals ?awayGoals . }}
            OPTIONAL {{ ?match striko:result ?result . }}
        }}
        ORDER BY DESC(?date)
        LIMIT {limit}
        """
        
        results = []
        for row in self.graph.query(query, initNs={"schema": SCHEMA, "striko": STRIKO}):
            results.append({
                'match': str(row.match),
                'date': str(row.date),
                'home_team': str(row.homeTeam),
                'away_team': str(row.awayTeam),
                'home_goals': int(row.homeGoals) if row.homeGoals else None,
                'away_goals': int(row.awayGoals) if row.awayGoals else None,
                'result': str(row.result) if row.result else None
            })
        
        query_away = f"""
        SELECT ?match ?date ?homeTeam ?awayTeam ?homeGoals ?awayGoals ?result
        WHERE {{
            ?match schema:awayTeam <{team_uri}> .
            ?match schema:startDate ?date .
            ?match schema:homeTeam ?homeTeam .
            ?match schema:awayTeam ?awayTeam .
            OPTIONAL {{ ?match striko:homeGoals ?homeGoals . }}
            OPTIONAL {{ ?match striko:awayGoals ?awayGoals . }}
            OPTIONAL {{ ?match striko:result ?result . }}
        }}
        ORDER BY DESC(?date)
        LIMIT {limit}
        """
        
        for row in self.graph.query(query_away, initNs={"schema": SCHEMA, "striko": STRIKO}):
            results.append({
                'match': str(row.match),
                'date': str(row.date),
                'home_team': str(row.homeTeam),
                'away_team': str(row.awayTeam),
                'home_goals': int(row.homeGoals) if row.homeGoals else None,
                'away_goals': int(row.awayGoals) if row.awayGoals else None,
                'result': str(row.result) if row.result else None
            })
        
        results.sort(key=lambda x: x['date'], reverse=True)
        return results[:limit]
    
    def get_match_predictions(self, home_team: str, away_team: str, match_date: str) -> dict:
        match_uri = self.kg._get_match_uri(home_team, away_team, match_date)
        
        query = f"""
        SELECT ?market ?outcome ?probability ?model
        WHERE {{
            ?prediction striko:forMatch <{match_uri}> .
            ?prediction striko:market ?market .
            ?prediction striko:predictedOutcome ?outcome .
            ?prediction striko:probability ?probability .
            ?prediction striko:modelName ?model .
        }}
        ORDER BY ?market
        """
        
        predictions = {}
        for row in self.graph.query(query, initNs={"striko": STRIKO}):
            market = str(row.market)
            if market not in predictions:
                predictions[market] = {}
            
            predictions[market][str(row.outcome)] = {
                'probability': float(row.probability),
                'model': str(row.model)
            }
        
        return predictions
    
    def find_value_bets(self, min_probability_diff: float = 0.1) -> list:
        query = f"""
        SELECT ?match ?homeTeam ?awayTeam ?date ?market ?predictedProb ?impliedProb ?odds
        WHERE {{
            ?match schema:homeTeam ?homeTeam .
            ?match schema:awayTeam ?awayTeam .
            ?match schema:startDate ?date .
            ?prediction striko:forMatch ?match .
            ?prediction striko:market ?market .
            ?prediction striko:probability ?predictedProb .
            ?odds striko:forMatch ?match .
            ?odds striko:impliedProbHome ?impliedProb .
            ?odds striko:homeOdds ?odds .
            FILTER (?predictedProb - ?impliedProb > {min_probability_diff})
        }}
        ORDER BY DESC(?predictedProb - ?impliedProb)
        LIMIT 20
        """
        
        value_bets = []
        for row in self.graph.query(query, initNs={"schema": SCHEMA, "striko": STRIKO}):
            value_bets.append({
                'match': str(row.match),
                'home_team': str(row.homeTeam),
                'away_team': str(row.awayTeam),
                'date': str(row.date),
                'market': str(row.market),
                'predicted_probability': float(row.predictedProb),
                'implied_probability': float(row.impliedProb),
                'odds': float(row.odds),
                'value': float(row.predictedProb) - float(row.impliedProb)
            })
        
        return value_bets
    
    def get_team_statistics(self, team_name: str) -> dict:
        team_uri = self.kg._get_team_uri(team_name)
        
        query_home = f"""
        SELECT (COUNT(?match) as ?totalMatches)
               (AVG(?homeGoals) as ?avgGoalsScored)
               (AVG(?awayGoals) as ?avgGoalsConceded)
        WHERE {{
            ?match schema:homeTeam <{team_uri}> .
            ?match striko:homeGoals ?homeGoals .
            ?match striko:awayGoals ?awayGoals .
        }}
        """
        
        query_away = f"""
        SELECT (COUNT(?match) as ?totalMatches)
               (AVG(?awayGoals) as ?avgGoalsScored)
               (AVG(?homeGoals) as ?avgGoalsConceded)
        WHERE {{
            ?match schema:awayTeam <{team_uri}> .
            ?match striko:homeGoals ?homeGoals .
            ?match striko:awayGoals ?awayGoals .
        }}
        """
        
        stats = {'team': team_name}
        
        for row in self.graph.query(query_home, initNs={"schema": SCHEMA, "striko": STRIKO}):
            stats['home_matches'] = int(row.totalMatches) if row.totalMatches else 0
            stats['home_avg_goals_scored'] = float(row.avgGoalsScored) if row.avgGoalsScored else 0
            stats['home_avg_goals_conceded'] = float(row.avgGoalsConceded) if row.avgGoalsConceded else 0
        
        for row in self.graph.query(query_away, initNs={"schema": SCHEMA, "striko": STRIKO}):
            stats['away_matches'] = int(row.totalMatches) if row.totalMatches else 0
            stats['away_avg_goals_scored'] = float(row.avgGoalsScored) if row.avgGoalsScored else 0
            stats['away_avg_goals_conceded'] = float(row.avgGoalsConceded) if row.avgGoalsConceded else 0
        
        return stats
    
    def get_upcoming_matches_with_predictions(self, limit: int = 10) -> list:
        query = f"""
        SELECT DISTINCT ?match ?homeTeam ?awayTeam ?date
        WHERE {{
            ?match schema:homeTeam ?homeTeam .
            ?match schema:awayTeam ?awayTeam .
            ?match schema:startDate ?date .
            ?prediction striko:forMatch ?match .
            FILTER (?date > NOW())
        }}
        ORDER BY ?date
        LIMIT {limit}
        """
        
        matches = []
        for row in self.graph.query(query, initNs={"schema": SCHEMA, "striko": STRIKO}):
            match_uri = str(row.match)
            predictions = self.get_match_predictions(
                str(row.homeTeam), str(row.awayTeam), str(row.date)
            )
            
            matches.append({
                'match': match_uri,
                'home_team': str(row.homeTeam),
                'away_team': str(row.awayTeam),
                'date': str(row.date),
                'predictions': predictions
            })
        
        return matches
    
    def get_overall_match_statistics(self) -> dict:
        query = f"""
        SELECT ?result (COUNT(?match) as ?matchCount)
        WHERE {{
            ?match striko:result ?result .
        }}
        GROUP BY ?result
        """
        
        total_matches = 0
        result_counts = {}
        
        for row in self.graph.query(query, initNs={"striko": STRIKO}):
            result = str(row.result)
            match_count = int(row.matchCount) if row.matchCount else 0
            result_counts[result] = match_count
            total_matches += match_count
        
        stats = {
            'total_matches': total_matches,
            'outcomes': {}
        }
        
        if total_matches > 0:
            for result, count in result_counts.items():
                stats['outcomes'][result] = {
                    'count': count,
                    'percentage': round((count / total_matches) * 100, 2)
                }
        
        return stats
