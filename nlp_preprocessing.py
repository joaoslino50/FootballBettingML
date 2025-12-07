"""
NLP Preprocessing Module for Football Betting Chatbot

Uses spaCy for entity extraction and intent classification as specified in the proposal.
Extracts team names, dates, and betting-related intents from user queries.

Usage:
    from nlp_preprocessing import NLPPreprocessor
    
    nlp = NLPPreprocessor()
    result = nlp.process_query("Who will win between Liverpool and Arsenal?")
    # Returns: {'teams': ['Liverpool', 'Arsenal'], 'intent': 'prediction', ...}
"""

import re
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")


class NLPPreprocessor:
    """NLP preprocessing for football betting queries using spaCy."""
    
    INTENT_PATTERNS = {
        'prediction': [
            r'\b(who will win|predict|prediction|probability|chance|likely|forecast)\b',
            r'\b(win|beat|defeat|lose|draw)\b.*\?',
            r'\bwhat.*(happen|result|outcome)\b',
        ],
        'statistics': [
            r'\b(statistics|stats|average|mean|total|how many|how much)\b',
            r'\b(goals|scored|conceded|matches|games|points)\b',
            r'\b(form|performance|record|history)\b',
        ],
        'value_bet': [
            r'\b(value bet|value|odds|betting|bet|wager)\b',
            r'\b(expected value|ev|edge|profit)\b',
            r'\b(worth betting|good bet|bad bet)\b',
        ],
        'comparison': [
            r'\b(compare|vs|versus|against|between)\b',
            r'\b(better|worse|stronger|weaker)\b',
            r'\b(head to head|h2h)\b',
        ],
        'team_info': [
            r'\b(tell me about|info|information|about)\b',
            r'\b(who is|what is|describe)\b',
        ],
    }
    
    BETTING_KEYWORDS = {
        'match_result': ['win', 'lose', 'draw', 'winner', 'result', '1x2'],
        'over_under': ['over', 'under', 'goals', '2.5', '1.5', '3.5', 'total goals'],
        'btts': ['btts', 'both teams', 'score', 'both teams to score'],
        'clean_sheet': ['clean sheet', 'shut out', 'nil', 'no goals'],
    }
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = None
        self.team_names = set()
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                print(f"spaCy model '{model_name}' not found.")
                print(f"Download it with: python -m spacy download {model_name}")
                self.nlp = None
    
    def load_team_names(self, team_names: List[str]):
        """Load known team names for fuzzy matching."""
        self.team_names = set(team_names)
    
    def load_teams_from_dataframe(self, df):
        """Load team names from a matches DataFrame."""
        if 'HomeTeam' in df.columns:
            self.team_names.update(df['HomeTeam'].unique())
        if 'AwayTeam' in df.columns:
            self.team_names.update(df['AwayTeam'].unique())
        print(f"Loaded {len(self.team_names)} team names for entity extraction")
    
    def extract_entities_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy NER."""
        entities = {
            'teams': [],
            'dates': [],
            'numbers': [],
            'locations': [],
        }
        
        if self.nlp is None:
            return entities
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'GPE', 'PERSON']:
                matched_team = self._fuzzy_match_team(ent.text)
                if matched_team:
                    entities['teams'].append(matched_team)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            elif ent.label_ in ['CARDINAL', 'MONEY', 'PERCENT']:
                entities['numbers'].append(ent.text)
            elif ent.label_ == 'LOC':
                entities['locations'].append(ent.text)
        
        return entities
    
    def extract_teams_pattern(self, text: str) -> List[str]:
        """Extract team names using pattern matching and fuzzy matching."""
        found_teams = []
        
        text_lower = text.lower()
        
        vs_patterns = [
            r'(\w+(?:\s+\w+)*)\s+(?:vs\.?|versus|against|v\.?)\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+(?:and|&)\s+(\w+(?:\s+\w+)*)',
            r'between\s+(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)',
        ]
        
        for pattern in vs_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                for team in match:
                    matched = self._fuzzy_match_team(team.strip())
                    if matched:
                        found_teams.append(matched)
        
        words = text.split()
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):
                phrase = ' '.join(words[i:j])
                matched = self._fuzzy_match_team(phrase)
                if matched and matched not in found_teams:
                    found_teams.append(matched)
        
        return list(dict.fromkeys(found_teams))
    
    def _fuzzy_match_team(self, text: str, threshold: float = 0.7) -> Optional[str]:
        """Fuzzy match a text string to known team names."""
        text_lower = text.lower().strip()
        
        for team in self.team_names:
            if text_lower == team.lower():
                return team
        
        for team in self.team_names:
            if text_lower in team.lower() or team.lower() in text_lower:
                return team
        
        best_match = None
        best_score = threshold
        
        for team in self.team_names:
            score = SequenceMatcher(None, text_lower, team.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = team
        
        return best_match
    
    def classify_intent(self, text: str) -> str:
        """Classify the user's intent from the query text."""
        text_lower = text.lower()
        
        intent_scores = {}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            intent_scores[intent] = score
        
        if max(intent_scores.values()) == 0:
            if '?' in text:
                return 'prediction'
            return 'general'
        
        return max(intent_scores, key=intent_scores.get)
    
    def extract_betting_market(self, text: str) -> Optional[str]:
        """Extract which betting market the user is asking about."""
        text_lower = text.lower()
        
        for market, keywords in self.BETTING_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return market
        
        return 'match_result'
    
    def extract_odds(self, text: str) -> Dict[str, float]:
        """Extract betting odds from text."""
        odds = {}
        
        patterns = [
            r'home[:\s]+(\d+\.?\d*)',
            r'draw[:\s]+(\d+\.?\d*)',
            r'away[:\s]+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*[,/]\s*(\d+\.?\d*)\s*[,/]\s*(\d+\.?\d*)',
        ]
        
        for pattern in patterns[:3]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                key = pattern.split('[')[0]
                odds[key] = float(match.group(1))
        
        triple_match = re.search(patterns[3], text)
        if triple_match and len(odds) == 0:
            odds['home'] = float(triple_match.group(1))
            odds['draw'] = float(triple_match.group(2))
            odds['away'] = float(triple_match.group(3))
        
        return odds
    
    def process_query(self, text: str) -> Dict:
        """
        Full NLP preprocessing pipeline for a user query.
        
        Returns:
            Dict with extracted entities, intent, and metadata
        """
        result = {
            'original_query': text,
            'teams': [],
            'intent': 'general',
            'betting_market': None,
            'dates': [],
            'odds': {},
            'entities': {},
        }
        
        if self.nlp:
            result['entities'] = self.extract_entities_spacy(text)
            result['teams'].extend(result['entities'].get('teams', []))
            result['dates'] = result['entities'].get('dates', [])
        
        pattern_teams = self.extract_teams_pattern(text)
        for team in pattern_teams:
            if team not in result['teams']:
                result['teams'].append(team)
        
        result['intent'] = self.classify_intent(text)
        
        if result['intent'] in ['prediction', 'value_bet', 'comparison']:
            result['betting_market'] = self.extract_betting_market(text)
        
        result['odds'] = self.extract_odds(text)
        
        return result
    
    def enhance_query_for_agent(self, text: str) -> str:
        """
        Enhance the original query with extracted entity information.
        
        This provides additional context to the LangChain agent.
        """
        processed = self.process_query(text)
        
        enhanced_parts = [text]
        
        if processed['teams']:
            enhanced_parts.append(f"\n[Extracted teams: {', '.join(processed['teams'])}]")
        
        if processed['intent'] != 'general':
            enhanced_parts.append(f"[Intent: {processed['intent']}]")
        
        if processed['betting_market']:
            enhanced_parts.append(f"[Market: {processed['betting_market']}]")
        
        if processed['odds']:
            odds_str = ', '.join(f"{k}: {v}" for k, v in processed['odds'].items())
            enhanced_parts.append(f"[Odds: {odds_str}]")
        
        return ' '.join(enhanced_parts)


def main():
    """Demo of NLP preprocessing."""
    nlp_processor = NLPPreprocessor()
    
    demo_teams = [
        'Manchester United', 'Manchester City', 'Liverpool', 'Arsenal', 
        'Chelsea', 'Tottenham', 'Real Madrid', 'Barcelona', 'Bayern Munich',
        'Porto', 'Benfica', 'Sporting', 'Nice', 'PSG', 'Valencia'
    ]
    nlp_processor.load_team_names(demo_teams)
    
    test_queries = [
        "Who will win between Liverpool and Arsenal?",
        "What's the probability of Real Madrid beating Barcelona?",
        "Show me statistics for Manchester United",
        "Is there value in betting on Porto at 1.25 vs Nice at 7.25?",
        "Will both teams score in the Chelsea vs Tottenham match?",
        "What are the chances of over 2.5 goals in Bayern vs PSG?",
        "Compare Manchester City and Liverpool head to head",
    ]
    
    print("=" * 60)
    print("NLP PREPROCESSING DEMO")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = nlp_processor.process_query(query)
        print(f"  Teams: {result['teams']}")
        print(f"  Intent: {result['intent']}")
        print(f"  Market: {result['betting_market']}")
        if result['odds']:
            print(f"  Odds: {result['odds']}")
        print("-" * 40)


if __name__ == '__main__':
    main()

