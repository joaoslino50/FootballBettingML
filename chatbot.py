"""
Football Betting Chatbot - Local LLM Agent Interface

SETUP INSTRUCTIONS:
1. Install Ollama: https://ollama.ai
2. Pull a recommended model (supports function calling):
   - llama3.2 (recommended): ollama pull llama3.2
   - mistral: ollama pull mistral
   - qwen2.5: ollama pull qwen2.5
3. Start Ollama server: ollama serve (runs on http://localhost:11434 by default)
4. Install Python dependencies: pip install -r requirements.txt
5. Run the chatbot: streamlit run chatbot.py

PREREQUISITES:
- Trained models in models/ directory (run: python model_training.py)
- Knowledge graph at data/knowledge_graph/football_kg.ttl (run: python knowledge_graph.py)
- Raw data at data/raw/matches.csv

RECOMMENDED MODEL:
llama3.2 (3B or 1B) - Best balance of performance and function calling support

NLP PREPROCESSING:
Uses spaCy for entity extraction (team names) and intent classification
as specified in the project proposal.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_ollama import OllamaLLM as ChatOllama

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    USE_LANGGRAPH = False
except ImportError:
    try:
        from langchain.agents import create_react_agent as create_agent_new
        USE_LANGGRAPH = True
    except ImportError:
        from langgraph.prebuilt import create_react_agent as create_agent_new
        USE_LANGGRAPH = True
    AgentExecutor = None

from predict import BettingPredictor
from knowledge_graph import FootballKnowledgeGraph
from sparql_queries import KnowledgeGraphQueries
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from nlp_preprocessing import NLPPreprocessor


@st.cache_resource
def load_predictor():
    try:
        return BettingPredictor()
    except (FileNotFoundError, ValueError) as e:
        return None


@st.cache_resource
def load_knowledge_graph():
    kg_path = Path("data/knowledge_graph/football_kg.ttl")
    if not kg_path.exists():
        return None
    
    kg = FootballKnowledgeGraph()
    kg.load(str(kg_path))
    return KnowledgeGraphQueries(kg)


@st.cache_data
def get_featured_data():
    preprocessor = DataPreprocessor('data/raw/matches.csv')
    clean_data = preprocessor.preprocess()
    engineer = FeatureEngineer(clean_data)
    featured_data = engineer.create_all_features(include_goal_features=True)
    return featured_data

@st.cache_data
def get_recent_featured_data_for_prediction(limit=1000):
    preprocessor = DataPreprocessor('data/raw/matches.csv')
    clean_data = preprocessor.preprocess()
    recent_data = clean_data.tail(limit).copy()
    engineer = FeatureEngineer(recent_data)
    featured_data = engineer.create_all_features(include_goal_features=True)
    return featured_data


@st.cache_resource
def load_nlp_preprocessor():
    """Load NLP preprocessor with spaCy for entity extraction."""
    nlp = NLPPreprocessor()
    
    try:
        preprocessor = DataPreprocessor('data/raw/matches.csv')
        clean_data = preprocessor.preprocess()
        nlp.load_teams_from_dataframe(clean_data)
    except Exception as e:
        print(f"Warning: Could not load team names: {e}")
    
    return nlp


def preprocess_query_with_spacy(query: str, nlp_processor: NLPPreprocessor) -> Dict:
    """
    Pre-process user query using spaCy for entity extraction.
    
    This implements the proposal requirement:
    "User queries are parsed using spaCy/NLTK to identify intents"
    """
    result = nlp_processor.process_query(query)
    return result


def get_team_from_data(team_name: str, featured_data: pd.DataFrame) -> Optional[str]:
    all_teams = set(featured_data['HomeTeam'].unique()) | set(featured_data['AwayTeam'].unique())
    team_lower = team_name.lower()
    
    for team in all_teams:
        if team_lower in team.lower() or team.lower() in team_lower:
            return team
    return None


@tool
def get_team_statistics(team_name: str) -> str:
    """Get team statistics including average goals scored, goals conceded, and match history.
    
    Args:
        team_name: Name of the team to query
    
    Returns:
        JSON string with team statistics
    """
    kg_queries = load_knowledge_graph()
    if kg_queries is None:
        return json.dumps({"error": "Knowledge graph not loaded. Please run knowledge_graph.py first."})
    
    try:
        stats = kg_queries.get_team_statistics(team_name)
        return json.dumps(stats, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get team statistics: {str(e)}"})


@tool
def get_team_matches(team_name: str, limit: int = 10) -> str:
    """Get recent matches for a team.
    
    Args:
        team_name: Name of the team
        limit: Maximum number of matches to return (default: 10)
    
    Returns:
        JSON string with match history
    """
    kg_queries = load_knowledge_graph()
    if kg_queries is None:
        return json.dumps({"error": "Knowledge graph not loaded. Please run knowledge_graph.py first."})
    
    try:
        matches = kg_queries.get_team_matches(team_name, limit=limit)
        return json.dumps(matches, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to get team matches: {str(e)}"})


@tool
def predict_match_result(home_team: str, away_team: str) -> str:
    """Predict the outcome of a match between two teams using ML models.
    
    Args:
        home_team: Name of the home team
        away_team: Name of the away team
    
    Returns:
        JSON string with predictions for all betting markets
    """
    predictor = load_predictor()
    if predictor is None:
        return json.dumps({"error": "Models not loaded. Please train models first."})
    
    try:
        featured_data = get_recent_featured_data_for_prediction(limit=2000)
        
        home_team_actual = get_team_from_data(home_team, featured_data)
        away_team_actual = get_team_from_data(away_team, featured_data)
        
        if not home_team_actual or not away_team_actual:
            return json.dumps({
                "error": f"Teams not found. Home: {home_team_actual}, Away: {away_team_actual}"
            })
        
        exact_match = featured_data[
            (featured_data['HomeTeam'] == home_team_actual) & 
            (featured_data['AwayTeam'] == away_team_actual)
        ].sort_values('MatchDate', ascending=False)
        
        if len(exact_match) > 0:
            match_data = exact_match.iloc[0:1].copy()
        else:
            reversed_match = featured_data[
                (featured_data['HomeTeam'] == away_team_actual) & 
                (featured_data['AwayTeam'] == home_team_actual)
            ].sort_values('MatchDate', ascending=False)
            
            if len(reversed_match) == 0:
                return json.dumps({
                    "error": f"No recent matches found between {home_team_actual} and {away_team_actual} in the last 2000 matches. Try teams that have played recently."
                })
            
            match_row = reversed_match.iloc[0:1].copy()
            match_data = match_row.copy()
            
            swap_pairs = [
                ('HomeTeam', 'AwayTeam'),
                ('HomeElo', 'AwayElo'),
                ('Form3Home', 'Form3Away'),
                ('Form5Home', 'Form5Away'),
                ('OddHome', 'OddAway'),
            ]
            
            for home_col, away_col in swap_pairs:
                if home_col in match_data.columns and away_col in match_data.columns:
                    match_data[home_col], match_data[away_col] = match_data[away_col].values, match_data[home_col].values
            
            match_data['HomeTeam'] = home_team_actual
            match_data['AwayTeam'] = away_team_actual
        
        predictions = predictor.predict_match(match_data, home_team_actual, away_team_actual)
        
        result = {
            "home_team": home_team_actual,
            "away_team": away_team_actual,
            "predictions": {}
        }
        
        for market, pred_data in predictions.items():
            if market == 'match_result':
                probs = pred_data['probabilities']
                outcome_map = {'H': f'{home_team_actual} (Home) wins', 
                              'D': 'Draw', 
                              'A': f'{away_team_actual} (Away) wins'}
                result["predictions"][market] = {
                    "predicted_outcome": pred_data['prediction'],
                    "predicted_outcome_description": outcome_map.get(pred_data['prediction'], pred_data['prediction']),
                    "probabilities": {
                        outcome_map.get(k, k): float(v) 
                        for k, v in probs.items()
                    },
                    "raw_probabilities": probs
                }
            else:
                market_descriptions = {
                    'over_under_25': 'Over 2.5 goals',
                    'btts': 'Both teams to score',
                    'clean_sheet_home': f'{home_team_actual} keeps clean sheet',
                    'clean_sheet_away': f'{away_team_actual} keeps clean sheet'
                }
                result["predictions"][market] = {
                    "prediction": pred_data['prediction'],
                    "prediction_description": market_descriptions.get(market, market),
                    "probability": pred_data['probability'],
                    "interpretation": "Yes" if pred_data['prediction'] else "No"
                }
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to predict match: {str(e)}"})


@tool
def get_overall_statistics() -> str:
    """Get overall statistics about match outcomes across all matches in the database.
    
    Returns:
        JSON string with overall statistics including home win rate, draw rate, away win rate
    """
    kg_queries = load_knowledge_graph()
    if kg_queries is None:
        return json.dumps({"error": "Knowledge graph not loaded. Please run knowledge_graph.py first."})
    
    try:
        stats = kg_queries.get_overall_match_statistics()
        return json.dumps(stats, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to get overall statistics: {str(e)}"})


@tool
def analyze_betting_value(home_team: str, away_team: str, home_odds: float, draw_odds: float, away_odds: float) -> str:
    """Analyze betting value by comparing ML predictions with bookmaker odds.
    
    Args:
        home_team: Name of the home team
        away_team: Name of the away team
        home_odds: Decimal odds for home win
        draw_odds: Decimal odds for draw
        away_odds: Decimal odds for away win
    
    Returns:
        JSON string with value analysis and betting recommendations
    """
    predictor = load_predictor()
    if predictor is None:
        return json.dumps({"error": "Models not loaded. Please train models first."})
    
    try:
        featured_data = get_recent_featured_data_for_prediction(limit=2000)
        
        home_team_actual = get_team_from_data(home_team, featured_data)
        away_team_actual = get_team_from_data(away_team, featured_data)
        
        if not home_team_actual or not away_team_actual:
            return json.dumps({
                "error": f"Teams not found. Home: {home_team_actual}, Away: {away_team_actual}"
            })
        
        exact_match = featured_data[
            (featured_data['HomeTeam'] == home_team_actual) & 
            (featured_data['AwayTeam'] == away_team_actual)
        ].sort_values('MatchDate', ascending=False)
        
        if len(exact_match) > 0:
            match_data = exact_match.iloc[0:1].copy()
        else:
            reversed_match = featured_data[
                (featured_data['HomeTeam'] == away_team_actual) & 
                (featured_data['AwayTeam'] == home_team_actual)
            ].sort_values('MatchDate', ascending=False)
            
            if len(reversed_match) == 0:
                return json.dumps({
                    "error": f"No recent matches found between {home_team_actual} and {away_team_actual}"
                })
            
            match_row = reversed_match.iloc[0:1].copy()
            match_data = match_row.copy()
            
            swap_pairs = [
                ('HomeTeam', 'AwayTeam'),
                ('HomeElo', 'AwayElo'),
                ('Form3Home', 'Form3Away'),
                ('Form5Home', 'Form5Away'),
                ('OddHome', 'OddAway'),
            ]
            
            for home_col, away_col in swap_pairs:
                if home_col in match_data.columns and away_col in match_data.columns:
                    match_data[home_col], match_data[away_col] = match_data[away_col].values, match_data[home_col].values
            
            match_data['HomeTeam'] = home_team_actual
            match_data['AwayTeam'] = away_team_actual
        
        predictions = predictor.predict_match(match_data, home_team_actual, away_team_actual)
        
        if 'match_result' not in predictions:
            return json.dumps({"error": "Match result predictions not available"})
        
        model_probs = predictions['match_result']['probabilities']
        
        implied_prob_home = 1.0 / home_odds
        implied_prob_draw = 1.0 / draw_odds
        implied_prob_away = 1.0 / away_odds
        
        overround = implied_prob_home + implied_prob_draw + implied_prob_away
        
        true_prob_home = implied_prob_home / overround if overround > 0 else implied_prob_home
        true_prob_draw = implied_prob_draw / overround if overround > 0 else implied_prob_draw
        true_prob_away = implied_prob_away / overround if overround > 0 else implied_prob_away
        
        model_home = model_probs.get('H', 0.0)
        model_draw = model_probs.get('D', 0.0)
        model_away = model_probs.get('A', 0.0)
        
        value_home = model_home - true_prob_home
        value_draw = model_draw - true_prob_draw
        value_away = model_away - true_prob_away
        
        expected_value_home = (model_home * home_odds) - 1.0
        expected_value_draw = (model_draw * draw_odds) - 1.0
        expected_value_away = (model_away * away_odds) - 1.0
        
        options = [
            {
                'outcome': 'Home Win',
                'team': home_team_actual,
                'odds': home_odds,
                'model_probability': round(model_home * 100, 2),
                'implied_probability': round(true_prob_home * 100, 2),
                'value': round(value_home * 100, 2),
                'expected_value': round(expected_value_home * 100, 2)
            },
            {
                'outcome': 'Draw',
                'team': 'Draw',
                'odds': draw_odds,
                'model_probability': round(model_draw * 100, 2),
                'implied_probability': round(true_prob_draw * 100, 2),
                'value': round(value_draw * 100, 2),
                'expected_value': round(expected_value_draw * 100, 2)
            },
            {
                'outcome': 'Away Win',
                'team': away_team_actual,
                'odds': away_odds,
                'model_probability': round(model_away * 100, 2),
                'implied_probability': round(true_prob_away * 100, 2),
                'value': round(value_away * 100, 2),
                'expected_value': round(expected_value_away * 100, 2)
            }
        ]
        
        options.sort(key=lambda x: x['expected_value'], reverse=True)
        
        result = {
            'match': f"{home_team_actual} vs {away_team_actual}",
            'bookmaker_margin': round((overround - 1.0) * 100, 2),
            'options': options,
            'recommendation': {
                'best_bet': options[0]['outcome'],
                'reason': f"Highest expected value: {options[0]['expected_value']}%"
            }
        }
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to analyze betting value: {str(e)}"})


@tool
def find_value_bets(min_probability_diff: float = 0.1) -> str:
    """Find value bets where model probability exceeds bookmaker implied probability.
    
    Args:
        min_probability_diff: Minimum difference between predicted and implied probability (default: 0.1)
    
    Returns:
        JSON string with list of value bets
    """
    kg_queries = load_knowledge_graph()
    if kg_queries is None:
        return json.dumps({"error": "Knowledge graph not loaded. Please run knowledge_graph.py first."})
    
    try:
        value_bets = kg_queries.find_value_bets(min_probability_diff=min_probability_diff)
        return json.dumps(value_bets, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to find value bets: {str(e)}"})


SYSTEM_PROMPT = """You are a football statistics assistant. Your role is to help users with:
1. General statistical queries (overall home win rates, match outcome distributions) - use get_overall_statistics
2. Statistical queries about specific teams (goals, matches, history) - use get_team_statistics or get_team_matches
3. Match predictions for specific matches (who will win, over/under, BTTS, etc.) - use predict_match_result

IMPORTANT RULES:
- You MUST use the provided tools to answer questions. Do NOT make up answers based on your training data.
- For GENERAL statistical questions (e.g., "what's the probability of home team winning", "overall home win rate") use get_overall_statistics
- For statistical questions about SPECIFIC teams, use get_team_statistics or get_team_matches
- For prediction questions about SPECIFIC matches, use predict_match_result with the team names
- Always present results in a clear, user-friendly format
- If a tool returns an error, explain it to the user and suggest what they need to do
- Be conversational and helpful, but always base your answers on tool results

BETTING ANALYSIS REDIRECT:
- If the user asks about odds, betting value, expected value, whether to bet, or provides bookmaker odds, 
  politely redirect them to the "💰 Odds Analyzer" tab which is designed specifically for that purpose.
- Say something like: "For betting analysis and odds evaluation, please use the **Odds Analyzer** tab. 
  There you can select teams, enter odds, and get a detailed value analysis with recommendations."
- Do NOT analyze odds or give betting recommendations in this chat.

PREDICTION INTERPRETATION GUIDE:
- Match result predictions use: H = Home team wins, D = Draw, A = Away team wins
- When interpreting predictions, clearly state which team is home and which is away
- For match_result probabilities, all three outcomes (H, D, A) should sum to approximately 100%
- For binary predictions (over_under_25, btts, clean_sheet), the probability is for the positive outcome
- Clean sheet predictions: clean_sheet_home = home team doesn't concede, clean_sheet_away = away team doesn't concede
- Present ALL probabilities from match_result predictions, not just the predicted outcome
- Include model performance context: Match Result ~61% accuracy, Over/Under ~75%, BTTS ~74%, Clean Sheets ~78-82%

DISCLAIMER:
- These are ML model predictions based on historical data patterns, not guarantees
- Model accuracy varies by market (see performance metrics above)"""


def create_agent(llm):
    tools = [
        get_team_statistics,
        get_team_matches,
        predict_match_result,
        get_overall_statistics
    ]
    
    if not USE_LANGGRAPH and AgentExecutor is not None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor
    else:
        return create_agent_new(llm, tools)


def get_all_teams():
    """Get list of all teams for dropdowns."""
    try:
        preprocessor = DataPreprocessor('data/raw/matches.csv')
        clean_data = preprocessor.preprocess()
        all_teams = sorted(set(clean_data['HomeTeam'].unique()) | set(clean_data['AwayTeam'].unique()))
        return all_teams, clean_data
    except:
        return [], None


def get_match_features(clean_data, home_team, away_team):
    """Get feature data for a match."""
    recent_data = clean_data.tail(2000).copy()
    engineer = FeatureEngineer(recent_data)
    featured_data = engineer.create_all_features(include_goal_features=True)
    
    exact_match = featured_data[
        (featured_data['HomeTeam'] == home_team) & 
        (featured_data['AwayTeam'] == away_team)
    ].sort_values('MatchDate', ascending=False)
    
    if len(exact_match) > 0:
        return exact_match.iloc[0:1].copy()
    
    reversed_match = featured_data[
        (featured_data['HomeTeam'] == away_team) & 
        (featured_data['AwayTeam'] == home_team)
    ].sort_values('MatchDate', ascending=False)
    
    if len(reversed_match) > 0:
        match_data = reversed_match.iloc[0:1].copy()
        swap_pairs = [('HomeTeam', 'AwayTeam'), ('HomeElo', 'AwayElo'), ('Form3Home', 'Form3Away'), ('Form5Home', 'Form5Away')]
        for home_col, away_col in swap_pairs:
            if home_col in match_data.columns and away_col in match_data.columns:
                match_data[home_col], match_data[away_col] = match_data[away_col].values, match_data[home_col].values
        match_data['HomeTeam'] = home_team
        match_data['AwayTeam'] = away_team
        return match_data
    
    return featured_data.iloc[-1:].copy()


def calculate_value(model_prob, implied_prob, odds):
    """Calculate expected value."""
    return (model_prob * odds) - 1.0


def get_recommendation(ev, model_prob, implied_prob):
    """Get recommendation based on expected value with explanation."""
    diff = (model_prob - implied_prob) * 100
    
    if ev > 0.10:
        emoji = "✨"
        label = "STRONG BET"
        explanation = f"Our model gives this outcome {diff:.1f}% higher probability than the bookmaker. This represents significant value."
    elif ev > 0.05:
        emoji = "🟢"
        label = "BET"
        explanation = f"Our model estimates {diff:.1f}% higher probability than implied by odds. Good value opportunity."
    elif ev > 0:
        emoji = "🟡"
        label = "SLIGHT VALUE"
        explanation = f"Marginally positive value ({diff:.1f}% edge). Consider only with high confidence."
    elif ev > -0.05:
        emoji = "🟠"
        label = "AVOID"
        explanation = f"Fair odds. Model probability is close to bookmaker's implied probability."
    else:
        emoji = "🔴"
        label = "DON'T BET"
        explanation = f"Bookmaker overestimates this outcome by {-diff:.1f}%. Negative expected value."
    
    return emoji, label, explanation


def render_odds_analyzer():
    """Render the odds analysis tab."""
    st.header("💰 Odds Value Analyzer")
    
    predictor = load_predictor()
    if predictor is None:
        st.error("❌ ML Models not found. Run `python model_training.py` first.")
        return
    
    all_teams, clean_data = get_all_teams()
    if not all_teams:
        st.error("❌ Failed to load team data.")
        return
    
    tab1, tab2 = st.tabs(["🏆 Match Result (1X2)", "⚽ Goals Markets"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("🏠 Home Team", all_teams, key="home_1x2")
        with col2:
            away_options = [t for t in all_teams if t != home_team]
            away_team = st.selectbox("✈️ Away Team", away_options, key="away_1x2")
        
        st.subheader("Enter Bookmaker Odds")
        col1, col2, col3 = st.columns(3)
        with col1:
            home_odds = st.number_input("Home Win", min_value=1.01, max_value=50.0, value=2.00, step=0.05)
        with col2:
            draw_odds = st.number_input("Draw", min_value=1.01, max_value=50.0, value=3.50, step=0.05)
        with col3:
            away_odds = st.number_input("Away Win", min_value=1.01, max_value=50.0, value=3.00, step=0.05)
        
        if st.button("🔍 Analyze Match", type="primary"):
            with st.spinner("Running ML prediction..."):
                match_data = get_match_features(clean_data, home_team, away_team)
                predictions = predictor.predict_match(match_data, home_team, away_team, include_uncertainty=False)
            
            if 'match_result' in predictions:
                probs = predictions['match_result']['probabilities']
                
                overround = (1/home_odds) + (1/draw_odds) + (1/away_odds)
                implied_h = (1/home_odds) / overround
                implied_d = (1/draw_odds) / overround
                implied_a = (1/away_odds) / overround
                
                model_h, model_d, model_a = probs.get('H', 0), probs.get('D', 0), probs.get('A', 0)
                ev_h = calculate_value(model_h, implied_h, home_odds)
                ev_d = calculate_value(model_d, implied_d, draw_odds)
                ev_a = calculate_value(model_a, implied_a, away_odds)
                
                st.subheader("📊 Results")
                
                with st.expander("ℹ️ What does EV mean?", expanded=False):
                    st.markdown("""
                    **EV (Expected Value)** is the theoretical profit/loss per €1 bet over many bets.
                    
                    - **EV +10%** → Expect to profit €0.10 per €1 wagered long-term
                    - **EV 0%** → Break-even odds (fair bet)
                    - **EV -10%** → Expect to lose €0.10 per €1 wagered
                    
                    We compare our ML model's probability vs the bookmaker's implied probability.
                    When our model is more confident than the bookmaker, that's **value**.
                    """)
                
                st.caption(f"Bookmaker Margin: {(overround-1)*100:.1f}%")
                
                results = []
                
                col1, col2, col3 = st.columns(3)
                for col, label, team, prob, ev, odds, implied in [
                    (col1, "🏠 Home Win", home_team, model_h, ev_h, home_odds, implied_h),
                    (col2, "🤝 Draw", "Draw", model_d, ev_d, draw_odds, implied_d),
                    (col3, "✈️ Away Win", away_team, model_a, ev_a, away_odds, implied_a)
                ]:
                    with col:
                        emoji, rec_label, explanation = get_recommendation(ev, prob, implied)
                        ev_display = round(ev * 100, 1)
                        st.metric(label, f"{prob*100:.1f}%", delta=f"{ev_display:+.1f}% EV", delta_color="normal")
                        st.caption(f"Odds: {odds:.2f} | Implied: {implied*100:.1f}%")
                        st.markdown(f"**{emoji} {rec_label}**")
                        results.append((team, prob, implied, ev, emoji, rec_label, explanation))
                
                best_ev = max(ev_h, ev_d, ev_a)
                if best_ev > 0:
                    if best_ev == ev_h: 
                        best = f"{home_team} @ {home_odds:.2f}"
                        best_team = home_team
                    elif best_ev == ev_d: 
                        best = f"Draw @ {draw_odds:.2f}"
                        best_team = "Draw"
                    else: 
                        best = f"{away_team} @ {away_odds:.2f}"
                        best_team = away_team
                    st.success(f"💡 **Best Value:** {best} (EV: {best_ev*100:+.1f}%)")
                else:
                    st.warning("⚠️ No value bets found in this match.")
                
                st.divider()
                st.subheader("📝 Detailed Analysis")
                
                for team, prob, implied, ev, emoji, rec_label, explanation in results:
                    with st.container():
                        st.markdown(f"**{team}:** {explanation}")
                
                st.divider()
                if best_ev > 0.05:
                    st.markdown(f"""
                    ### 🎯 Recommendation
                    
                    Based on our analysis of **{home_team} vs {away_team}**, we recommend betting on **{best_team}**.
                    
                    Our ML model (trained on 80,000+ historical matches) estimates this outcome has a 
                    **{best_ev*100:.1f}% edge** over the bookmaker's odds. This suggests the bookmaker 
                    may be undervaluing this outcome.
                    
                    *Note: This is a statistical analysis, not financial advice. Past performance doesn't guarantee future results.*
                    """)
                elif best_ev > 0:
                    st.markdown(f"""
                    ### 🎯 Recommendation
                    
                    There's marginal value on **{best_team}** (+{best_ev*100:.1f}% EV), but the edge is small.
                    Consider this bet only if you have additional reasons to be confident.
                    
                    *Note: Small edges can be eaten up by variance. Proceed with caution.*
                    """)
                else:
                    st.markdown("""
                    ### 🎯 Recommendation
                    
                    **Skip this match.** None of the available odds offer positive expected value based on our model.
                    The bookmaker's prices are fair or unfavorable for all outcomes.
                    
                    *Better opportunities may exist in other matches.*
                    """)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            home_team_g = st.selectbox("🏠 Home Team", all_teams, key="home_goals")
        with col2:
            away_options_g = [t for t in all_teams if t != home_team_g]
            away_team_g = st.selectbox("✈️ Away Team", away_options_g, key="away_goals")
        
        market = st.radio("Select Market", ["Over/Under 2.5", "BTTS", "Clean Sheet Home", "Clean Sheet Away"], horizontal=True)
        
        market_map = {"Over/Under 2.5": "over_under_25", "BTTS": "btts", "Clean Sheet Home": "clean_sheet_home", "Clean Sheet Away": "clean_sheet_away"}
        
        col1, col2 = st.columns(2)
        with col1:
            yes_odds = st.number_input("Yes Odds", min_value=1.01, max_value=20.0, value=1.90, step=0.05)
        with col2:
            no_odds = st.number_input("No Odds", min_value=1.01, max_value=20.0, value=1.90, step=0.05)
        
        if st.button("🔍 Analyze Market", type="primary", key="analyze_goals"):
            with st.spinner("Running ML prediction..."):
                match_data = get_match_features(clean_data, home_team_g, away_team_g)
                predictions = predictor.predict_match(match_data, home_team_g, away_team_g, include_uncertainty=False)
            
            market_key = market_map[market]
            if market_key in predictions:
                prob_yes = predictions[market_key]['probability']
                prob_no = 1 - prob_yes
                
                overround = (1/yes_odds) + (1/no_odds)
                implied_yes = (1/yes_odds) / overround
                implied_no = (1/no_odds) / overround
                
                ev_yes = calculate_value(prob_yes, implied_yes, yes_odds)
                ev_no = calculate_value(prob_no, implied_no, no_odds)
                
                st.subheader("📊 Results")
                
                market_labels = {
                    "over_under_25": ("Over 2.5 Goals", "Under 2.5 Goals"),
                    "btts": ("Both Teams Score", "Clean Sheet"),
                    "clean_sheet_home": (f"{home_team_g} Clean Sheet", f"{home_team_g} Concedes"),
                    "clean_sheet_away": (f"{away_team_g} Clean Sheet", f"{away_team_g} Concedes")
                }
                yes_label, no_label = market_labels.get(market_key, ("Yes", "No"))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    emoji_yes, rec_yes, expl_yes = get_recommendation(ev_yes, prob_yes, implied_yes)
                    ev_yes_display = round(ev_yes * 100, 1)
                    st.metric(f"✅ {yes_label}", f"{prob_yes*100:.1f}%", delta=f"{ev_yes_display:+.1f}% EV", delta_color="normal")
                    st.caption(f"Odds: {yes_odds:.2f} | Implied: {implied_yes*100:.1f}%")
                    st.markdown(f"**{emoji_yes} {rec_yes}**")
                
                with col2:
                    emoji_no, rec_no, expl_no = get_recommendation(ev_no, prob_no, implied_no)
                    ev_no_display = round(ev_no * 100, 1)
                    st.metric(f"❌ {no_label}", f"{prob_no*100:.1f}%", delta=f"{ev_no_display:+.1f}% EV", delta_color="normal")
                    st.caption(f"Odds: {no_odds:.2f} | Implied: {implied_no*100:.1f}%")
                    st.markdown(f"**{emoji_no} {rec_no}**")
                
                best_ev = max(ev_yes, ev_no)
                best_choice = yes_label if best_ev == ev_yes else no_label
                best_odds = yes_odds if best_ev == ev_yes else no_odds
                
                if best_ev > 0:
                    st.success(f"💡 **Best Value:** {best_choice} @ {best_odds:.2f} (EV: {best_ev*100:+.1f}%)")
                else:
                    st.warning("⚠️ No value bets found.")
                
                # Detailed analysis
                st.divider()
                st.subheader("📝 Analysis")
                
                st.markdown(f"**{yes_label}:** {expl_yes}")
                st.markdown(f"**{no_label}:** {expl_no}")
                
                st.divider()
                if best_ev > 0.05:
                    st.markdown(f"""
                    ### 🎯 Recommendation
                    
                    Bet on **{best_choice}** at odds of {best_odds:.2f}.
                    Our model estimates a **{best_ev*100:.1f}% edge** over these odds.
                    
                    *Note: Statistical analysis only. Not financial advice.*
                    """)
                elif best_ev > 0:
                    st.markdown(f"""
                    ### 🎯 Recommendation
                    
                    Small value on **{best_choice}** (+{best_ev*100:.1f}% EV). 
                    The edge is marginal - consider carefully.
                    """)
                else:
                    st.markdown("""
                    ### 🎯 Recommendation
                    
                    **Skip this market.** No positive expected value available.
                    """)


def render_chatbot():
    """Render the chatbot tab."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "nlp_processor" not in st.session_state:
        st.session_state.nlp_processor = load_nlp_preprocessor()
    
    if "agent" not in st.session_state:
        with st.spinner("Initializing AI agent..."):
            try:
                llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434", temperature=0)
                st.session_state.agent = create_agent(llm)
                st.session_state.llm_ready = True
            except Exception as e:
                st.error(f"Failed to connect to Ollama: {e}")
                st.info("Start Ollama: `ollama serve` then `ollama pull llama3.2`")
                st.session_state.llm_ready = False
                return
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask me about teams, matches..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    nlp_result = preprocess_query_with_spacy(prompt, st.session_state.nlp_processor)
                    enhanced_input = prompt
                    if nlp_result['teams']:
                        enhanced_input += f"\n\n[Teams: {', '.join(nlp_result['teams'])}]"
                    
                    chat_history = [
                        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                        for m in st.session_state.messages[:-1]
                    ]
                    
                    if not USE_LANGGRAPH and AgentExecutor is not None:
                        response = st.session_state.agent.invoke({"input": enhanced_input, "chat_history": chat_history})
                        answer = response.get("output", "I couldn't process that.")
                    else:
                        messages = chat_history + [HumanMessage(content=enhanced_input)]
                        response = st.session_state.agent.invoke({"messages": messages})
                        answer = response["messages"][-1].content if "messages" in response else str(response)
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")


def main():
    st.set_page_config(page_title="Football Betting Assistant", page_icon="⚽", layout="wide")
    
    st.title("⚽ Football Betting Assistant")
    
    tab1, tab2 = st.tabs(["💬 AI Chatbot", "💰 Odds Analyzer"])
    
    with tab1:
        render_chatbot()
    
    with tab2:
        render_odds_analyzer()
    
    with st.sidebar:
        st.header("📊 System Status")
        
        if load_predictor():
            st.success("✅ ML Models Loaded")
        else:
            st.error("❌ ML Models Not Found")
        
        if load_knowledge_graph():
            st.success("✅ Knowledge Graph Loaded")
        else:
            st.warning("⚠️ Knowledge Graph Not Found")
        
        st.header("💡 Quick Guide")
        st.markdown("""
        **Chatbot Tab:** Ask questions in natural language
        
        **Odds Analyzer Tab:** 
        1. Select teams
        2. Enter bookmaker odds
        3. Get value analysis
        """)


if __name__ == "__main__":
    main()

