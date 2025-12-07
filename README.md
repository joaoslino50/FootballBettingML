# Football Betting ML System

A machine learning system for football betting predictions with uncertainty quantification, knowledge graph integration, and an interactive chatbot interface.

## Features

- **ML Models**: Predicts match results, over/under 2.5 goals, BTTS, and clean sheets
- **Uncertainty Quantification**: Bootstrap-based confidence intervals for predictions
- **Knowledge Graph**: RDF-based knowledge graph with SPARQL queries
- **Interactive Chatbot**: AI-powered assistant for team statistics and predictions
- **Odds Analyzer**: Value betting analysis with expected value calculations
- **NLP Processing**: spaCy-based entity extraction and intent classification

## Prerequisites

- Python 3.10+ (tested with Python 3.12)
- Conda (recommended) or pip
- Ollama (for chatbot functionality)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd kl-project
```

### 2. Create Conda Environment

```bash
conda create -n kl-project python=3.12
conda activate kl-project
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Install and Setup Ollama (for Chatbot)

1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Pull the recommended model:
   ```bash
   ollama pull llama3.2
   ```
3. Start Ollama server:
   ```bash
   ollama serve
   ```

## Data Setup

Ensure you have the raw match data at:
```
data/raw/matches.csv
```

The CSV should contain columns: `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `Date`, `Div`, etc.

## Usage

### 1. Train ML Models

Train all models with full dataset:
```bash
python model_training.py --full
```

Train with SPARQL-based feature extraction (for proposal compliance):
```bash
python model_training.py --sparql --sample 5000
```

**Note**: Training takes approximately 10-30 minutes on an M1 MacBook Pro depending on dataset size.

### 2. Build Knowledge Graph

Generate the knowledge graph from match data:
```bash
python knowledge_graph.py
```

This creates `data/knowledge_graph/football_kg.ttl` with all match data, features, and predictions.

### 3. Run the Application

Start the Streamlit interface:
```bash
streamlit run chatbot.py
```

The app will open at `http://localhost:8501` with two tabs:
- **💬 AI Chatbot**: Ask questions about teams, statistics, and predictions
- **💰 Odds Analyzer**: Analyze betting odds and find value bets

### 4. Make Predictions (Command Line)

Predict a single match:
```bash
python predict.py --home "Liverpool" --away "Arsenal"
```

## Project Structure

```
kl-project/
├── data/
│   ├── raw/
│   │   └── matches.csv          # Raw match data
│   ├── knowledge_graph/
│   │   └── football_kg.ttl      # RDF knowledge graph
│   └── ontology/
│       └── football_betting.owl  # OWL ontology definition
├── models/                       # Trained ML models
├── app.py                        # Legacy Streamlit app
├── chatbot.py                    # Main chatbot interface
├── model_training.py             # ML model training
├── predict.py                    # Prediction script
├── knowledge_graph.py            # Knowledge graph builder
├── sparql_queries.py             # SPARQL query interface
├── sparql_feature_extraction.py # SPARQL-based features
├── data_preprocessing.py         # Data cleaning
├── feature_engineering.py       # Feature creation
├── nlp_preprocessing.py          # NLP utilities
└── requirements.txt              # Python dependencies
```

## Model Performance

- **Match Result**: ~61% accuracy
- **Over/Under 2.5 Goals**: ~75% accuracy
- **Both Teams To Score (BTTS)**: ~74% accuracy
- **Clean Sheets**: ~78-82% accuracy

## Key Technologies

- **ML**: scikit-learn (Random Forest, Gradient Boosting, Calibrated Classifiers)
- **Uncertainty**: Bootstrap methods (UQ360 alternative)
- **Knowledge Graph**: RDFLib, SPARQL, Schema.org, OWL
- **NLP**: spaCy
- **Chatbot**: LangChain, Ollama
- **UI**: Streamlit

## Troubleshooting

### Ollama Connection Error

If the chatbot shows "Failed to connect to Ollama":
1. Ensure Ollama is running: `ollama serve`
2. Verify the model is installed: `ollama list`
3. Pull the model if missing: `ollama pull llama3.2`

### Missing Models Error

If you see "ML Models Not Found":
1. Train the models: `python model_training.py --full`
2. Ensure models are in the `models/` directory

### Knowledge Graph Not Found

If you see "Knowledge Graph Not Found":
1. Build the knowledge graph: `python knowledge_graph.py`
2. Ensure `data/knowledge_graph/football_kg.ttl` exists

### spaCy Model Missing

If you see "spaCy Not Available":
```bash
python -m spacy download en_core_web_sm
```

## Notes

- The system uses bootstrap methods for uncertainty quantification (UQ360 had compatibility issues with Python 3.12)
- SPARQL feature extraction is available but slower; standard feature extraction is recommended for practical use
- All predictions include uncertainty intervals when available
- The knowledge graph integrates predictions with uncertainty bounds

## License

[Add your license here]

## Contact

[Add contact information here]

