"""
Backwards-compatibility alias module forwarding to app.py
"""
from app import app, generate_option_chain_snapshot, compute_technicals, fetch_and_score_sentiment, run_prediction_engine

__all__ = ["app", "generate_option_chain_snapshot", "compute_technicals", "fetch_and_score_sentiment", "run_prediction_engine"]
