# main.py
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import ulid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langfuse import Langfuse

from utils.loaders import load_data
from utils.output import write_output
from preprocessing.audio import AudioTranscriber
from scoring.profiler import build_user_profiles
from scoring.aggregator import ScoreAggregator
from ml.isolation_forest import IsolationForestScorer
from ml.graph_analyzer import GraphAnalyzer
from agents.comms import CommsAgent
from agents.coordinator import FraudCoordinator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def generate_session_id() -> str:
    """Generate unique session ID for Langfuse tracking."""
    team = os.getenv("TEAM_NAME", "default").replace(" ", "-")
    return f"{team}-{ulid.new().str}"


def create_model():
    """Create LangChain chat model via OpenRouter."""
    return ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3-plus",
        temperature=0.3,
        max_tokens=4000,
    )


def run_pipeline(dataset_name: str) -> str:
    """
    Run the full fraud detection pipeline for a dataset.

    Args:
        dataset_name: Name of the dataset to process

    Returns:
        Path to the output file
    """
    session_id = generate_session_id()
    logger.info(f"Starting pipeline for '{dataset_name}' | Session: {session_id}")

    # Initialize Langfuse
    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
    )

    # Load data
    logger.info("Loading data...")
    data = load_data(dataset_name)

    # Audio transcription (if available)
    audio_transcripts = {}
    if data["has_audio"]:
        logger.info("Transcribing audio files...")
        try:
            transcriber = AudioTranscriber()
            audio_transcripts = transcriber.transcribe_directory(data["audio_path"])
        except Exception as e:
            logger.warning(f"Audio transcription failed: {e}")

    # Build user profiles
    logger.info("Building user profiles...")
    profiles = build_user_profiles(data["train_transactions"], data["users"])

    # Create user name -> IBAN mapping for comms agent
    user_mapping = {
        profile["name"]: iban
        for iban, profile in profiles.items()
        if profile.get("name")
    }

    # Run ML layer
    logger.info("Running ML anomaly detection...")

    # Isolation Forest
    aggregator = ScoreAggregator(profiles)
    train_scores = aggregator.score_all(
        data["train_transactions"],
        data["train_locations"]
    )

    # Prepare feature matrix for Isolation Forest
    feature_cols = [c for c in train_scores.columns if c.endswith("_score")]
    feature_cols = [c for c in feature_cols if c not in ["isolation_forest_score", "graph_score"]]

    if feature_cols:
        train_features = train_scores[feature_cols].fillna(0).values

        iforest = IsolationForestScorer()
        iforest.fit(train_features)

        # Score eval transactions (without ML scores initially)
        eval_scores_initial = aggregator.score_all(
            data["eval_transactions"],
            data["eval_locations"]
        )
        eval_features = eval_scores_initial[feature_cols].fillna(0).values
        iforest_scores = iforest.score(eval_features)

        ml_scores = dict(zip(
            eval_scores_initial["transaction_id"],
            iforest_scores
        ))
    else:
        ml_scores = {}

    # Graph Analysis
    logger.info("Running graph analysis...")
    all_transactions = pd.concat([
        data["train_transactions"],
        data["eval_transactions"]
    ], ignore_index=True)

    graph_analyzer = GraphAnalyzer()
    graph_analyzer.build_graph(all_transactions)
    graph_scores = graph_analyzer.score_transactions(data["eval_transactions"])

    # Final score aggregation with ML scores
    logger.info("Aggregating scores...")
    risk_table = aggregator.score_all(
        data["eval_transactions"],
        data["eval_locations"],
        ml_scores=ml_scores,
        graph_scores=graph_scores
    )

    # Run LLM agents
    logger.info("Running LLM agents...")
    model = create_model()

    # Communications agent
    comms_agent = CommsAgent(model, session_id)
    comms_signals = comms_agent.analyze(
        sms=data["train_sms"] + data["eval_sms"],
        mails=data["train_mails"] + data["eval_mails"],
        audio_transcripts=audio_transcripts,
        user_mapping=user_mapping
    )
    logger.info(f"Comms agent found {len(comms_signals.get('signals', []))} signals")

    # Fraud coordinator
    coordinator = FraudCoordinator(model, session_id)
    flagged_ids = coordinator.decide(
        risk_table=risk_table,
        comms_signals=comms_signals,
        eval_transactions=data["eval_transactions"]
    )
    logger.info(f"Coordinator flagged {len(flagged_ids)} transactions")

    # Write output
    output_path = write_output(flagged_ids, dataset_name)
    logger.info(f"Output written to: {output_path}")

    # Flush Langfuse traces
    langfuse_client.flush()
    logger.info(f"Pipeline complete | Session: {session_id}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Agent")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'The Truman Show', 'Brave New World', 'Deus Ex')"
    )

    args = parser.parse_args()

    try:
        output_path = run_pipeline(args.dataset)
        print(f"\nSuccess! Output written to: {output_path}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
