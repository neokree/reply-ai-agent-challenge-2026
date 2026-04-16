# tests/test_integration.py
"""
Integration tests for the full fraud detection pipeline.
Run with: pytest tests/test_integration.py -v -s
Requires: .env file with API keys
"""
import pytest
import os
from pathlib import Path

# Skip if no API keys
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="Requires OPENROUTER_API_KEY"
)

class TestIntegration:

    def test_full_pipeline_truman_show(self):
        """Test full pipeline on The Truman Show dataset."""
        from main import run_pipeline

        output_path = run_pipeline("The Truman Show")

        assert os.path.exists(output_path)

        with open(output_path) as f:
            lines = f.read().strip().split('\n')

        # Should have at least 1 flagged transaction
        assert len(lines) >= 1

        # Should not flag all transactions (80 in dataset)
        assert len(lines) < 80

        # Each line should be a valid UUID-like transaction ID
        for line in lines:
            assert len(line) > 10
            assert "-" in line

    def test_output_format_valid(self):
        """Verify output meets challenge requirements."""
        from main import run_pipeline
        from utils.loaders import load_data

        output_path = run_pipeline("The Truman Show")
        data = load_data("The Truman Show")

        valid_ids = set(data["eval_transactions"]["transaction_id"])

        with open(output_path) as f:
            flagged = [line.strip() for line in f if line.strip()]

        # All flagged IDs must be valid
        for tx_id in flagged:
            assert tx_id in valid_ids, f"Invalid transaction ID: {tx_id}"

        # Not empty
        assert len(flagged) > 0, "Output is empty"

        # Not all
        assert len(flagged) < len(valid_ids), "All transactions flagged"
