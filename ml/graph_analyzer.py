# ml/graph_analyzer.py
import networkx as nx
import pandas as pd
from datetime import timedelta
from typing import Optional

class GraphAnalyzer:
    """Analyze transaction graph for fraud patterns."""

    def __init__(self):
        self.graph: Optional[nx.DiGraph] = None

    def build_graph(self, transactions: pd.DataFrame) -> None:
        """Build directed graph from transactions."""
        self.graph = nx.DiGraph()

        for _, tx in transactions.iterrows():
            sender = tx["sender_id"]
            recipient = tx["recipient_id"]

            if pd.isna(recipient) or not recipient:
                continue

            self.graph.add_edge(
                sender,
                recipient,
                tx_id=tx["transaction_id"],
                amount=tx["amount"],
                timestamp=tx["timestamp"]
            )

    def detect_circular_flows(self) -> list[list[str]]:
        """Find cycles in the transaction graph (potential money laundering)."""
        if not self.graph:
            return []

        cycles = list(nx.simple_cycles(self.graph))
        # Only return cycles of length >= 3
        return [c for c in cycles if len(c) >= 3]

    def detect_mule_accounts(
        self,
        in_threshold: int = 5,
        out_threshold: int = 3
    ) -> list[str]:
        """Find accounts with high in-degree and out-degree (potential mules)."""
        if not self.graph:
            return []

        suspicious = []
        for node in self.graph.nodes():
            in_deg = self.graph.in_degree(node)
            out_deg = self.graph.out_degree(node)
            if in_deg >= in_threshold and out_deg >= out_threshold:
                suspicious.append(node)

        return suspicious

    def detect_rapid_layering(self, max_hours: float = 2.0) -> list[str]:
        """Find transactions involved in rapid layering pattern."""
        if not self.graph:
            return []

        suspicious_txs = []

        for node in self.graph.nodes():
            in_edges = list(self.graph.in_edges(node, data=True))
            out_edges = list(self.graph.out_edges(node, data=True))

            for in_edge in in_edges:
                in_data = in_edge[2]
                in_ts = in_data.get("timestamp")
                in_amount = in_data.get("amount", 0)

                if in_ts is None:
                    continue

                for out_edge in out_edges:
                    out_data = out_edge[2]
                    out_ts = out_data.get("timestamp")
                    out_amount = out_data.get("amount", 0)

                    if out_ts is None:
                        continue

                    # Check timing
                    time_diff = (out_ts - in_ts).total_seconds() / 3600
                    if 0 < time_diff <= max_hours:
                        # Check similar amount (±20%)
                        if in_amount > 0 and 0.8 <= out_amount / in_amount <= 1.2:
                            suspicious_txs.append(in_data.get("tx_id"))
                            suspicious_txs.append(out_data.get("tx_id"))

        return list(set(suspicious_txs))

    def score_transactions(self, transactions: pd.DataFrame) -> dict[str, float]:
        """Score each transaction based on graph pattern involvement."""
        if not self.graph:
            return {tx_id: 0.0 for tx_id in transactions["transaction_id"]}

        # Find all suspicious patterns
        cycles = self.detect_circular_flows()
        mules = set(self.detect_mule_accounts())
        layering_txs = set(self.detect_rapid_layering())

        # Get all nodes in cycles
        cycle_nodes = set()
        for cycle in cycles:
            cycle_nodes.update(cycle)

        scores = {}
        for _, tx in transactions.iterrows():
            tx_id = tx["transaction_id"]
            score = 0.0

            sender = tx["sender_id"]
            recipient = tx["recipient_id"]

            # Check if tx is in layering pattern
            if tx_id in layering_txs:
                score = max(score, 0.8)

            # Check if sender/recipient is in a cycle
            if sender in cycle_nodes or recipient in cycle_nodes:
                score = max(score, 0.6)

            # Check if recipient is a mule
            if recipient in mules:
                score = max(score, 0.7)

            # Check if sender is a mule
            if sender in mules:
                score = max(score, 0.5)

            scores[tx_id] = score

        return scores
