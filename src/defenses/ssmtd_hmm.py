import copy as cp
import numpy as np
import torch
from scipy.special import expit  # sigmoid

from src.fl.server import FedAvgAggregator
from src.defenses.metrics_logger import DefenseMetrics


class SSMTDHMMFilteringServer(DefenseMetrics, FedAvgAggregator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = kwargs.get('config', {})
        defense_config = kwargs.get('defense_cfg', {}) or config.get('defense_params', {})

        self.beta = defense_config.get('beta', 0.2)   # HMM smoothing factor
        self.trust_state = {}  # hidden state per client

        print(f"--- SSMTD Filtering Server (FIXED) ---")
        print(f"beta: {self.beta}")

    # -----------------------------
    # utilities
    # -----------------------------
    def _flatten(self, params):
        return torch.cat([p.detach().flatten().cpu() for p in params.values()])

    def _cosine(self, a, b):
        a = a / (a.norm() + 1e-12)
        b = b / (b.norm() + 1e-12)
        return torch.dot(a, b).item()

    def _apply_update(self, base_params, update_params):
        return {
            k: base_params[k] + update_params[k]
            for k in base_params
        }

    # -----------------------------
    # SSMTD aggregation
    # -----------------------------
    def aggregate(self, current_round=0):

        assert len(self.received_updates) > 0

        client_ids = list(self.received_updates.keys())
        n = len(client_ids)

        base_params = self.get_params()

        base_vec = self._flatten(base_params)

        scores = {}

        print("Evaluating SSMTD cosine + HMM trust...")

        # --------------------------------------------------
        # 1. compute similarity + hidden state update
        # --------------------------------------------------
        for cid in client_ids:

            update = self.received_updates[cid]['params']

            new_params = self._apply_update(base_params, update)
            new_vec = self._flatten(new_params)

            sim = self._cosine(base_vec, new_vec)

            # -----------------------------
            # initialize hidden state
            # -----------------------------
            if cid not in self.trust_state:
                self.trust_state[cid] = 0.5  # neutral prior

            prev_state = self.trust_state[cid]

            # HMM-like update (emission = similarity)
            # higher similarity → higher trust
            emission = expit(5 * (sim - 0.2))  # sharpened sigmoid

            new_state = (
                (1 - self.beta) * prev_state +
                self.beta * emission
            )

            self.trust_state[cid] = new_state

            # final SSMTD score
            score = 0.7 * sim + 0.3 * new_state
            scores[cid] = score

        # --------------------------------------------------
        # 2. median + MAD selection (SSMTD-consistent)
        # --------------------------------------------------

        scores_arr = np.array(list(scores.values()))
        client_ids_arr = np.array(client_ids)

        # median of reputations
        median_r = np.median(scores_arr)

        # MAD (Median Absolute Deviation)
        mad = np.median(np.abs(scores_arr - median_r)) + 1e-12

        # robust threshold (lambda controls strictness)
        lam = 0.0
        threshold = median_r + lam * mad

        print(f"SSMTD Scores: {scores}")


        # select based on robust cutoff
        selected_clients = {
         cid for cid, r in scores.items()
         if r >= threshold
           }

        rejected_clients = set(client_ids) - selected_clients 

        print(f"[SSMTD MAD] median={median_r:.4f}, MAD={mad:.4f}, threshold={threshold:.4f}")
        print(f"Selected: {sorted(selected_clients)}")
        print(f"Rejected: {sorted(rejected_clients)}")

        # --------------------------------------------------
        # 3. filter updates
        # --------------------------------------------------
        self.received_updates = {
            cid: self.received_updates[cid]
            for cid in selected_clients
        }

        self.set_params(base_params)

        # --------------------------------------------------
        # 4. FedAvg
        # --------------------------------------------------
        aggregated = super().aggregate()

        self.update_defense_metrics(
            client_ids_received=set(client_ids),
            rejected_client_ids=rejected_clients
        )

        return aggregated