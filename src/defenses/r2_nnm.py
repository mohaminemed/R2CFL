import torch
from typing import Dict, List, Tuple

from src.fl.server import FedAvgAggregator
from src.defenses.metrics_logger import DefenseMetrics


class REPNNMKrumServer(DefenseMetrics, FedAvgAggregator):
    """
    Reputation-weighted NNM + Multi-Krum
    WITH contribution-aware metrics (corrected + optimized)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = kwargs.get('config', {})
        defense_config = kwargs.get('defense_cfg', {})

        if not config and args:
            config = args[0]
        if not defense_config:
            defense_config = config.get('defense_params', {})

        # Krum params
        self.krum_f = defense_config.get('krum_f', 5)
        self.krum_m = defense_config.get('krum_m', 13)

        # NNM params
        self.nnm_k = defense_config.get('nnm_k', 5)

        # Reputation state
        self.client_reputation: Dict[int, float] = {}
        self.round = 0
        self.T_warmup = 50

        self.malicious_ids = set([0, 1, 2, 3, 4, 5])
        self.rejected_clients: set = set()

        print(
            f"--- Initializing REP-NNM + Multi-Krum Server ---\n"
            f"    Krum f={self.krum_f}, Krum m={self.krum_m}, NNM k={self.nnm_k}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([p.reshape(-1) for p in params.values()])

    def _vector_to_state(
        self, vec: torch.Tensor, reference_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        new_state, ptr = {}, 0
        for k, v in reference_state.items():
            n = v.numel()
            new_state[k] = vec[ptr: ptr + n].view(v.shape)
            ptr += n
        return new_state

    # ------------------------------------------------------------------
    # REP-NNM  (pre-computed distance matrix passed in)
    # ------------------------------------------------------------------

    def _rep_nnm(
        self,
        updates: torch.Tensor,       # (n, d)
        dist_matrix: torch.Tensor,   # (n, n) — pre-computed
        client_ids: List[int],
    ) -> Tuple[torch.Tensor, Dict[int, Dict[int, float]]]:
        """
        Returns:
            mixed_updates : (n, d)
            weight_map    : {update_index -> {client_id -> weight}}
        """
        n = updates.shape[0]
        device = updates.device

        reputations = torch.tensor(
            [self.client_reputation.get(cid, 0.5) for cid in client_ids],
            dtype=torch.float32, device=device,
        )  # (n,)
        kappa = 0.25
        gamma_t = kappa * min(1.0, self.round / self.T_warmup) 
        k = min(self.nnm_k, n - 1)          # neighbours (excluding self)

        mixed = torch.empty_like(updates)
        weight_map: Dict[int, Dict[int, float]] = {}

        # Fetch k nearest neighbours for all nodes at once: (n, k)
        _, nn_indices = torch.topk(dist_matrix, k=k + 1, largest=False, dim=1)
        nn_indices = nn_indices[:, 1:]  # drop self (col 0, distance=0)

        for i in range(n):
            nbr_idx = nn_indices[i]                   # (k,)
            contrib = torch.cat([torch.tensor([i], device=device), nbr_idx])  # (k+1,)

            vectors = updates[contrib]                # (k+1, d)
            rep_w = reputations[contrib]
            rep_w = rep_w / rep_w.sum()

            unif_w = torch.full_like(rep_w, 1.0 / len(contrib))
            weights = (1 - gamma_t) * unif_w + gamma_t * rep_w  # (k+1,)

            mixed[i] = (weights.unsqueeze(1) * vectors).sum(dim=0)

            weight_map[i] = {
                client_ids[idx.item()]: weights[j].item()
                for j, idx in enumerate(contrib)
            }

        return mixed, weight_map

    # ------------------------------------------------------------------
    # Krum scores  (distance matrix passed in)
    # ------------------------------------------------------------------

    def _compute_krum_scores(
        self, dist_matrix: torch.Tensor, n: int
    ) -> torch.Tensor:
        k_neighbors = max(1, n - self.krum_f - 2)
        # topk smallest distances per row; skip the first (self, dist=0)
        top_dists, _ = torch.topk(dist_matrix, k=k_neighbors + 1, largest=False, dim=1)
        return top_dists[:, 1:].sum(dim=1)   # (n,)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def aggregate(self, current_round: int = 0):
        if not self.received_updates:
            return

        self.round = current_round
        client_ids = list(self.received_updates.keys())
        n = len(client_ids)

        if n < self.krum_f + 3:
            print(f"Warning: Only {n} clients — falling back to FedAvg.")
            super().aggregate()
            return

        # ── Flatten ────────────────────────────────────────────────────
        param_refs = [self.received_updates[cid]['params'] for cid in client_ids]
        flat_updates = torch.stack([self._flatten(p) for p in param_refs])  # (n, d)

        # ── Shared pairwise distances (computed once) ──────────────────
        dist_matrix = torch.cdist(flat_updates, flat_updates, p=2)          # (n, n)

        # ── REP-NNM ────────────────────────────────────────────────────
        print(f"Round {self.round}: Applying R2-NNM (k={self.nnm_k})…")
        mixed_updates, weight_map = self._rep_nnm(flat_updates, dist_matrix, client_ids)

        # ── Krum scoring on mixed updates ──────────────────────────────
        mixed_dist = torch.cdist(mixed_updates, mixed_updates, p=2)
        scores = self._compute_krum_scores(mixed_dist, n)

        # ── Multi-Krum selection ───────────────────────────────────────
        m_select = min(n, max(1, self.krum_m))
        _, top_indices = torch.topk(scores, k=m_select, largest=False)

        # ── Effective contributors ─────────────────────────────────────
        contributing_clients: set = set()
        contribution_weights: Dict[int, float] = {}

        for i in top_indices.tolist():
            for cid, w in weight_map[i].items():
                contributing_clients.add(cid)
                contribution_weights[cid] = contribution_weights.get(cid, 0.0) + w

        self.rejected_clients = set(client_ids) - contributing_clients

        print(f"Selected centers : {[client_ids[i] for i in top_indices.tolist()]}")
        print(f"Effective contributors: {sorted(contributing_clients)}")
        print(f"Rejected (true)  : {sorted(self.rejected_clients)}")

        # ── Final aggregation ──────────────────────────────────────────
        final_update = mixed_updates[top_indices].mean(dim=0)
        new_params = self._vector_to_state(final_update, param_refs[0])

        # ── Defense metrics ────────────────────────────────────────────
        self.update_defense_metrics(
            client_ids_received=set(client_ids),
            rejected_client_ids=self.rejected_clients,
        )

        # ── EMPR (optional diagnostic) ─────────────────────────────────
        if hasattr(self, 'malicious_ids'):
            total_w = sum(contribution_weights.values())
            mal_w = sum(w for cid, w in contribution_weights.items() if cid in self.malicious_ids)
            print(f"EMPR: {mal_w / total_w:.4f}" if total_w > 0 else "EMPR: N/A")

        # ── Reputation update ──────────────────────────────────────────
        self._update_reputation(client_ids, scores)

        # ── Commit global model ────────────────────────────────────────
        self.set_params({k: v.to(self.device) for k, v in new_params.items()})
        self.received_updates = {}

    # ------------------------------------------------------------------
    # Reputation update  (vectorized)
    # ------------------------------------------------------------------

    def _update_reputation(self, client_ids: List[int], scores: torch.Tensor):
        eps = 1e-12
        tau = 2.0
        alpha = 0.25
        beta = 0.95

        s_vals = scores                            # already a tensor
        median_s = s_vals.median()
        mad_s = (s_vals - median_s).abs().median().clamp(min=eps)

        scaled = (s_vals - median_s) / mad_s      # (n,)
        C_vals = torch.sigmoid(-scaled / tau)      # (n,) — lower score ⇒ higher C

        for i, cid in enumerate(client_ids):
            reliable = 0.0 if cid in self.rejected_clients else 1.0
            C_i = alpha * C_vals[i].item() + (1 - alpha) * reliable

            if cid not in self.client_reputation:
                self.client_reputation[cid] = 0.7

            self.client_reputation[cid] = (
                beta * self.client_reputation[cid] + (1 - beta) * C_i
            )

            print(
                f"Client {cid:3d}: Krum={scores[i].item():.4f}, "
                f"C={C_i:.4f}, rep={self.client_reputation[cid]:.4f}"
            )

        print(f"[Reputation] Updated for {len(client_ids)} clients")