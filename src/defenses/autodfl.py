import numpy as np
from src.fl.server import FedAvgAggregator
from src.defenses.metrics_logger import DefenseMetrics


class AutoDFLFilteringServer(DefenseMetrics, FedAvgAggregator):
    """
    AutoDFL-style data-driven filtering defense.

    Pipeline:
    1. Evaluate base model on validation set
    2. For each client:
        - Apply update
        - Evaluate new model
        - Compute delta performance (acc + loss improvement)
    3. Select top-k clients
    4. Run standard FedAvg on selected updates
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = kwargs.get('config', {})
        defense_config = kwargs.get('defense_cfg', {})

        if not defense_config:
            defense_config = config.get('defense_params', {})

        self.keep_ratio = defense_config.get('keep_ratio', 0.7)

        print(f"--- AutoDFL Filtering Server ---")
        print(f"Keep ratio: {self.keep_ratio}")

    # ------------------------------------------------------------
    # Apply update (w + Δ)
    # ------------------------------------------------------------
    def _apply_update(self, base_params, update_params):
        return {
            k: base_params[k] + update_params[k]
            for k in base_params
        }

    # ------------------------------------------------------------
    # Main aggregation
    # ------------------------------------------------------------
    def aggregate(self, current_round=0):

        assert len(self.received_updates) > 0, "No updates received."

        client_ids = list(self.received_updates.keys())
        n = len(client_ids)

        # Save base model
        base_params = self.get_params()

        # --------------------------------------------------
        # 1. Evaluate base model
        # --------------------------------------------------
        self.set_params(base_params)
        base_eval = self.evaluate()

        base_acc = base_eval['metrics']['main_accuracy']
        base_loss = base_eval['metrics']['loss']

        scores = []

        print("Evaluating AutoDFL-style contributions...")

        # --------------------------------------------------
        # 2. Evaluate each client update
        # --------------------------------------------------
        for cid in client_ids:

            update = self.received_updates[cid]['params']

            # Apply update
            new_params = self._apply_update(base_params, update)
            self.set_params(new_params)

            eval_res = self.evaluate()

            acc = eval_res['metrics']['main_accuracy']
            loss = eval_res['metrics']['loss']

            # AutoDFL scoring (delta-based)
            delta_acc = acc - base_acc
            delta_loss = base_loss - loss

            score = delta_acc + delta_loss
            scores.append(score)

        scores = np.array(scores)

        # --------------------------------------------------
        # 3. Select top-k clients
        # --------------------------------------------------
        k = max(1, int(self.keep_ratio * n))
        top_indices = np.argsort(scores)[-k:]

        selected_clients = {client_ids[i] for i in top_indices}
        rejected_clients = set(client_ids) - selected_clients

        print(f"Selected Clients: {sorted(selected_clients)}")
        print(f"Rejected Clients: {sorted(rejected_clients)}")

        # --------------------------------------------------
        # 4. Filter updates
        # --------------------------------------------------
        self.received_updates = {
            cid: self.received_updates[cid]
            for cid in selected_clients
        }

        # Restore base model before aggregation
        self.set_params(base_params)

        # --------------------------------------------------
        # 5. Standard FedAvg
        # --------------------------------------------------
        aggregated = super().aggregate()

        # --------------------------------------------------
        # 6. Metrics
        # --------------------------------------------------
        self.update_defense_metrics(
            client_ids_received=set(client_ids),
            rejected_client_ids=rejected_clients
        )

        return aggregated
