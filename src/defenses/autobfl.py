import copy as cp
import numpy as np
import torch

from src.fl.server import FedAvgAggregator
from src.defenses.metrics_logger import DefenseMetrics


class AutoBFLFilteringServer(DefenseMetrics, FedAvgAggregator):
    """
    AutoBFL-style data-driven filtering using existing FedAvg + evaluate()

    Pipeline:
    1. For each client:
        - apply update
        - evaluate using server validation set
    2. Score clients
    3. Keep top-k
    4. Call FedAvg on filtered updates
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = kwargs.get('config', {})
        defense_config = kwargs.get('defense_cfg', {})

        if not defense_config:
            defense_config = config.get('defense_params', {})

        self.keep_ratio = defense_config.get('keep_ratio', 0.7)

        print(f"--- AutoBFL Filtering Server ---")
        print(f"Keep ratio: {self.keep_ratio}")

    def _apply_update(self, base_params, update_params):
        return {
            k: base_params[k] + update_params[k]
            for k in base_params
        }    

    def aggregate(self):

        assert len(self.received_updates) > 0, "No updates received."

        client_ids = list(self.received_updates.keys())
        n = len(client_ids)

        # Save original global model
        base_params = self.get_params()

        scores = []

        print("Evaluating client updates (AutoBFL-style)...")

        # --------------------------------------------------
        # 1. Evaluate each client update
        # --------------------------------------------------
        for cid in client_ids:

            update = self.received_updates[cid]['params']

            # apply update
            new_params = self._apply_update(base_params, update)

            # temporarily load model
            self.set_params(new_params)

            eval_res = self.evaluate()
            acc = eval_res['metrics']['main_accuracy']
            loss = eval_res['metrics']['loss']

            # AutoBFL-like scoring
            score = acc - loss

            scores.append(score)

        scores = np.array(scores)    

        # --------------------------------------------------
        # 2. Select top clients
        # --------------------------------------------------
        k = max(1, int(self.keep_ratio * n))

        top_indices = np.argsort(scores)[-k:]

        selected_clients = {client_ids[i] for i in top_indices}
        rejected_clients = set(client_ids) - selected_clients

        print(f"Selected Clients: {sorted(selected_clients)}")
        print(f"Rejected Clients: {sorted(rejected_clients)}")


        # --------------------------------------------------
        # 3. Filter updates BEFORE FedAvg
        # --------------------------------------------------
        self.received_updates = {
            cid: self.received_updates[cid]
            for cid in selected_clients
        }

        # restore base model before aggregation
        self.set_params(base_params)

        # --------------------------------------------------
        # 4. Call standard FedAvg
        # --------------------------------------------------
        aggregated = super().aggregate()


        self.update_defense_metrics(
            client_ids_received=set(client_ids),
            rejected_client_ids=rejected_clients
        )

        return aggregated