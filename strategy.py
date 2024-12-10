from typing import Optional, Tuple, List, Dict, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import numpy as np
import torch


from model import Net, test
from server import get_evaluate_fn
from collections import OrderedDict


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def save_checkpoint(parameters: Parameters, round_number: int) -> None:
  """Save model parameters to a checkpoint."""
  ndarrays = parameters_to_ndarrays(parameters)
  model_state_dict = OrderedDict({f"layer_{i}": torch.tensor(arr) for i, arr in enumerate(ndarrays)})

  # Save as a PyTorch model checkpoint
  checkpoint_path = f"/content/saved_weights/round-{round_number}.pth"
  torch.save(model_state_dict, checkpoint_path)
  print(f"Checkpoint saved to {checkpoint_path}")



class FedCustomRobust(Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        clip_threshold: float = 1.0,  # Threshold for gradient clipping
        trim_fraction: float = 0.2,   # Fraction to trim during aggregation
        initial_parameters: Optional[Parameters] = None,  # Initial weights and biases
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.clip_threshold = clip_threshold
        self.trim_fraction = trim_fraction
        self.initial_parameters = initial_parameters

    def __repr__(self) -> str:
        return "FedCustomRobust"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        if self.initial_parameters is not None:
            print("Using provided initial parameters.")
            return self.initial_parameters  # Use provided initial weights and biases
        
        # Default initialization
        net = Net()
        ndarrays = get_parameters(net)
        print("Using default initialized parameters.")
        return ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configurations for clients
        fit_configurations = []
        for client in clients:
            config = {"lr": 0.001}
            fit_configurations.append((client, FitIns(parameters, config)))
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using robust aggregation (Trimmed Mean)."""
        if not results:
            return None, {}

        # Convert updates to ndarray format and clip gradients
        updates = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        clipped_updates = [
            (self._clip_gradients(update), num_examples) for update, num_examples in updates
        ]

        # Apply Trimmed Mean aggregation
        aggregated_update = self._trimmed_mean(clipped_updates, self.trim_fraction)

        # Convert back to Parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_update)
        metrics_aggregated = {}

        # Save model parameters every 5 rounds
        if server_round % 5 == 0:
            print(f"Saving model checkpoint for round {server_round}")
            save_checkpoint(parameters_aggregated, server_round)

        return parameters_aggregated, metrics_aggregated


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        # Assume no server-side evaluation
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def _clip_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Clip gradients to limit the impact of outliers."""
        return [
            np.clip(grad, -self.clip_threshold, self.clip_threshold) for grad in gradients
        ]

    def _trimmed_mean(
        self, updates: List[Tuple[List[np.ndarray], int]], trim_fraction: float
    ) -> List[np.ndarray]:
        """Perform Trimmed Mean aggregation."""
        num_clients = len(updates)
        num_to_trim = int(trim_fraction * num_clients)

        # Stack updates
        stacked_updates = [np.stack([upd[i] for upd, _ in updates]) for i in range(len(updates[0][0]))]

        # Perform trimming and calculate mean
        trimmed_mean = [
            np.mean(
                np.sort(stack, axis=0)[num_to_trim: -num_to_trim],
                axis=0,
            )
            for stack in stacked_updates
        ]
        return trimmed_mean
