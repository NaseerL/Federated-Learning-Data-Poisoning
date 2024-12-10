import pickle
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import Strategy

import torch

from dataset import prepare_dataset
from client import generate_client_fn
from server import get_evaluate_fn
from model import Net
from strategy import FedCustomRobust


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    if cfg.poison_dataset == False:
        data_dir_pos = None
        num_partitions_pos = None
        num_partitions = cfg.num_clients

    elif cfg.poison_dataset == True:
        data_dir_pos = cfg.data_dir_pos_path
        num_partitions = int(cfg.num_clients) - int(cfg.no_of_clients_poison)
        num_partitions_pos = cfg.no_of_clients_poison

    ## 2. Prepare your dataset
    trainloaders, valloaders, testloader = prepare_dataset(num_partitions = num_partitions,
                                                           num_partitions_pos = num_partitions_pos,
                                                           data_dir = cfg.data_dir_path,
                                                           data_dir_pos = data_dir_pos,
                                                           batch_size = cfg.batch_size,
                                                           val_ratio = 0.1)

    ## 3. Define your clients
    client_fn = generate_client_fn(trainloaders, valloaders)
    client = ClientApp(client_fn=client_fn)

    def load_weights(model, weights_path):
      """Load weights into the global model."""
      checkpoint = torch.load(weights_path)  # Load the checkpoint file
      if "model_state_dict" in checkpoint:
          model.load_state_dict(checkpoint["model_state_dict"])  # Extract and load model weights
      else:
          model.load_state_dict(checkpoint)  # Directly load if it's a state_dict
      return model
    
    # Convert state_dict to a list of NumPy arrays
    def state_dict_to_ndarrays(state_dict):
        return [param.cpu().numpy() for param in state_dict.values()]
    




    ## 4. Load global model weights if resume
    initial_parameters = None
    if cfg.train_cont == True:
        print(f"Loading weights from {cfg.resume_weights_path}")
        global_model = Net(cfg.num_classes)
        global_model = load_weights(global_model, cfg.resume_weights_path)
        # Convert to Flower Parameters
        initial_parameters = ndarrays_to_parameters(state_dict_to_ndarrays(global_model.state_dict()))
        print("model loaded")

    checkpoint_path = "saved_weights"
    evaluate_fn = get_evaluate_fn(testloader, checkpoint_path=checkpoint_path)



    strategy = FedCustomRobust(fraction_fit=0.001,
                               fraction_evaluate=0.001, 
                               min_fit_clients=10, 
                               min_evaluate_clients=10,  
                               min_available_clients=10, 
                               clip_threshold=1.0, 
                               trim_fraction=0.2,
                               #evaluate_fn=evaluate_fn,
                               initial_parameters=initial_parameters)


    def server_fn(context: Context) -> ServerAppComponents:
        # Configure the server for just 3 rounds of training
        config = ServerConfig(num_rounds=20)
        return ServerAppComponents(
            config=config,
            strategy=strategy,  # <-- pass the new strategy here
        )

    # Create the ServerApp
    server = ServerApp(server_fn=server_fn)


    backend_config = {"client_resources": {"num_gpus": 1, "num_cpus": 2}}

    run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=20,
    backend_config=backend_config
    )
    
    
if __name__ == "__main__":

    main()