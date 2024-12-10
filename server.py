from collections import OrderedDict
from omegaconf import DictConfig

from hydra.utils import instantiate

import os

import torch

from model import test


def save_checkpoint(model, optimizer, server_round, filepath="saved_weights"):
    """Save training state to a checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }

    # Ensure the folder exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # For multiples of 5 (e.g., 0, 5, 10,...)
    if server_round % 5 == 0:
        # Format the filename to include the server_round (e.g., saved_weights/model_0.pth, saved_weights/model_5.pth)
        filename = os.path.join(filepath, f"saved_weights_{server_round}.pth")
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved at round {server_round} as {filename}.")
    else:
        # For other rounds, save as model.pth inside the folder
        filename = os.path.join(filepath, "model.pth")
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved at round {server_round} as {filename}.")


def get_on_fit_config(config: DictConfig):

    def fit_config_fn(server_round: int):
        
        return{'lr': config.lr,
               'momentum': config.momentum,
               'local_epochs': config.local_epochs}
    
    return fit_config_fn


def get_evaluate_fn(model_cfg, testloader, config: DictConfig, checkpoint_path="saved_weights"):
    """Define function for global evaluation on the server."""

    lr = config['lr']
    momentum = config['momentum']

    def evaluate_fn(server_round: int, parameters, config):
        # This function is called by the strategy's `evaluate()` method
        # and receives as input arguments the current round number and the
        # parameters of the global model.
        # this function takes these parameters and evaluates the global model
        # on a evaluation / test dataset.

        model = instantiate(model_cfg)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        #model.load_state_dict(state_dict, strict=False)
        model.load_state_dict(state_dict, strict=True)

        # Here we evaluate the global model on the test set. Recall that in more
        # realistic settings you'd only do this at the end of your FL experiment
        # you can use the `server_round` input argument to determine if this is the
        # last round. If it's not, then preferably use a global validation set.
        loss, accuracy = test(model, testloader, device)

        save_checkpoint(model, optimizer, server_round, checkpoint_path)
        #save_checkpoint(model, optimizer, server_round, checkpoint_path)

        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global test accuracy.
        return loss, {"accuracy": accuracy}

    return evaluate_fn


'''
def get_evaluate_fn(num_classes: int, testloader, config: DictConfig, checkpoint_path="checkpoint.pth"):
    """Define function for global evaluation on the server."""

    lr = config['lr']
    momentum = config['momentum']

    def evaluate_fn(server_round: int, parameters, config):
        # This function is called by the strategy's `evaluate()` method
        # and receives as input arguments the current round number and the
        # parameters of the global model.
        # this function takes these parameters and evaluates the global model
        # on a evaluation / test dataset.

        model = Net(num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        for key, value in zip(model.state_dict().keys(), parameters):
            print(f"Key: {key}, Expected Shape: {model.state_dict()[key].shape}, Provided Shape: {value.shape}")

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


        model.load_state_dict(state_dict, strict=True)
        #model.load_state_dict(state_dict, strict=True)

        # Here we evaluate the global model on the test set. Recall that in more
        # realistic settings you'd only do this at the end of your FL experiment
        # you can use the `server_round` input argument to determine if this is the
        # last round. If it's not, then preferably use a global validation set.
        loss, accuracy = test(model, testloader, device)

        save_checkpoint(model, optimizer, checkpoint_path)
        #save_checkpoint(model, optimizer, server_round, checkpoint_path)

        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global test accuracy.
        return loss, {"accuracy": accuracy}

    return evaluate_fn
'''

'''
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
for key, value in zip(model.state_dict().keys(), parameters):
    # Get the expected shape from the model's state dictionary
    expected_shape = model.state_dict()[key].shape
    provided_shape = value.shape

    # Print key and shape details
    print(f"Key: {key}, Expected Shape: {expected_shape}, Provided Shape: {provided_shape}")

    # Check if the shapes match
    if expected_shape != provided_shape:
        print(f"Shape mismatch for key: {key}")
        print(f"Expected shape: {expected_shape}, Provided shape: {provided_shape}")
    else:
        print(f"Shape matches for key: {key}")
        
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
'''