from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

import torch
import flwr as fl
from flwr.common import Context
from flwr.client import Client, ClientApp, NumPyClient


from hydra.utils import instantiate

from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 partition_id,
                 net,
                 trainloader,
                 valloader) -> None:
        super().__init__()

        self.partition_id = partition_id
        self.net = Net(10)

        self.trainloader = trainloader
        self.valloader = valloader

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def set_parameters(self, parameters):
        #print("set_parameters set_parameters set_parameters set_parameters")

        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        #self.model.state_dict().keys() = odict_keys(['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias'])
        #state_dict = state_dict <class 'collections.OrderedDict'>

        self.net.load_state_dict(state_dict, strict=True)

    
    def get_parameters(self, config: Dict[str, Scalar]):
        #print("get_parameters get_parameters get_parameters get_parameters")
        #self.model.state_dict().keys() = odict_keys(['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias'])
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]


    def fit(self, parameters, config):
        #print("fit fit fit fit fit fit fit fit fit fit fit fit")
        self.set_parameters(parameters)
        lr = 0.01
        epochs = 10
        #lr = config['lr']
        #momentum = config['momentum']
        #epochs = config['local_epochs']

        #optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        optim = torch.optim.SGD(self.net.parameters(), lr=lr)
        optim.param_groups[0]['initial_lr'] = lr 
        # Create learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=3, verbose=True)

        #Local Training
        train(self.net, self.trainloader, optim, epochs, self.device, scheduler)

        return self.get_parameters({}), len(self.trainloader), {}
    

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        #print("evaluate evaluate evaluate evaluate evaluate evaluate")
        self.set_parameters(parameters)

        loss, accuracy = test(self.net, self.valloader, self.device)

        return float(loss), len(self.valloader), {'accuracy': accuracy}

'''
def client_fn(context: Context) -> Client:
    DEVICE = "cpu"
    net = Net().to(DEVICE)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, _ = load_datasets(partition_id, num_partitions)
    return FlowerClient(partition_id, net, trainloader, valloader).to_client()
'''

def generate_client_fn(trainloaders, valloaders):

    def client_fn(context: Context):
        DEVICE = "cpu"
        net = Net().to(DEVICE)
        partition_id = context.node_config["partition-id"]
        #num_partitions = context.node_config["num-partitions"]
        cid = context.node_config['partition-id']
        return FlowerClient(partition_id,
                            net,
                            trainloaders[int(cid)],
                            valloaders[int(cid)],
                            ).to_client()

    return client_fn

'''
def generate_client_fn(trainloaders, valloaders, model_cfg):

    def client_fn(context: Context):
        cid = context.node_config['partition-id']
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader = valloaders[int(cid)],
                            model_cfg = model_cfg
                            ).to_client()

    return client_fn
'''

'''
def generate_client_fn(trainloaders, valloaders, num_classes):

    def client_fn(cid: str):
        print("11111111111111111111111")
        print(int(cid))
        print("11111111111111111111111")
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader = valloaders[int(cid)],
                            num_classes = num_classes
                            ).to_client()

    return client_fn
'''