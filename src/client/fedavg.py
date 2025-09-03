from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from data.utils.datasets import BaseDataset
from src.utils.metrics import Metrics
from src.utils.models import DecoupledModel
from src.utils.tools import NestedNamespace, evalutate_model, get_optimal_cuda_device
from rich.console import Console
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from src.utils.tools import (
    Logger,
    NestedNamespace,
    fix_random_seed,
    get_optimal_cuda_device,
)
from src.utils.constants import (
    DATA_MEAN,
    DATA_STD,
    FLBENCH_ROOT,
    LR_SCHEDULERS,
    MODE,
    OPTIMIZERS,
    OUT_DIR,
)
from src.utils import utils
class FedAvgClient:
    def __init__(
        self,
        model: DecoupledModel,
        optimizer_cls: type[torch.optim.Optimizer],
        lr_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler],
        args: NestedNamespace,
        dataset: BaseDataset,
        data_indices: list,
        device: torch.device | None,
        return_diff: bool,
    ):
        self.client_id: int = None
        self.args = args
        if device is None:
            self.device = get_optimal_cuda_device(use_cuda=self.args.common.use_cuda)
        else:
            self.device = device
        self.dataset = dataset
        
        #self.dataset = torch.utils.data.DataLoader(dataset,  
        #transform=transforms.Compose([transforms.ToTensor(), 
        #transforms.Normalize((0.1307,), (0.3081,))])), 
        #batch_size=self.args.common.batch_size, shuffle=True)
        
        self.model = model.to(self.device)
        self.regular_model_params: OrderedDict[str, torch.Tensor]
        self.personal_params_name: list[str] = []
        self.regular_params_name = list(key for key, _ in self.model.named_parameters())
        if self.args.common.buffers == "local":
            self.personal_params_name.extend(
                [name for name, _ in self.model.named_buffers()]
            )
        elif self.args.common.buffers == "drop":
            self.init_buffers = deepcopy(OrderedDict(self.model.named_buffers()))

        self.optimizer = optimizer_cls(params=self.model.parameters())
        self.init_optimizer_state = deepcopy(self.optimizer.state_dict())

        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.init_lr_scheduler_state: dict = None
        self.lr_scheduler_cls = None
        if lr_scheduler_cls is not None:
            self.lr_scheduler_cls = lr_scheduler_cls
            self.lr_scheduler = lr_scheduler_cls(optimizer=self.optimizer)
            self.init_lr_scheduler_state = deepcopy(self.lr_scheduler.state_dict())

        # [{"train": [...], "val": [...], "test": [...]}, ...]
        self.data_indices = data_indices
        # Please don't bother with the [0], which is only for avoiding raising runtime error by setting Subset(indices=[]) with `DataLoader(shuffle=True)`
        self.trainset = Subset(self.dataset, indices=[0])        
        self.valset = Subset(self.dataset, indices=[])
        self.testset = Subset(self.dataset, indices=[])
        self.trainloader = DataLoader(
            self.trainset, batch_size=self.args.common.batch_size, shuffle=True
        )
        self.valloader = DataLoader(self.valset, batch_size=self.args.common.batch_size)
        self.testloader = DataLoader(
            self.testset, batch_size=self.args.common.batch_size
        )
        
        self.testing = False

        self.local_epoch = self.args.common.local_epoch
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.eval_results = {}

        self.return_diff = return_diff
        

    def load_data_indices(self):
        """This function is for loading data indices for No.`self.client_id`
        client."""
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.testset.indices = self.data_indices[self.client_id]["test"] 
        
       

    def train_with_eval(self):
        if self.local_epoch > 0:
            self.fit()


    def set_parameters(self, package: dict[str, Any]):
        self.client_id = package["client_id"]
        self.local_epoch = package["local_epoch"]
        self.load_data_indices()

        if package["optimizer_state"]:
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)

        if self.lr_scheduler is not None:
            if package["lr_scheduler_state"]:
                self.lr_scheduler.load_state_dict(package["lr_scheduler_state"])
            else:
                self.lr_scheduler.load_state_dict(self.init_lr_scheduler_state)

        self.model.load_state_dict(package["regular_model_params"], strict=False)
        self.model.load_state_dict(package["personal_model_params"], strict=False)
        if self.args.common.buffers == "drop":
            self.model.load_state_dict(self.init_buffers, strict=False)

        if self.return_diff:
            model_params = self.model.state_dict()
            self.regular_model_params = OrderedDict(
                (key, model_params[key].clone().cpu())
                for key in self.regular_params_name
            )

    def train(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        self.train_with_eval()
        client_package = self.package()
        return client_package

    def package(self):
        model_params = self.model.state_dict()
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            regular_model_params={
                key: model_params[key].clone().cpu() for key in self.regular_params_name
            },
            personal_model_params={
                key: model_params[key].clone().cpu()
                for key in self.personal_params_name
            },
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {}
                if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),
        )
        if self.return_diff:
            client_package["model_params_diff"] = {
                key: param_old - param_new
                for (key, param_new), param_old in zip(
                    client_package["regular_model_params"].items(),
                    self.regular_model_params.values(),
                )
            }
            client_package.pop("regular_model_params")
        return client_package
        
        
    def show_dataset_information(self):
        count = [0 for i in range(10)]
        for x, y in self.trainloader:
            #print(y)
            for i in y:
                
                count[i] = count[i] + 1
        print('client ', self.client_id, ' train ', end='')
        for j in range(len(count)):
            print(',', j , ':', count[j], end='')
        print()
        print('client ', self.client_id, ' test ',end='')
        count = [0 for i in range(10)]
        for x, y in self.testloader:
            for i in y:
                count[i] = count[i] + 1
        for i in range(len(count)):
            print(',', i , ':', count[i],end='')
        print()
        
        
    def fit(self):
        #torch.save(self.model.state_dict(), 'model.pt')
        #print('save model')
        #exit(0)

        self.model.train()
        self.dataset.train()  
        self.show_dataset_information()
        
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:                
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        [num, correct, loss, acc] = self.evaluate()
               
    def criterion(self, outputs, targets):
        return self.mse(outputs, targets)

    @torch.no_grad()
    #def evaluate(self, model: torch.nn.Module = None) -> dict[str, Metrics]:
    def evaluate(self, model: torch.nn.Module = None):        
        target_model = self.model if model is None else model
        target_model.eval()
        #self.dataset.eval()      
        if self.args.common.method == 'fednaca':
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        
        #if len(self.testset) > 0 and self.args.common.eval_test:            
        [num, correct, loss, acc] = evalutate_model(
            model=target_model,
            dataloader=self.testloader,
            criterion=criterion,
            device=self.device,
        )
        
        #print("Evaluate in client {:2d}, testing sample {:5d}, loss {:>2.4f}, acc {:>2.2f}%".format(self.client_id, num, loss, acc * 100))
        return [num, correct, loss, acc]

    #def test(self, server_package: dict[str, Any]) -> dict[str, dict[str, Metrics]]:
    def test(self, server_package: dict[str, Any]):       
        self.testing = True
        self.set_parameters(server_package)

        [num, correct, loss, acc] = self.evaluate()
        self.testing = False
        #return results
        utils.testing_result.append([num, correct, loss, acc])
        
        return [num, loss, acc]
    def finetune(self):
        """Client model finetuning.

        This function will only be activated in `test()`
        """
        self.model.train()
        self.dataset.train()
        for _ in range(self.args.common.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
