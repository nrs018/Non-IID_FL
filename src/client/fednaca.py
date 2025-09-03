from src.client.fedavg import FedAvgClient
import torch
import numpy as np
from src.utils import utils
import time
from src.utils.tools import NestedNamespace, evalutate_model, get_optimal_cuda_device
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchsummary import summary
import torch.nn.functional as F
import torch.optim as optim

class FedNacaClient(FedAvgClient):
    def __init__(self, **commons):
        super(FedNacaClient, self).__init__(**commons)
    #def __init__(self, model, nlab, nepochs=100, sbatch=16, lr=0.01, lr_min=5e-4, lr_factor=1, lr_patience=5, clipgrad=10000, args=None):
        #self.model = model
        #self.args = args
        #self.nepochs = nepochs
        #self.batch_size = sbatch
        self.lr = 0.01
        #self.lr_min = lr_min
        #self.lr_factor = lr_factor
        #self.lr_patience = lr_patience
        #self.clipgrad = clipgrad
        self.nlab = 10 #nlab
        self.mse = torch.nn.MSELoss()
        self.optimizer = self._get_optimizer()

        return
    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        
        return torch.optim.SGD(self.model.parameters(), lr=lr) 
        
    #def train(self, t, xtrain, ytrain, xvalid, yvalid):    
    def fit(self):
        self.model.train()
        self.dataset.train()
        self.show_dataset_information()
        
        global_params = list(self.model.parameters())    
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        
        #patience = self.lr_patience        
        #self.optimizer = self._get_optimizer(lr=0.0005)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0005, momentum=0.9, nesterov=False)
        loss = (F.mse_loss, (lambda l: l))
        # Loop epochs
        count_num = [0 for i in range(10)]        
        for e in range(self.local_epoch):            
            # Train
            clock0 = time.time()            
            for x, y in self.trainloader:           
                #x, y = x.to(device), y.to(device)
                #with torch.no_grad():
                images = torch.autograd.Variable(x)
                target = torch.autograd.Variable(y)
                targets = torch.zeros(images.shape[0], self.nlab).to(target.device).scatter_(1, target.unsqueeze(1).long(), 1.0)
                self.optimizer.zero_grad()    
                output = self.model.forward(images, targets)
                loss_val = loss[0](output, loss[1](targets))
                loss_val.backward(retain_graph=False)
                # Apply step                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10000)
                self.optimizer.step()
                #     
                # Forward 正确运行的代码                                
                #output, _ = self.model.forward(images, targets)      
                # Apply step                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10000)
                #self.optimizer.step()
                #self.optimizer.zero_grad() 
                #正确运行的代码                
            
               
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
    
    def eval(self):
        print('local test, client id:',client_id) 
        target_model = self.model
        target_model.eval()
        self.dataset.eval() 
        
        if self.args.common.method == 'fednaca':
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction="sum")
            
        [num, correct, loss, acc] = evalutate_model(
                model=target_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
            )
        print(num, correct, loss, acc, 'kkkkkkkkkkkkkkkkkkk')
        utils.testing_result.append([num, correct, loss, acc])
        return loss, acc
    