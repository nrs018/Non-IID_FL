from collections import OrderedDict
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor

from src.utils.constants import DATA_SHAPE, INPUT_CHANNELS, NUM_CLASSES
from src.utils.tools import NestedNamespace
from src.utils import utils
import torch.nn.functional as F
import numpy as np
import math
from numpy import prod
import datetime

class DecoupledModel(nn.Module):
    def __init__(self):
        super(DecoupledModel, self).__init__()
        self.need_all_features_flag = False
        self.all_features = []
        self.base: nn.Module = None
        self.classifier: nn.Module = None
        self.dropout: list[nn.Module] = []

    def need_all_features(self):
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def _get_feature_hook_fn(model, input, output):
            if self.need_all_features_flag:
                self.all_features.append(output.detach().clone())

        for module in target_modules:
            module.register_forward_hook(_get_feature_hook_fn)

    def check_and_preprocess(self, args: NestedNamespace):
        if self.base is None or self.classifier is None:
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )
        self.dropout = [
            module
            for module in list(self.base.modules()) + list(self.classifier.modules())
            if isinstance(module, nn.Dropout)
        ]
        if args.common.buffers == "global":
            for module in self.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    buffers_list = list(module.named_buffers())
                    for name_buffer, buffer in buffers_list:
                        # transform buffer to parameter
                        # for showing out in model.parameters()
                        delattr(module, name_buffer)
                        module.register_parameter(
                            name_buffer,
                            torch.nn.Parameter(buffer.float(), requires_grad=False),
                        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.base(x))

    def get_last_features(self, x: Tensor, detach=True) -> Tensor:
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
        out = self.base(x)

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return func(out)

    def get_all_features(self, x: Tensor) -> Optional[list[Tensor]]:
        feature_list = None
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        self.need_all_features_flag = True
        _ = self.base(x)
        self.need_all_features_flag = False

        if len(self.all_features) > 0:
            feature_list = self.all_features
            self.all_features = []

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return feature_list


# CNN used in FedAvg
class FedAvgCNN(DecoupledModel):
    feature_length = {
        "mnist": 1024,
        "medmnistS": 1024,
        "medmnistC": 1024,
        "medmnistA": 1024,
        "covid19": 196736,
        "fmnist": 1024,
        "emnist": 1024,
        "femnist": 1,
        "cifar10": 1600,
        "cinic10": 1600,
        "cifar100": 1600,
        "tiny_imagenet": 3200,
        "celeba": 133824,
        "svhn": 1600,
        "usps": 800,
    }

    def __init__(self, dataset: str):
        super(FedAvgCNN, self).__init__()
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 32, 5),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(self.feature_length[dataset], 512),
                activation3=nn.ReLU(),
            )
        )
        self.classifier = nn.Linear(512, NUM_CLASSES[dataset])


class LeNet5(DecoupledModel):
    feature_length = {
        "mnist": 256,
        "medmnistS": 256,
        "medmnistC": 256,
        "medmnistA": 256,
        "covid19": 49184,
        "fmnist": 256,
        "emnist": 256,
        "femnist": 256,
        "cifar10": 400,
        "cinic10": 400,
        "svhn": 400,
        "cifar100": 400,
        "celeba": 33456,
        "usps": 200,
        "tiny_imagenet": 2704,
    }

    def __init__(self, dataset: str) -> None:
        super(LeNet5, self).__init__()
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 6, 5),
                bn1=nn.BatchNorm2d(6),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(6, 16, 5),
                bn2=nn.BatchNorm2d(16),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(self.feature_length[dataset], 120),
                activation3=nn.ReLU(),
                fc2=nn.Linear(120, 84),
                activation4=nn.ReLU(),
            )
        )

        self.classifier = nn.Linear(84, NUM_CLASSES[dataset])
        
############################################
def expectation(labels):
    sigma = 1
    delta_mu = 1

    max_len = labels.shape[1]
    a = np.array([np.sqrt(2) * np.sqrt(np.log(max_len / i)) * sigma for i in range(1, max_len + 1)])
    a = a / a.max() * (2 * (max_len - delta_mu))
    b = delta_mu + a
    a = torch.tensor(a.astype('int')).to(labels.device)
    assert len(set(a.cpu().numpy().tolist())) == len(a.cpu().numpy().tolist()), 'error in expectation'
    b = torch.tensor(b.astype('int')).to(labels.device)
    Ea = a[torch.max(labels, 1)[1].cpu()]
    Eb = b[torch.max(labels, 1)[1].cpu()]    
    Ea = torch.zeros(labels.shape[0], 2 * labels.shape[1], device=labels.device).scatter_(1, Ea.unsqueeze(1).long(), 1.0)
    
    Eb = torch.zeros(labels.shape[0], 2 * labels.shape[1], device=labels.device).scatter_(1, Eb.unsqueeze(1).long(), 1.0)
    return (Ea + Eb) / 2


def local_modulation(neuromodulator_level):
    
    lambda_inv = 0.5#utils.args.common.lambda_inv
    theta_max = 1.2#utils.args.common.theta_max
    with torch.no_grad():
        nl_ = neuromodulator_level.clone().detach()
        modulation = torch.zeros_like(neuromodulator_level).cpu()
        phase_one = theta_max - (theta_max - 1) * (4 * nl_ - lambda_inv).pow(2) / lambda_inv**2
        phase_two = 4 * (nl_ - lambda_inv).pow(2) / lambda_inv**2
        phase_three = -4 * ((2 * lambda_inv - nl_) - lambda_inv).pow(2) / lambda_inv**2
        phase_four = (theta_max - 1) * (4 * (2 * lambda_inv - nl_) - lambda_inv).pow(2) / lambda_inv**2 - theta_max

        modulation[neuromodulator_level <= 0.5 * lambda_inv] = phase_one[neuromodulator_level <= 0.5 * lambda_inv]
        modulation[(0.5 * lambda_inv < neuromodulator_level) & (neuromodulator_level <= lambda_inv)] = phase_two[(0.5 * lambda_inv < neuromodulator_level) & (neuromodulator_level <= lambda_inv)]
        modulation[(lambda_inv < neuromodulator_level) & (neuromodulator_level <= 1.5 * lambda_inv)] = phase_three[(lambda_inv < neuromodulator_level) & (neuromodulator_level <= 1.5 * lambda_inv)]
        modulation[1.5 * lambda_inv < neuromodulator_level] = phase_four[1.5 * lambda_inv < neuromodulator_level]
    
    return modulation


def reset_weights_NI(NI):
    #if utils.args.distribution == 'uniform':
    torch.nn.init.uniform_(NI)
    #elif utils.args.distribution == 'normal':
    #    torch.nn.init.normal_(NI, mean=0.5, std=1)
    #    NI.clamp_(0, 1)
    #elif utils.args.distribution == 'beta':
    #    dist = Beta(torch.ones_like(NI) * 0.5, torch.ones_like(NI) * 0.5)
    #    NI.data = dist.sample()
    return NI
    
    
#---------------------------------------------

class Linear(torch.nn.Module):
    def __init__(self, args, in_features, out_features, nlab, layer=None):
        super(Linear, self).__init__()
        self.args = args
        if layer != -1:
            self.fc = torch.nn.Linear(in_features, out_features, bias=False)
            self.NI = torch.empty(2 * nlab, out_features).cpu()
            self.NI = reset_weights_NI(self.NI)
            self.NI.requires_grad = False
        else:
            self.fc = torch.nn.Linear(in_features, out_features, bias=False)
        
        self.in_features = in_features
        self.out_features = out_features
        self.nlab = nlab
        self.layer = layer
        self.act = nn.Sigmoid()

    # t: current task, x: input, y: output
    def forward(self, x, y):
        self.input_ = x  
        if len(y.shape) < 2:
            y = F.one_hot(y.to(torch.int64), num_classes=10)        
        self.label_ = y
       
        u = self.act(self.fc(x))
        if self.layer != -1:
            u_mask = torch.mean(self.input_, 0, False)
            u_mask = F.interpolate(u_mask.unsqueeze(0).unsqueeze(0), size=[self.out_features])
            u_mask = u_mask.squeeze(0)
            
            if utils.bias[self.args.common.experiment] is not None:
                bias = utils.bias[self.args.common.experiment]
            else:
                bias = u_mask.max() - utils.delta_bias[self.args.common.experiment]
            
            u_mask = torch.sigmoid(1000 * (u_mask - bias))
            u = u * u_mask.expand_as(u)
        u.requires_grad_()    
        if y is not None:
            # Hidden layers
            if self.layer != -1:                
                neuromodulator_level = expectation(y).mm(self.NI.view(-1, prod(self.NI.shape[1:]))).view(u.shape)                
                u.backward(gradient=local_modulation(neuromodulator_level), retain_graph=True)
            # Output layers
            else:
                # MSE
                err = u - y
                err = torch.matmul(err, torch.eye(err.shape[1]).to(err.device))
                u.backward(gradient=err, retain_graph=False)
        return u

    
class MLPNet(DecoupledModel):
    feature_length = {
        "mnist": 784,
        "medmnistS": 784,
        "medmnistC": 784,
        "medmnistA": 784,
        "fmnist": 784,
        "emnist": 784,
        "femnist": 784,
        "cifar10": 3072,
        "cinic10": 3072,
        "svhn": 3072,
        "cifar100": 3072,
        "usps": 1536,
        "synthetic": DATA_SHAPE["synthetic"],
    }

    def __init__(self, args, dataset):
        super(MLPNet, self).__init__()
        self.args = args
        self.nhid = 1000
        self.labsize = 10
        self.layers = nn.ModuleList()
        self.nlayers = 1 
        self.fcs = nn.ModuleList()

        # hidden layer
        for i in range(self.nlayers):
            if i == 0:                
                fc = Linear(self.args, self.feature_length[self.args.common.dataset], self.nhid, self.labsize, layer=i)
            else:
                fc = Linear(self.args, self.nhid, self.nhid, self.labsize, layer=i)
            self.fcs.append(fc)

        self.last = Linear(self.args, self.nhid, self.labsize, self.labsize, layer=-1)

        self.gate = torch.nn.Sigmoid()

        return
    def need_all_features(self):
        return

    #def forward(self, x):
    def forward(self, x, laby=None):
        '''x = torch.flatten(x, start_dim=1)
        x = self.classifier(self.base(x))
        '''
        # input
        u = x.reshape(x.size(0), -1)
        # hidden
        
        for li in range(len(self.fcs)):            
            u = self.fcs[li](u, laby)
            u = u.detach()
        # output
        #if self.args.multi_output:
        #    y = self.last[t](u, laby)
        #else:
        y = self.last(u, laby)

        hidden_out = u
        return y, hidden_out
        #return x

    def get_last_features(self, x, detach=True):
        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        x = torch.flatten(x, start_dim=1)
        x = self.base(x)
        return func(x)

    def get_all_features(self, x):
        raise RuntimeError("MLPNet has 0 Conv layer, so is unable to get all features.")
 
class Activation(nn.Module):
    def __init__(self, activation):
        super(Activation, self).__init__()

        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "none":
            self.act = None
        else:
            raise NameError("=== ERROR: activation " + str(activation) + " not supported")

    def forward(self, x):
        if self.act == None:
            return x
        else:
            return self.act(x)

 
class FC_block(nn.Module):
    def __init__(self, in_features, out_features, bias, activation, dropout, dim_hook, fc_zero_init, batch_size):
        super(FC_block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.dropout = dropout
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        if fc_zero_init:
            torch.zero_(self.fc.weight.data)
        self.act = Activation(activation)
        if dropout != 0:
            self.drop = nn.Dropout(p=dropout)

        self.dim_hook = dim_hook
        if self.dim_hook is not None:
            self.NI = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            self.reset_weights()

    def reset_weights(self):
        self.NI.requires_grad = False
        #if utils.args.lambda_0:
        #    torch.nn.init.zeros_(self.NI)
        #    return
        #if utils.args.distribution == 'uniform':
        torch.nn.init.uniform_(self.NI)
        '''
        elif utils.args.distribution == 'normal':
            torch.nn.init.normal_(self.NI, mean=0.5, std=1)
            self.NI.clamp_(0, 1)
        elif utils.args.distribution == 'beta':
            dist = Beta(torch.ones_like(self.NI) * 0.5, torch.ones_like(self.NI) * 0.5)
            self.NI.data = dist.sample()
        '''
    def forward(self, x, labels, y):
        x = self.fc(x)   
        x = self.act(x)
        if self.dropout != 0:
            x = self.drop(x)

        if x.requires_grad == False:
            x.requires_grad=True
        #print('x grad_fn:', x.grad_fn, datetime.datetime.now())
        #x = Variable(x.data, requires_grad=True)
        if labels is not None and self.dim_hook is not None:
            neuromodulator_level = expectation(labels).mm(self.NI.view(-1, prod(self.NI.shape[1:]))).view(x.shape)
            x.backward(gradient=local_modulation(neuromodulator_level), retain_graph=True)

        self.out = x.detach()

        return x


spike_args = {}

class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, activation, dim_hook, batch_size):
        super(CNN_block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.act = Activation(activation)
        #if utils.args.pool == 'Avg':
        #    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        #else:
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.time_counter = 0
        self.batch_size = batch_size
        self.out_channels = out_channels

        self.dim_hook = dim_hook
        if self.dim_hook is not None:
            self.NI = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            self.reset_weights()

    def reset_weights(self):
        self.NI.requires_grad = False
        #if utils.args.lambda_0:
        #    torch.nn.init.zeros_(self.NI)
        #    return
        #if utils.args.distribution == 'uniform':
        torch.nn.init.uniform_(self.NI)
        #elif utils.args.distribution == 'normal':
        #    torch.nn.init.normal_(self.NI, mean=0.5, std=1)
        #    self.NI.clamp_(0, 1)
        #elif utils.args.distribution == 'beta':
        #    dist = Beta(torch.ones_like(self.NI) * 0.5, torch.ones_like(self.NI) * 0.5)
        #    self.NI.data = dist.sample()

    def forward(self, x, labels, y):
        x = self.conv(x)
        x = self.act(x)
        if x.requires_grad == False:
            x.requires_grad=True
        if labels is not None and self.dim_hook is not None:
            neuromodulator_level = expectation(labels).mm(self.NI.view(-1, prod(self.NI.shape[1:]))).view(x.shape)
            x.backward(gradient=local_modulation(neuromodulator_level), retain_graph=True)
        x = self.pool(x)

        return x



class NetworkBuilder(nn.Module):
    def __init__(self, args, dataset):
        super(NetworkBuilder, self).__init__()
        
        topology = 'CONV_64_5_1_2_CONV_128_5_1_2_FC_1000_FC_10' #'FC_1000_FC_10'
        input_size = 32
        input_channels = 3
        label_features = 10
        train_batch_size = 32
        dropout = 0
        fc_zero_init = False
        spike_window = 20
        randKill = 1
        conv_act = 'sigmoid'
        hidden_act = 'sigmoid'
        output_act = 'sigmoid'
        
        
        
        self.layers = nn.ModuleList()
        self.batch_size = train_batch_size
        self.spike_window = spike_window
        self.randKill = randKill
        self.device = 'cpu' #device
        spike_args['thresh'] = 0.5 # thresh
        spike_args['lens'] = 0.5 # lens
        spike_args['decay'] = 0.2 # decay

        topology = topology.split('_')
        self.topology = topology
        topology_layers = []
        num_layers = 0
        for elem in topology:
            if not any(i.isdigit() for i in elem):
                num_layers += 1
                topology_layers.append([])
            topology_layers[num_layers - 1].append(elem)
        for i in range(num_layers):
            layer = topology_layers[i]
            try:
                if layer[0] == "CONV" : # and utils.args.network == 'ANN':
                    in_channels = input_channels if (i == 0) else out_channels
                    out_channels = int(layer[1])
                    input_dim = input_size if (i == 0) else int(output_dim / 2)
                    output_dim = int((input_dim - int(layer[2]) + 2 * int(layer[4])) / int(layer[3])) + 1
                    self.layers.append(CNN_block(in_channels=in_channels, out_channels=int(layer[1]), kernel_size=int(layer[2]), stride=int(layer[3]), padding=int(layer[4]), bias=True, activation=conv_act, dim_hook=[2 * label_features, out_channels, output_dim, output_dim], batch_size=self.batch_size))
                elif layer[0] == "FC" : #and utils.args.network == 'ANN':
                    if (i == 0):
                        input_dim = input_size**2 * input_channels
                        self.conv_to_fc = 0
                    elif topology_layers[i - 1][0] == "CONV":
                        input_dim = pow(int(output_dim / 2), 2) * int(topology_layers[i - 1][1])
                        self.conv_to_fc = i
                    else:
                        input_dim = output_dim

                    output_dim = int(layer[1])
                    output_layer = (i == (num_layers - 1))
                    self.layers.append(FC_block(in_features=input_dim, out_features=output_dim, bias=False, activation=output_act if output_layer else hidden_act, dropout=dropout, dim_hook=None if output_layer else [2 * label_features, output_dim], fc_zero_init=fc_zero_init, batch_size=train_batch_size))
                elif layer[0] == "CONV" : # and utils.args.network == 'SNN':
                    in_channels = input_channels if (i == 0) else out_channels
                    out_channels = int(layer[1])
                    input_dim = input_size if (i == 0) else int(output_dim / 2)
                    output_dim = int((input_dim - int(layer[2]) + 2 * int(layer[4])) / int(layer[3])) + 1
                    self.layers.append(SpikeCNN_block(in_channels=in_channels, out_channels=int(layer[1]), kernel_size=int(layer[2]), stride=int(layer[3]), padding=int(layer[4]), bias=True, dim_hook=[2 * label_features, out_channels, output_dim, output_dim], batch_size=self.batch_size, spike_window=self.spike_window))
                elif layer[0] == "FC" : # and utils.args.network == 'SNN':
                    if (i == 0):
                        input_dim = input_size**2 * input_channels
                        self.conv_to_fc = 0
                    elif topology_layers[i - 1][0] == "CONV":
                        input_dim = pow(int(output_dim / 2), 2) * int(topology_layers[i - 1][1])
                        self.conv_to_fc = i
                    else:
                        input_dim = output_dim

                    output_dim = int(layer[1])
                    output_layer = (i == (num_layers - 1))
                    self.layers.append(SpikeFC_block(in_features=input_dim, out_features=output_dim, bias=True, dropout=dropout, dim_hook=None if output_layer else [2 * label_features, output_dim], fc_zero_init=fc_zero_init, batch_size=train_batch_size, spike_window=self.spike_window))
                else:
                    raise NameError("=== ERROR: layer construct " + str(elem) + " not supported")
            except ValueError as e:
                raise ValueError("=== ERROR: unsupported layer parameter format: " + str(e))

    def forward(self, input, labels):
        input = input.float().to(self.device)
        if len(labels.shape) < 2:
            labels = F.one_hot(labels.to(torch.int64), num_classes=10)
        '''
        if utils.args.network == 'SNN':
            for _ in range(self.spike_window):
                x = input > torch.rand(input.size()).float().to(utils.device) * self.randKill
                x = x.float()
                for i in range(len(self.layers)):
                    if i == self.conv_to_fc:
                        x = x.reshape(x.size(0), -1)
                    x = x.detach()
                    x = self.layers[i](x, labels)
            x = self.layers[-1].sumspike / self.spike_window

        elif utils.args.network == 'ANN':
        '''
        x = input.float().to(self.device)
        for i in range(len(self.layers)):
            if i == self.conv_to_fc:
                x = x.reshape(x.size(0), -1)
            x = x.detach()
            x = self.layers[i](x, labels, None)

        return x


#------------------------------------------

class TwoNN(DecoupledModel):
    feature_length = {
        "mnist": 784,
        "medmnistS": 784,
        "medmnistC": 784,
        "medmnistA": 784,
        "fmnist": 784,
        "emnist": 784,
        "femnist": 784,
        "cifar10": 3072,
        "cinic10": 3072,
        "svhn": 3072,
        "cifar100": 3072,
        "usps": 1536,
        "synthetic": DATA_SHAPE["synthetic"],
    }

    def __init__(self, dataset, args=None):
        super(TwoNN, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(self.feature_length[dataset], 1000, bias=False),
            #nn.ReLU(),
            nn.Sigmoid(),
            #nn.Linear(200, 200),
            #nn.ReLU(),
            #nn.Sigmoid(),
        )
        # self.base = nn.Linear(features_length[dataset], 200)

        self.classifier = nn.Linear(1000, NUM_CLASSES[dataset], bias=False)
        

    def need_all_features(self):
        return

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(self.base(x))
        return x

    def get_last_features(self, x, detach=True):
        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        x = torch.flatten(x, start_dim=1)
        x = self.base(x)
        return func(x)

    def get_all_features(self, x):
        raise RuntimeError("2NN has 0 Conv layer, so is unable to get all features.")


class AlexNet(DecoupledModel):
    def __init__(self, dataset):
        super().__init__()

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        alexnet = models.alexnet(
            weights=models.AlexNet_Weights.DEFAULT if pretrained else None
        )
        self.base = alexnet
        self.classifier = nn.Linear(
            alexnet.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier[-1] = nn.Identity()


class SqueezeNet(DecoupledModel):
    def __init__(self, version, dataset):
        super().__init__()

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        archs = {
            "0": (models.squeezenet1_0, models.SqueezeNet1_0_Weights.DEFAULT),
            "1": (models.squeezenet1_1, models.SqueezeNet1_1_Weights.DEFAULT),
        }
        squeezenet: models.SqueezeNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = squeezenet.features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(
                squeezenet.classifier[1].in_channels,
                NUM_CLASSES[dataset],
                kernel_size=1,
            ),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )


class DenseNet(DecoupledModel):
    def __init__(self, version, dataset):
        super().__init__()
        archs = {
            "121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
            "161": (models.densenet161, models.DenseNet161_Weights.DEFAULT),
            "169": (models.densenet169, models.DenseNet169_Weights.DEFAULT),
            "201": (models.densenet201, models.DenseNet201_Weights.DEFAULT),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        densenet: models.DenseNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = densenet
        self.classifier = nn.Linear(
            densenet.classifier.in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier = nn.Identity()


class ResNet(DecoupledModel):
    def __init__(self, version, dataset):
        super().__init__()
        archs = {
            "18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
            "152": (models.resnet152, models.ResNet152_Weights.DEFAULT),
        }

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        resnet: models.ResNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = resnet
        self.classifier = nn.Linear(self.base.fc.in_features, NUM_CLASSES[dataset])
        self.base.fc = nn.Identity()


class MobileNet(DecoupledModel):
    def __init__(self, version, dataset):
        super().__init__()
        archs = {
            "2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
            "3s": (
                models.mobilenet_v3_small,
                models.MobileNet_V3_Small_Weights.DEFAULT,
            ),
            "3l": (
                models.mobilenet_v3_large,
                models.MobileNet_V3_Large_Weights.DEFAULT,
            ),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        mobilenet = archs[version][0](weights=archs[version][1] if pretrained else None)
        self.base = mobilenet
        self.classifier = nn.Linear(
            mobilenet.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier[-1] = nn.Identity()


class EfficientNet(DecoupledModel):
    def __init__(self, version, dataset):
        super().__init__()
        archs = {
            "0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            "1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            "2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            "3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
            "4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
            "5": (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
            "6": (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT),
            "7": (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        efficientnet: models.EfficientNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = efficientnet
        self.classifier = nn.Linear(
            efficientnet.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier[-1] = nn.Identity()


class ShuffleNet(DecoupledModel):
    def __init__(self, version, dataset):
        super().__init__()
        archs = {
            "0_5": (
                models.shufflenet_v2_x0_5,
                models.ShuffleNet_V2_X0_5_Weights.DEFAULT,
            ),
            "1_0": (
                models.shufflenet_v2_x1_0,
                models.ShuffleNet_V2_X1_0_Weights.DEFAULT,
            ),
            "1_5": (
                models.shufflenet_v2_x1_5,
                models.ShuffleNet_V2_X1_5_Weights.DEFAULT,
            ),
            "2_0": (
                models.shufflenet_v2_x2_0,
                models.ShuffleNet_V2_X2_0_Weights.DEFAULT,
            ),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        shufflenet: models.ShuffleNetV2 = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = shufflenet
        self.classifier = nn.Linear(shufflenet.fc.in_features, NUM_CLASSES[dataset])
        self.base.fc = nn.Identity()


class VGG(DecoupledModel):
    def __init__(self, version, dataset):
        super().__init__()
        archs = {
            "11": (models.vgg11, models.VGG11_Weights.DEFAULT),
            "13": (models.vgg13, models.VGG13_Weights.DEFAULT),
            "16": (models.vgg16, models.VGG16_Weights.DEFAULT),
            "19": (models.vgg19, models.VGG19_Weights.DEFAULT),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        vgg: models.VGG = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = vgg
        self.classifier = nn.Linear(
            vgg.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier[-1] = nn.Identity()


# NOTE: You can build your custom model here.
# What you only need to do is define the architecture in __init__().
# Don't need to consider anything else, which are handled by DecoupledModel well already.
# Run `python *.py -m custom` to use your custom model.
class CustomModel(DecoupledModel):
    def __init__(self, dataset):
        super().__init__()
        # You need to define:
        # 1. self.base (the feature extractor part)
        # 2. self.classifier (normally the final fully connected layer)
        # The default forwarding process is: out = self.classifier(self.base(input))
        pass


MODELS = {
    "custom": CustomModel,
    "lenet5": LeNet5,
    "avgcnn": FedAvgCNN,
    "alex": AlexNet,
    "2nn": TwoNN,
    "NetworkBuilder": NetworkBuilder,
    "mlp": MLPNet,
    "squeeze0": partial(SqueezeNet, version="0"),
    "squeeze1": partial(SqueezeNet, version="1"),
    "res18": partial(ResNet, version="18"),
    "res34": partial(ResNet, version="34"),
    "res50": partial(ResNet, version="50"),
    "res101": partial(ResNet, version="101"),
    "res152": partial(ResNet, version="152"),
    "dense121": partial(DenseNet, version="121"),
    "dense161": partial(DenseNet, version="161"),
    "dense169": partial(DenseNet, version="169"),
    "dense201": partial(DenseNet, version="201"),
    "mobile2": partial(MobileNet, version="2"),
    "mobile3s": partial(MobileNet, version="3s"),
    "mobile3l": partial(MobileNet, version="3l"),
    "efficient0": partial(EfficientNet, version="0"),
    "efficient1": partial(EfficientNet, version="1"),
    "efficient2": partial(EfficientNet, version="2"),
    "efficient3": partial(EfficientNet, version="3"),
    "efficient4": partial(EfficientNet, version="4"),
    "efficient5": partial(EfficientNet, version="5"),
    "efficient6": partial(EfficientNet, version="6"),
    "efficient7": partial(EfficientNet, version="7"),
    "shuffle0_5": partial(ShuffleNet, version="0_5"),
    "shuffle1_0": partial(ShuffleNet, version="1_0"),
    "shuffle1_5": partial(ShuffleNet, version="1_5"),
    "shuffle2_0": partial(ShuffleNet, version="2_0"),
    "vgg11": partial(VGG, version="11"),
    "vgg13": partial(VGG, version="13"),
    "vgg16": partial(VGG, version="16"),
    "vgg19": partial(VGG, version="19"),
}
