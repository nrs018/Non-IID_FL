import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm
import argparse
args = None
epoch = []

delta_bias = {
    "mnist_classIL": 0.2,
    "cifar_classIL": 0.1,
    "gesture_classIL": 0.2,
    "alphabet_classIL": 0.2,
    "mathgreek_classIL": None
}

bias = {
    "mnist_classIL": None,
    "cifar_classIL": None,
    "gesture_classIL": None,
    "alphabet_classIL": None,
    "mathgreek_classIL": 0.8
}

testing_result = []
def print_model_report(model):
    print('-' * 100)
    print(model)
    print('Dimensions =', end=' ')
    count = 0
    for p in model.parameters():
        print(p.size(), end=' ')
        count += np.prod(p.size())
    print()
    print('Num parameters = %s' % (human_format(count)))
    print('-' * 100)
    return count


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim, '=', end=' ')
        opt = optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n + ':', opt[n], end=', ')
        print()
    return


def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean = 0
    std = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean += image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded = mean.view(mean.size(0), mean.size(1), 1, 1).expand_as(image)
    for image, _ in loader:
        std += (image - mean_expanded).pow(2).sum(3).sum(2)

    std = (std / (len(dataset) * image.size(2) * image.size(3) - 1)).sqrt()

    return mean, std


def fisher_matrix_diag(t, x, y, model, criterion, sbatch=20):
    # Init
    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = 0 * p.data
    # Compute
    model.train()
    for i in tqdm(range(0, x.size(0), sbatch), desc='Fisher diagonal', ncols=100, ascii=True):
        b = torch.LongTensor(np.arange(i, np.min([i + sbatch, x.size(0)]))).cpu()
        images = torch.autograd.Variable(x[b], volatile=False)
        target = torch.autograd.Variable(y[b], volatile=False)
        # Forward and backward
        model.zero_grad()
        outputs = model.forward(images, t)
        if args.multi_output:
            output = outputs[t]
        else:
            output = outputs
        loss = criterion(t, output, target)
        loss.backward()
        # Get gradients
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += sbatch * p.grad.data.pow(2)
    # Mean
    for n, _ in model.named_parameters():
        fisher[n] = fisher[n] / x.size(0)
        fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)
    return fisher


def cross_entropy(outputs, targets, exp=1, size_average=True, eps=1e-5):
    out = torch.nn.functional.softmax(outputs)
    tar = torch.nn.functional.softmax(targets)
    if exp != 1:
        out = out.pow(exp)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar * out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce


def set_req_grad(layer, req_grad):
    if hasattr(layer, 'weight'):
        layer.weight.requires_grad = req_grad
    if hasattr(layer, 'bias'):
        layer.bias.requires_grad = req_grad
    return


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
    
def parse_add():
    parser = argparse.ArgumentParser(description='')
    common_arg = parser.add_argument_group('common_arg', 'common args description')
    # Common parameters for all methods
    common_arg.add_argument('--method', type=str, default='fednaca')
    common_arg.add_argument('--mode', type=str, default='serial')
    common_arg.add_argument('--dataset', type=str, default='mnist')
    common_arg.add_argument('--experiment', type=str, default='mnist_classIL')
    common_arg.add_argument('--seed', type=int, default=42)
    common_arg.add_argument('--model', type=str, default='mlp')
    common_arg.add_argument('--join_ratio', type=float, default=0.2)
    common_arg.add_argument('--global_epoch', type=int, default=1)
    common_arg.add_argument('--local_epoch', type=int, default=5)
    common_arg.add_argument('--finetune_epoch', type=int, default=0)
    common_arg.add_argument('--batch_size', type=int, default=32)
    common_arg.add_argument('--test_interval', type=int, default=100)
    common_arg.add_argument('--straggler_ratio', type=float, default=0)
    common_arg.add_argument('--straggler_min_local_epoch', type=int, default=0)
    common_arg.add_argument('--buffers', type=str, default='local')
    common_arg.add_argument('--external_model_params_file', type=str, default=None)
    common_arg.add_argument('--lambda_inv', type=float, default=0.5)
    common_arg.add_argument('--theta_max', type=float, default=1.2)
    
    optim = parser.add_argument_group('optim', 'optimizer description')
    optim.add_argument('--lr', type=float, default=0.01)
    optim.add_argument('--dampening', type=float, default=0)
    optim.add_argument('--weight_decay', type=float, default=0)
    optim.add_argument('--momentum', type=float, default=0)
    optim.add_argument('--nesterov', type=bool, default=False)
    optim.add_argument('--optimizer', type=str, default='sgd')
    
    
    common_arg.add_argument('--eval_test', type=bool, default=True)
    common_arg.add_argument('--eval_val', type=bool, default=False)
    common_arg.add_argument('--eval_train', type=bool, default=False)
    common_arg.add_argument('--verbose_gap', type=int, default=10)
    common_arg.add_argument('--visible', type=bool, default=False)
    common_arg.add_argument('--use_cuda', type=bool, default=True)
    common_arg.add_argument('--save_log', type=bool, default=True)
    common_arg.add_argument('--save_model', type=bool, default=False)
    common_arg.add_argument('--save_fig', type=bool, default=True)
    common_arg.add_argument('--save_metrics', type=bool, default=True)
    common_arg.add_argument('--delete_useless_run', type=bool, default=True)
    common_arg.add_argument('--ray_cluster_addr', type=str, default='null')
    common_arg.add_argument('--num_gpus', type=int, default=0)
    common_arg.add_argument('--num_workers', type=int, default=2)
    common_arg.add_argument('--mu', type=float, default=1.0)
    common_arg.add_argument('--client_num', type=int, default=10)
    common_arg.add_argument('--test_ratio', type=float, default=0.25)
    common_arg.add_argument('--val_ratio', type=float, default=0)
    common_arg.add_argument('--split', type=str, default='sample')
    common_arg.add_argument('--IID_ratio', type=float, default=0.0)
    common_arg.add_argument('--monitor_window_name_suffix', type=str, default='mnist-10clients-0%IID-Dir(0.1)-seed42')
    common_arg.add_argument('--alpha', type=float, default=0.1)
    common_arg.add_argument('--least_samples', type=int, default=40)
    common_arg.add_argument('--optim',type=str, default=None)
    args = parser.parse_args()
    return args

