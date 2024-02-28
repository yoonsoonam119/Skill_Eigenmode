import numpy as np
import torch
import torch.nn as nn
from train_package.train2 import train_loop
from data_generator import Loader
import argparse
import os
from executor import Dispensor, RangeSampler, ChopSampler, ListSampler

class FCN(nn.Module):
    def __init__(self, bits=16, middle=100, out=1, init=0.1, skill_cnt=5):
        super(FCN, self).__init__()
        self.bits = bits
        self.nonlin = nn.ReLU()
        self.layers = nn.ModuleList([nn.Linear(bits+skill_cnt,middle), nn.Linear(middle, middle), nn.Linear(middle,out)])
        with torch.no_grad():
            torch.nn.init.normal_(self.layers[0].weight, std=init)
            torch.nn.init.normal_(self.layers[0].bias, std=init)
            torch.nn.init.normal_(self.layers[-1].weight, std=init)
            torch.nn.init.normal_(self.layers[-1].bias, std=init)
        self.multiplier = 1

    def forward(self, x):
        x = self.nonlin(self.multiplier*self.layers[0](x))
        #x = self.nonlin(self.layers[1](x))
        return self.multiplier*self.layers[-1](x).flatten()

    def norm(self):
        with torch.no_grad():
            return torch.norm(self.layers[0].weight), torch.norm(self.layers[1].weight)

def dict2str(bits, skill_cnt, batch_mul, lr,alpha, init, skill_bit_cnt, y_scale):
    return '_'.join([str(bits), str(skill_cnt), str(batch_mul), str(lr).replace('.',''), str(alpha).replace('.',''),
                     str(init).replace('.',''), str(skill_bit_cnt), str(y_scale)])

def check_init_vals(skill_cnt, load_creator, model, optimizer):
    losses = []
    for i in range(skill_cnt):
        train_loader, test_loader, _, _ = load_creator.get_with_skill(skill_idx=i, train_cnt=100, test_cnt=20000,
                                                                      batch_size=20000)
        _, _, _, skill_loss = train_loop(model, train_loader, test_loader, optimizer, report=False, epochs=0,
                                         criterion=nn.MSELoss(), m=nn.Identity())
        losses.append(skill_loss)
    print(losses)

def check(dict_str, bits, skill_cnt=5, batch_mul=200, lr=0.001,alpha=2.0, skill_bit_cnt=3, init=0.1, y_scale=3):
    load_creator = Loader(bits=bits,skill_cnt=skill_cnt,skill_bit_cnt=skill_bit_cnt, alpha=alpha, y_scale=y_scale)
    model = FCN(bits=bits, init=init, skill_cnt=skill_cnt)
    state_dict = torch.load(f'data/model_{dict_str}')
    model.load_state_dict(state_dict)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    #check_init_vals(skill_cnt, load_creator, model, optimizer)
    norms = [[] for _ in range(skill_cnt)]
    multipliers = [1, np.sqrt(2), np.sqrt(3)]
    for m in multipliers:
        for i in range(skill_cnt):
            train_loader, test_loader, _, _ = load_creator.get_with_skill(skill_idx=i, train_cnt=10000, test_cnt=10, batch_size=10000)
            model.multiplier = m
            for x, y in train_loader:
                outs = model(x).detach().numpy()
                norm = np.linalg.norm(outs).item()
                norm2 = np.sum(np.power(outs,2)).item()
                norms[i].append(norm)
            w_norms = model.norm()
    return multipliers, norms, w_norms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="outputpath", type=str, default=os.getcwd())
    parser.add_argument("-n", "--worker_cnt", help="worker_cnt", type=int, default=1)
    parser.add_argument("-i", "--iter_dict", help="do_iteration", type=int, default=0)
    args = parser.parse_args()

    d = Dispensor(args.worker_cnt, dir=args.output, single_mode=args.worker_cnt == 1)
    d.add(ListSampler([20,50,100]), 'batch_mul')
    d.add(ListSampler([0.0001,0.001,0.01]), 'lr')
    d.add(ListSampler([0.05, 0.1, 0.2]), 'init')
    d.add(ListSampler([1, 3, 5]), 'y_scale')
    d.add(ListSampler([1.0, 1.5, 2.0]), 'alpha')
    # d.add(ListSampler([10,50,100,500,1000]), 'batch_size')

    for d_args in d:
        if args.iter_dict:
            param_dict = {'bits': 16, 'skill_cnt': 5,
                          'batch_mul': d_args['batch_mul'], 'lr': d_args['lr'],
                          'alpha': d_args['alpha'], 'init': d_args['init'], 'skill_bit_cnt':3, 'y_scale': 5}
        else:
            param_dict = {'bits': 16, 'skill_cnt': 5, 'batch_mul': 100, 'lr':0.001,'alpha':2.0, 'init':0.1, 'skill_bit_cnt':3, 'y_scale': 5}
        dict_str = dict2str(**param_dict)
        multipliers, norms, w_norms = check(dict_str=dict_str, **param_dict)
        print(norms)
        print(w_norms)
        if not args.iter_dict:
            exit()