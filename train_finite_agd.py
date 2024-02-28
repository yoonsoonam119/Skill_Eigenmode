import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_package.train2 import train_loop
from data_generator import Loader
import argparse
import os
from executor import Dispensor, RangeSampler, ChopSampler, ListSampler

def gelu(x): return F.gelu(x) * math.sqrt(2)
def relu(x): return F.relu(x) * math.sqrt(2)
lr = 0.5
beta = 0.0
wd = 0.0

def spectral_norm(p, u, num_steps=1):
    for _ in range(num_steps):
        u /= u.norm(dim=0, keepdim=True)
        v = torch.einsum('ab..., b... -> a...', p, u)
        u = torch.einsum('a..., ab... -> b...', v, p)
    return u.norm(dim=0, keepdim=True).sqrt(), u

class Linear(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.scale = math.sqrt(out_features / in_features)
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        torch.nn.init.orthogonal_(self.weight)
        self.weight.data *= self.scale
        self.register_buffer("momentum", torch.zeros_like(self.weight))
        self.register_buffer("u", torch.randn_like(self.weight[0]))

    def forward(self, input):
        return F.linear(input, self.weight)

    @torch.no_grad()
    def update(self, lr, beta, wd):
        self.momentum += (1 - beta) * (self.weight.grad - self.momentum)
        spec_norm, self.u = spectral_norm(self.momentum, self.u)
        self.weight -= lr * torch.nan_to_num(self.momentum / spec_norm, 0, 0, 0) * self.scale
        self.weight *= 1 - lr * wd

def train_loop(model, train_loader, test_loader, clip=False):
    tr_loss = 0
    cnts =0
    for iteration, (data, target) in enumerate(train_loader):
        # onehot = torch.nn.functional.one_hot(target, num_classes=output.shape[1]).float()
        logit = model(data)
        error = (logit - target).square()
        if iteration == 0:
            print('logit', logit[:10])
            print('target', target[:10])
            print('error', error)
        loss = error.mean()

        model.zero_grad()
        loss.backward()
        tr_loss += loss.detach().item()*len(target)
        cnts += len(target)
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        model.update(lr, beta, wd)
    tr_loss /= cnts
    print('training', tr_loss)


    te_loss = 0
    cnts =0
    with torch.no_grad():
        for iteration, (data, target) in enumerate(test_loader):
            # onehot = torch.nn.functional.one_hot(target, num_classes=output.shape[1]).float()
            if iteration == 0:
                print(logit[:10], target[:10])
            onehot = target.reshape(-1, 1)
            error = (model(data) - onehot).square()
            loss = error.mean()

            te_loss += loss.detach().item()*len(target)
            cnts += len(target)
    te_loss /= cnts
    print('test', te_loss)
    return tr_loss, te_loss

class FCN(torch.nn.Module):
    def __init__(self, bits=16, middle=1000, out=1, init=0.1, skill_cnt=5, init_fixed=False, act='relu'):
        super(FCN, self).__init__()
        output_dim = out
        width = middle
        input_dim = bits+skill_cnt
        depth = 2
        self.depth = depth
        #self.initial = Linear(input_dim, width)
        #print(torch.norm(self.initial.weight))
        self.initial = torch.nn.Linear(input_dim, width)
        #print(torch.norm(self.initial2.weight))
        self.layers = torch.nn.ModuleList([Linear(width, width) for _ in range(depth - 2)])
        #self.exit = Linear(width, output_dim)
        #print(torch.norm(self.exit.weight))
        self.exit = torch.nn.Linear(width,output_dim)
        #print(torch.norm(self.exit2.weight))
        #exit()
        with torch.no_grad():
            for p in self.parameters():
                p /= 50
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.initial(x)
        x = gelu(x)

        for layer in self.layers:
            x = layer(x)
            x = gelu(x)

        return self.exit(x).flatten()

    @torch.no_grad()
    def update(self, lr, beta, wd):
        self.opt.step()

    @torch.no_grad()
    def update2(self, lr, beta, wd):
        self.initial.update(lr / self.depth, beta, wd)

        for layer in self.layers:
            layer.update(lr / self.depth, beta, wd)

        self.exit.update(lr / self.depth, beta, wd)

def dict2str(bits, skill_cnt, batch_mul, lr,alpha, init, skill_bit_cnt, y_scale, data_cnt, opt=None, act='relu'):
    if opt in ['sgd',None]:
        opt = []
    else:
        opt = [opt]

    act = [act] if act != 'relu' else []
    return '_'.join([str(data_cnt), str(bits), str(skill_cnt), str(batch_mul), str(lr).replace('.',''), str(alpha).replace('.',''),
                     str(init).replace('.',''), str(skill_bit_cnt), str(y_scale)] + opt + act)

def check_init_vals(skill_cnt, load_creator, model):
    init_vals = []
    for i in range(skill_cnt):
        train_loader, test_loader, _, _ = load_creator.get_with_skill(skill_idx=i, train_cnt=100, test_cnt=20000,
                                                                      batch_size=20000)
        for x, y in test_loader:
            outs = model(x).detach().numpy()
            outs = (outs - np.mean(outs))/np.sqrt(20000)
            y = y.detach().numpy()
            y = (y-np.mean(y))/np.sqrt(20000)
            #print(i, np.inner(y,y))
            y /= np.linalg.norm(y)
            #print(i, np.inner(y,y))
            print(i, np.inner(outs,outs))
            #print(i, np.inner(y, outs))
            init_vals.append(np.inner(y, outs).item())
    return init_vals

def run(bits, data_cnt, skill_cnt=5, batch_mul=200, lr=0.001,alpha=2.0, skill_bit_cnt=3, init=0.1, y_scale=3, opt='sgd', act='relu'):
    load_creator = Loader(bits=bits,skill_cnt=skill_cnt,skill_bit_cnt=skill_bit_cnt, alpha=alpha, y_scale=y_scale, zero_mean=False)
    model = FCN(bits=bits, init=init, skill_cnt=skill_cnt, act=act)
    if opt in ['sgd','',None]:
        print('sgd')
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    elif opt == 'adam':
        print('adam')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    skill_losses = []
    #model.to(torch.device('cuda'))
    #train_loader, test_loader, _, _ = load_creator.get(train_cnt=200000, test_cnt=1000, batch_size=200000 // batch_mul)
    skill_tr_loaders = []
    skill_te_loaders = []
    init_vals_arr = []
    for i in range(skill_cnt):
        tr_loader, te_loader, _, _ = load_creator.get_with_skill(skill_idx=i, train_cnt=100, test_cnt=10000,
                                                                      batch_size=10000)
        skill_tr_loaders.append(tr_loader)
        skill_te_loaders.append(te_loader)
    train_loader, test_loader, _, _ = load_creator.get(train_cnt=data_cnt, test_cnt=1000, batch_size=data_cnt // batch_mul)
    for epo in range(10000):
        print('main', epo)
        train_loop(model, train_loader, test_loader)
        if epo % 1000 == 0:
            init_vals = check_init_vals(skill_cnt, load_creator, model)
            print(init_vals)
            init_vals_arr.append(init_vals)
    tr_loss, te_loss = train_loop(model, train_loader, test_loader)
    init_vals = check_init_vals(skill_cnt, load_creator, model)
    for i in range(skill_cnt):
        _, skill_loss = train_loop(model, skill_tr_loaders[i], skill_te_loaders[i])
        skill_losses.append(skill_loss)
    return te_loss, skill_losses, init_vals, load_creator.skill_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="outputpath", type=str, default=os.getcwd())
    parser.add_argument("-n", "--worker_cnt", help="worker_cnt", type=int, default=1)
    parser.add_argument("-i", "--iter_dict", help="do_iteration", type=int, default=0)
    parser.add_argument("-b", "--bits", help="bits", type=int, default=64)
    parser.add_argument("-p", "--opt", help="opts", type=str, default='adam')
    parser.add_argument("-a", "--act", help="activation", type=str, default='relu')
    args = parser.parse_args()

    d = Dispensor(args.worker_cnt, dir=args.output, single_mode=args.worker_cnt == 1)
    d.add(ListSampler([5]), 'batch_mul')
    #d.add(ListSampler([0.05,0.1, 0.01]), 'lr')
    #d.add(ListSampler([0.0001, 0.001, 0.01]), 'lr')
    d.add(ListSampler([0.01]), 'lr')
    #d.add(ListSampler([0.01, 0.05, 0.1]), 'init')
    #d.add(ListSampler([0.05]), 'init')
    d.add(ListSampler([0.05]), 'init')
    d.add(ListSampler([5]), 'y_scale')
    d.add(ListSampler([1.5]), 'alpha')
    d.add(ListSampler([10000,20000,50000,100000,200000,500000,1000000]), 'data_cnt')
    #d.add(ListSampler([10]), 'batch_mul')
    #d.add(ListSampler([0.1]), 'lr')
    #d.add(ListSampler([5]), 'y_scale')
    #d.add(ListSampler([2.0]), 'alpha')
    # d.add(ListSampler([10,50,100,500,1000]), 'batch_size')
    try_cnt = 2

    for d_args in d:
        if args.iter_dict:
            param_dict = {'bits': args.bits, 'skill_cnt': 20,
                          'batch_mul': d_args['data_cnt']//2000, 'lr': d_args['lr'],
                          'alpha': d_args['alpha'], 'init': d_args['init'], 'skill_bit_cnt':3,
                          'y_scale': d_args['y_scale'], 'opt': args.opt, 'act': args.act, 'data_cnt': d_args['data_cnt']}
        else:
            param_dict = {'bits': 64, 'skill_cnt': 20, 'batch_mul': 1000000//2000, 'lr':0.01,'alpha':1.5, 'init':0.05,
                          'skill_bit_cnt':3, 'y_scale': 5, 'data_cnt':1000000, 'opt':args.opt, 'act':args.act}
        dict_str = dict2str(**param_dict)
        te_losses_arr = []
        skill_losses_arr = []
        init_vals_arr = []
        for _ in range(try_cnt):
            #try_str = '_try' + str(d_args['try'])
            #for try_str in ['','_try_2', '_try_3', '_try_4']:
            te_loss, skill_losses, init_vals, skill_mask = run(**param_dict)
            print(skill_mask)
            print(init_vals)
            te_losses_arr.append(te_loss)
            skill_losses_arr.append(skill_losses)
            init_vals_arr.append(init_vals)
            #np.save(f'data/skill_mask_{dict_str}{try_str}', skill_mask)
            #torch.save(model.state_dict(), f'data/model_{dict_str}{try_str}')
        np.save(f'data/finite_agd_skill_losses_{dict_str}', np.stack(skill_losses_arr))
        np.save(f'data/finite_agd_init_vals_{dict_str}', np.stack(init_vals_arr))
        np.save(f'data/finite_agd_te_loss_{dict_str}', np.stack(te_losses_arr))
        if not args.iter_dict:
            exit()