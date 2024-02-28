import numpy as np
import torch
import torch.nn as nn
from train_package.train2 import train_loop
from data_generator import Loader
import argparse
import os
from executor import Dispensor, RangeSampler, ChopSampler, ListSampler

class FCN(nn.Module):
    def __init__(self, bits=16, middle=1000, out=1, init=0.1, skill_cnt=5, init_fixed=False, act='relu'):
        super(FCN, self).__init__()
        self.bits = bits
        if act == 'relu':
            self.nonlin = nn.ReLU()
        elif act == 'tanh':
            self.nonlin = nn.Tanh()
        self.layers = nn.ModuleList([nn.Linear(bits+skill_cnt,middle), nn.Linear(middle, middle), nn.Linear(middle,out)])
        with torch.no_grad():
            if init_fixed:
                torch.nn.init.constant_(self.layers[0].weight, init)
                torch.nn.init.constant_(self.layers[0].bias, init)
                torch.nn.init.constant_(self.layers[-1].weight, init)
                torch.nn.init.constant_(self.layers[-1].bias, init)
            else:
                torch.nn.init.normal_(self.layers[0].weight, std=init)
                torch.nn.init.normal_(self.layers[0].bias, std=init)
                torch.nn.init.normal_(self.layers[-1].weight, std=init)
                torch.nn.init.normal_(self.layers[-1].bias, std=init)

    def forward(self, x):
        x = self.nonlin(self.layers[0](x))
        #x = self.nonlin(self.layers[1](x))
        return self.layers[-1](x).flatten()

def dict2str(bits, skill_cnt, batch_mul, lr,alpha, init, skill_bit_cnt, y_scale, opt=None, act='relu'):
    if opt in ['sgd',None]:
        opt = []
    else:
        opt = [opt]

    act = [act] if act != 'relu' else []
    return '_'.join([str(bits), str(skill_cnt), str(batch_mul), str(lr).replace('.',''), str(alpha).replace('.',''),
                     str(init).replace('.',''), str(skill_bit_cnt), str(y_scale)] + opt + act)

def check_init_vals(skill_cnt, load_creator, model, optimizer):
    init_vals = []
    for i in range(skill_cnt):
        train_loader, test_loader, _, _ = load_creator.get_with_skill(skill_idx=i, train_cnt=100, test_cnt=20000,
                                                                      batch_size=20000)
        for x, y in test_loader:
            outs = model(x).detach().numpy()
            outs = (outs - np.mean(outs))/np.sqrt(20000)
            y = y.detach().numpy()
            y = (y-np.mean(y))/np.sqrt(20000)
            print(i, np.inner(y,y))
            y /= np.linalg.norm(y)
            print(i, np.inner(y,y))
            print(i, np.inner(outs,outs))
            print(i, np.inner(y, outs))
            init_vals.append(np.inner(y, outs).item())
        #_, _, _, skill_loss = train_loop(model, train_loader, test_loader, optimizer, report=False, epochs=0,
        #                                 criterion=nn.MSELoss(), m=nn.Identity())
        #losses.append(skill_loss)
    #train_loader, test_loader, _, _ = load_creator.get(train_cnt=100, test_cnt=20000, batch_size=20000)
    return init_vals

def run(bits, skill_idx, skill_cnt=5, batch_mul=200, lr=0.001,alpha=2.0, skill_bit_cnt=3, init=0.1, y_scale=3, opt='sgd', act='relu'):
    load_creator = Loader(bits=bits,skill_cnt=skill_cnt,skill_bit_cnt=skill_bit_cnt, alpha=alpha, y_scale=y_scale,
                          one_skill_idx=skill_idx)
    model = FCN(bits=bits, init=init, skill_cnt=skill_cnt, act=act)
    if opt == 'sgd':
        print('sgd')
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    elif opt == 'adam':
        print('adam')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    skill_losses = [[] for _ in range(skill_cnt)]
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
    for epo in range(1000):
        train_loader, test_loader, _, _ = load_creator.get(train_cnt=20000, test_cnt=1000, batch_size=20000//batch_mul)
        print('main', epo)
        train_loop(model, train_loader, test_loader, optimizer, report=False, epochs=1, criterion=nn.MSELoss(), m=nn.Identity())
        if epo % 10 == 0:
            init_vals = check_init_vals(skill_cnt, load_creator, model, optimizer)
            print(init_vals)
            init_vals_arr.append(init_vals)
        for i in range(skill_cnt):
            #train_loader, test_loader, _, _ = load_creator.get_with_skill(skill_idx=i, train_cnt=100, test_cnt=10000, batch_size=10000)
            #_, _, _, skill_loss = train_loop(model, train_loader, test_loader, optimizer, report=False, epochs=0, criterion=nn.MSELoss(), m=nn.Identity())
            _, _, _, skill_loss = train_loop(model, skill_tr_loaders[i], skill_te_loaders[i], optimizer, report=False, epochs=0, criterion=nn.MSELoss(), m=nn.Identity())
            skill_losses[i].append(skill_loss)
    return skill_losses, model, init_vals_arr, load_creator.skill_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="outputpath", type=str, default=os.getcwd())
    parser.add_argument("-n", "--worker_cnt", help="worker_cnt", type=int, default=1)
    parser.add_argument("-i", "--iter_dict", help="do_iteration", type=int, default=0)
    parser.add_argument("-b", "--bits", help="bits", type=int, default=16)
    parser.add_argument("-p", "--opt", help="opts", type=str, default='sgd')
    parser.add_argument("-a", "--act", help="activation", type=str, default='relu')
    parser.add_argument("-s", "--skill_idx", help="skill index", type=int, default=3)
    args = parser.parse_args()

    d = Dispensor(args.worker_cnt, dir=args.output, single_mode=args.worker_cnt == 1)
    d.add(ListSampler([1]), 'batch_mul')
    d.add(ListSampler([0.05,0.1, 0.01]), 'lr')
    #d.add(ListSampler([0.01, 0.05, 0.1]), 'init')
    d.add(ListSampler([1, 3, 5]), 'y_scale')
    d.add(ListSampler([1.0, 1.5, 2.0]), 'alpha')
    #d.add(ListSampler([10]), 'batch_mul')
    #d.add(ListSampler([0.1]), 'lr')
    d.add(ListSampler([0.05]), 'init')
    #d.add(ListSampler([5]), 'y_scale')
    #d.add(ListSampler([2.0]), 'alpha')
    # d.add(ListSampler([10,50,100,500,1000]), 'batch_size')
    try_cnt = 3

    for d_args in d:
        if args.iter_dict:
            param_dict = {'bits': args.bits, 'skill_cnt': 5,
                          'batch_mul': d_args['batch_mul'], 'lr': d_args['lr'],
                          'alpha': d_args['alpha'], 'init': d_args['init'], 'skill_bit_cnt':3,
                          'y_scale': d_args['y_scale'], 'opt': args.opt, 'act': args.act}
        else:
            param_dict = {'bits': 16, 'skill_cnt': 5, 'batch_mul': 5, 'lr':0.05,'alpha':1.5, 'init':0.05,
                          'skill_bit_cnt':3, 'y_scale': 5, 'opt':args.opt, 'act':args.act}
        dict_str = dict2str(**param_dict)
        skill_losses_arr = []
        init_vals_arr = []
        for _ in range(try_cnt):
            #try_str = '_try' + str(d_args['try'])
            #for try_str in ['','_try_2', '_try_3', '_try_4']:
            skill_losses, model, init_vals, skill_mask = run(skill_idx=args.skill_idx, **param_dict)
            skill_losses_arr.append(skill_losses)
            init_vals_arr.append(init_vals)
            #np.save(f'data/skill_mask_{dict_str}{try_str}', skill_mask)
            #torch.save(model.state_dict(), f'data/model_{dict_str}{try_str}')
        np.save(f'data/one{args.skill_idx}_skill_losses_{dict_str}', np.stack(skill_losses_arr))
        np.save(f'data/one{args.skill_idx}_init_vals_{dict_str}', np.stack(init_vals_arr))
        if not args.iter_dict:
            exit()