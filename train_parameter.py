from utils import write2file, FCN, check_corrs
import numpy as np
import torch
import torch.nn as nn
from train_package.train2 import train_loop
from data_generator import Loader
import argparse
import os
from executor import Dispensor, RangeSampler, ChopSampler, ListSampler

def arr2str(arr):
    return ' '.join([str(x) for x in arr])

def run(bits, width, skill_cnt=5, batch_mul=200, lr=0.001,alpha=2.0, skill_bit_cnt=3, init=0.1, y_scale=3, opt='sgd', act='relu', zero_mean=True):
    load_creator = Loader(bits=bits,skill_cnt=skill_cnt,skill_bit_cnt=skill_bit_cnt, alpha=alpha, y_scale=y_scale, zero_mean= zero_mean)
    model = FCN(bits=bits, init=init, skill_cnt=skill_cnt, act=act, middle=width)
    if opt in ['sgd','',None]:
        print('sgd')
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    elif opt == 'adam':
        print('adam')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    skill_tr_loaders = []
    skill_te_loaders = []
    te_loss_arr = []
    for i in range(skill_cnt):
        tr_loader, te_loader, _, _ = load_creator.get_with_skill(skill_idx=i, train_cnt=100, test_cnt=20000,
                                                                      batch_size=20000)
        skill_tr_loaders.append(tr_loader)
        skill_te_loaders.append(te_loader)
    for epo in range(100000):
        train_loader, test_loader, _, _ = load_creator.get(train_cnt=20000, test_cnt=1000, batch_size=20000//batch_mul)
        print('main', epo)
        tr_acc,te_acc, tr_loss, te_loss = train_loop(model, train_loader, test_loader, optimizer, report=False, epochs=1, criterion=nn.MSELoss(), m=nn.Identity())
        te_loss_arr.append(te_loss)
        if epo % 10000 == 0:
            optimizer.param_groups[0]['lr'] *= 0.5
        #    check_corrs(model, skill_tr_loaders, skill_te_loaders)
    tr_acc,te_acc, tr_loss, te_loss = train_loop(model, train_loader, test_loader, optimizer, report=False, epochs=1, criterion=nn.MSELoss(),
                                                 m=nn.Identity())
    print(width, 'tr_loss: ', tr_loss)
    corrs, skill_losses = check_corrs(model, skill_tr_loaders, skill_te_loaders)
    return te_loss, skill_losses, corrs, load_creator.skill_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="outputpath", type=str, default=os.getcwd())
    parser.add_argument("-n", "--worker_cnt", help="worker_cnt", type=int, default=1)
    parser.add_argument("-i", "--iter_dict", help="do_iteration", type=int, default=0)
    parser.add_argument("-b", "--bits", help="bits", type=int, default=32)
    parser.add_argument("-p", "--opt", help="opts", type=str, default='sgd')
    parser.add_argument("-a", "--act", help="activation", type=str, default='relu')
    #parser.add_argument("-w", "--width", help="width", type=int, default=100)
    parser.add_argument("-z", "--zero_mean", help="zero_mean", type=int, default=1)
    parser.add_argument("-c", "--skill_cnt", help="skill_cnt", type=int, default=5)
    args = parser.parse_args()

    d = Dispensor(args.worker_cnt, dir=args.output, single_mode=args.worker_cnt == 1)
    d.add(ListSampler([5]), 'batch_mul')
    d.add(ListSampler([0.05]), 'lr')
    d.add(ListSampler([0.001]), 'init')
    d.add(ListSampler([5]), 'y_scale')
    d.add(ListSampler([1.6, ]), 'alpha')
    d.add(ListSampler(np.arange(2,100,2)), 'width')
    d.add(ListSampler(np.arange(5)), 'dummy')
    zero_mean_str = 'zero_' if args.zero_mean else ''

    for d_args in d:
        param_dict = {'bits': args.bits, 'skill_cnt': args.skill_cnt, 'batch_mul': 5, 'lr':0.05,'alpha':d_args['alpha'], 'init':d_args['init'],
                      'skill_bit_cnt':3, 'y_scale': 5, 'opt':args.opt, 'act':args.act}
        print('START: ', d_args['width'], param_dict)
        te_loss, skill_loss, corrs, skill_mask = run(zero_mean=args.zero_mean, width=d_args['width'], **param_dict)
        write2file('parameter', d_args['width'], skill_loss, corrs, te_loss, param_dict, zero=args.zero_mean)