from utils import write2file, FCN, check_corrs
import numpy as np
import torch
import torch.nn as nn
from train_package.train2 import train_loop
from data_generator import Loader
import argparse
import os
from executor import Dispensor, RangeSampler, ChopSampler, ListSampler

def run(bits, skill_cnt=5, batch_mul=200, lr=0.001,alpha=2.0, skill_bit_cnt=3, init=0.1, y_scale=3, opt='sgd', act='relu', zero_mean=True):
    load_creator = Loader(bits=bits,skill_cnt=skill_cnt,skill_bit_cnt=skill_bit_cnt, alpha=alpha, y_scale=y_scale, zero_mean=zero_mean)
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
    corrs_arr = []
    te_loss_arr = []
    for i in range(skill_cnt):
        tr_loader, te_loader, _, _ = load_creator.get_with_skill(skill_idx=i, train_cnt=100, test_cnt=20000,
                                                                      batch_size=20000)
        skill_tr_loaders.append(tr_loader)
        skill_te_loaders.append(te_loader)
    for epo in range(6000):
        train_loader, test_loader, _, _ = load_creator.get(train_cnt=20000, test_cnt=1000, batch_size=20000//batch_mul)
        print('main', epo)
        tr_acc,te_acc, tr_loss, te_loss = train_loop(model, train_loader, test_loader, optimizer, report=False, epochs=1, criterion=nn.MSELoss(), m=nn.Identity())
        if epo % 10 == 0:
            te_loss_arr.append(te_loss)
            corrs, skill_loss = check_corrs(model, skill_tr_loaders, skill_te_loaders)
            corrs_arr.append(corrs)
            skill_losses.append(skill_loss)
    return te_loss_arr, skill_losses, corrs_arr, load_creator.skill_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="outputpath", type=str, default=os.getcwd())
    parser.add_argument("-n", "--worker_cnt", help="worker_cnt", type=int, default=1)
    parser.add_argument("-i", "--iter_dict", help="do_iteration", type=int, default=0)
    parser.add_argument("-b", "--bits", help="bits", type=int, default=32)
    parser.add_argument("-p", "--opt", help="opts", type=str, default='sgd')
    parser.add_argument("-a", "--act", help="activation", type=str, default='relu')
    parser.add_argument("-z", "--zero_mean", help="zero_mean", type=int, default=1)
    args = parser.parse_args()

    d = Dispensor(args.worker_cnt, dir=args.output, single_mode=args.worker_cnt == 1)
    d.add(ListSampler([5]), 'batch_mul')
    d.add(ListSampler([0.05]), 'lr')
    d.add(ListSampler([0.001]), 'init')
    d.add(ListSampler([5]), 'y_scale')
    d.add(ListSampler([1.3, 1.6, 1.9]), 'alpha')
    try_cnt = 20
    zero_mean_str = 'zero' if args.zero_mean else ''

    for d_args in d:
        if args.iter_dict:
            param_dict = {'bits': args.bits, 'skill_cnt': 5,
                          'batch_mul': d_args['batch_mul'], 'lr': d_args['lr'],
                          'alpha': d_args['alpha'], 'init': d_args['init'], 'skill_bit_cnt':3,
                          'y_scale': d_args['y_scale'], 'opt': args.opt, 'act': args.act}
        else:
            param_dict = {'bits': 32, 'skill_cnt': 5, 'batch_mul': 5, 'lr':0.05,'alpha':1.6, 'init':0.001,
                          'skill_bit_cnt':3, 'y_scale': 5, 'opt':args.opt, 'act':args.act}
        te_losses_arr = []
        skill_losses_arr = []
        corrs_arr = []
        for _ in range(try_cnt):
            #try_str = '_try' + str(d_args['try'])
            #for try_str in ['','_try_2', '_try_3', '_try_4']:
            te_loss, skill_losses, corrs, skill_mask = run(zero_mean = args.zero_mean, **param_dict)
            print(skill_mask)
            print(corrs)
            skill_losses_arr.append(skill_losses)
            corrs_arr.append(corrs)
            te_losses_arr.append(te_loss)
            #np.save(f'data/skill_mask_{dict_str}{try_str}', skill_mask)
            #torch.save(model.state_dict(), f'data/model_{dict_str}{try_str}')
        print(np.array(corrs_arr).shape)
        print(np.array(skill_losses_arr).shape)
        print(np.array(te_losses_arr).shape)
        write2file('time', None, skill_losses_arr, corrs_arr, te_losses_arr, param_dict, zero=args.zero_mean)
        if not args.iter_dict:
            exit()