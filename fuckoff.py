import sys
import os

try:
    sys.path.remove('/usr/local/shared/python/3.6.1/lib/python3.4')
    sys.path.remove('/usr/local/lib/python3.5/dist-packages')
except:
    pass

from utils_agd import write2file, FCN, check_corrs, train_loop
import numpy as np
import torch
import torch.nn as nn
from data_generator import Loader
import argparse
import os, math
from executor import Dispenser, RangeSampler, ChopSampler, ListSampler
from models_agd import MLP, ResMLP
from transformer_model import GPT

def mse_agd(logit, y):
    error = (logit - y).square()
    loss = error.mean()
    return loss

def run(args, bits, data_cnt, skill_cnt=5, batch_mul=200, lr=0.001,alpha=2.0, skill_bit_cnt=3, init=0.1, y_scale=3, opt='sgd', act='relu', zero_mean=True):
    load_creator = Loader(bits=bits,skill_cnt=skill_cnt,skill_bit_cnt=skill_bit_cnt, alpha=alpha, y_scale=y_scale, zero_mean=zero_mean)
    if args.model in ['FCN', 'MLP']:
        if args.opt != 'agd':
            model = FCN(bits=bits, init=init, skill_cnt=skill_cnt, act=act)
        else:
            model = MLP(input_dim=int(bits+skill_cnt), width = 1000, depth = 2, output_dim=1, init_scale=init)
    elif args.model == 'RESMLP':
        model = ResMLP(input_dim=int(bits+skill_cnt), output_dim=1, num_blocks=2, block_depth=2, width=1000)
    elif args.model == 'transformer':
        model = GPT(n_head=2, n_embd=int(bits+skill_cnt+2), n_layer=1, context_length=int(bits+skill_cnt), vocab_size=1)
        print('heh',model)
    if args.opt in ['sgd','',None]:
        print('sgd')
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    elif args.opt == 'adam':
        print('adam')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = args.opt
    skill_tr_loaders = []
    skill_te_loaders = []
    for i in range(skill_cnt):
        tr_loader, te_loader, _, _ = load_creator.get_with_skill(skill_idx=i, train_cnt=100, test_cnt=20000,
                                                                      batch_size=20000)
        skill_tr_loaders.append(tr_loader)
        skill_te_loaders.append(te_loader)
    train_loader, test_loader, _, _ = load_creator.get(train_cnt=data_cnt, test_cnt=1000, batch_size=int(4*data_cnt // batch_mul))
    model = model.to(device=args.device)
    for epo in range(1000000):
        train_loader, test_loader, _, _ = load_creator.get(train_cnt=20000, test_cnt=1000, batch_size=int(args.batchsize))
        print('main', epo)
        criterion = nn.MSELoss() if args.opt != 'agd' else mse_agd
        # if epo > 20000:
        #     print('sgd1')
        #     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr / 10, momentum=0.0, weight_decay=1e-5)
        # else:
        #     print('sgd2')
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0, weight_decay=1e-5)
        train_loop(args, model, train_loader, test_loader, optimizer, report=False, epochs=1, criterion=criterion, m=nn.Identity())
        if epo % 10000 == 0:
            if args.opt != 'agd':
                optimizer.param_groups[0]['lr'] *= 0.5
            check_corr = False
            if check_corr:
                check_corrs(model, skill_tr_loaders, skill_te_loaders)
            continue
    tr_acc,te_acc, tr_loss, te_loss = train_loop(args, model, train_loader, test_loader, optimizer, report=False, epochs=1, criterion=criterion,
               m=nn.Identity())
    print(data_cnt, 'tr_loss: ', tr_loss)
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
    parser.add_argument("-z", "--zero_mean", help="zero_mean", type=int, default=1)
    parser.add_argument("-lr", "--lr", help="zero_mean", type=float, default=0.001)
    parser.add_argument("-beta", "--beta", help="zero_mean", type=float, default=0.0)
    parser.add_argument("-wd", "--wd", help="zero_mean", type=float, default=0.0001)
    parser.add_argument("-init", "--init", help="zero_mean", type=float, default=1.0)
    parser.add_argument("-model", "--model", help="zero_mean", type=str, default='transformer')
    parser.add_argument("-batchsize", "--batchsize", help="zero_mean", type=int, default=2000)
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    d = Dispenser(args.worker_cnt, dir=args.output, single_mode=args.worker_cnt == 1)
    d.add(ListSampler([5]), 'batch_mul')
    d.add(ListSampler([args.lr]), 'lr')
    d.add(ListSampler([0.001]), 'init')
    d.add(ListSampler([5]), 'y_scale')
    d.add(ListSampler([1.6]), 'alpha')

    if args.iter_dict:
        data_list = list(np.arange(500,10500,500)) + list(np.arange(10000,40000,5000))
    else:
        data_list = [2000,4000]

    data_list = [int(1e4)]
    print(data_list)


    d.add(ListSampler(data_list), 'data_cnt')
    d.add(ListSampler(np.arange(5)), 'try')
    zero_mean_str = 'zero_' if args.zero_mean else ''

    bits, skill_cnt = 32, 6
    # bits, skill_cnt = 128, 32
    # bits, skill_cnt = 256, 80

    # bs = max(1, d_args['data_cnt']//2000)

    for d_args in d:
        print(d_args)
        if False: #args.iter_dict:
            param_dict = {'bits': args.bits, 'skill_cnt': 5*int(args.bits//16),
                          'batch_mul': max(1, d_args['data_cnt']//2000), 'lr': d_args['lr'],
                          'alpha': d_args['alpha'], 'init': d_args['init'], 'skill_bit_cnt':3,
                          'y_scale': d_args['y_scale'], 'opt': args.opt, 'act': args.act}
        else:
            param_dict = {'bits': bits, 'skill_cnt': skill_cnt, 'batch_mul': max(1, 50), 'lr':args.lr,'alpha':1.5, 'init':args.init,
                          'skill_bit_cnt':3, 'y_scale': 5, 'opt':args.opt, 'act':args.act}
        te_loss, skill_loss, corrs, skill_mask = run(args, zero_mean = args.zero_mean, data_cnt=d_args['data_cnt'], **param_dict)
        write2file('data', d_args['data_cnt'], skill_loss, corrs, te_loss, param_dict, zero=args.zero_mean)
        '''    
            te_losses_arr.append(te_loss)
            skill_losses_arr.append(skill_losses)
            corrs_arr.append(corrs)
            #np.save(f'data/skill_mask_{dict_str}{try_str}', skill_mask)
            #torch.save(model.state_dict(), f'data/model_{dict_str}{try_str}')
        print(dict_str)
        np.save(f'data/{zero_mean_str}finite_skill_losses_{dict_str}', np.stack(skill_losses_arr))
        np.save(f'data/{zero_mean_str}finite_corrs_{dict_str}', np.stack(corrs_arr))
        np.save(f'data/{zero_mean_str}finite_te_loss_{dict_str}', np.stack(te_losses_arr))
        '''
        #if not args.iter_dict:
        #    exit()