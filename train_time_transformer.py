import os
from utils_transformer import write2file, FCN, check_corrs, train_loop
import numpy as np
import torch
import torch.nn as nn
from data_generator import Loader
import argparse
import math
from executor import Dispenser, RangeSampler, ChopSampler, ListSampler
from transformer_model import GPT
import copy, datetime

current_time = datetime.datetime.now().time()
time_string = current_time.strftime("_%H_%M_%S_%f")
filename = time_string

def run(args, bits, data_cnt, skill_cnt=5, batch_mul=200, lr=0.001,alpha=2.0, skill_bit_cnt=3, init=0.1, y_scale=3, opt='sgd', act='relu', zero_mean=True):
    load_creator = Loader(bits=bits,skill_cnt=skill_cnt,skill_bit_cnt=skill_bit_cnt, alpha=alpha, y_scale=y_scale, zero_mean=zero_mean)
    
    model = GPT(n_head=args.heads, n_embd=int(bits+skill_cnt+2), n_layer=args.depth, context_length=int(bits+skill_cnt), vocab_size=1, n_skills=skill_cnt, ex=args.ex)
    if args.init != 1.0:
        for name, p in model.named_parameters():
            with torch.no_grad():
                p*=args.init

    if args.opt in ['sgd','',None]:
        print('sgd')
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.beta, weight_decay=args.wd)
    elif args.opt == 'adam':
        print('adam')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = args.opt
    skill_tr_loaders = []
    skill_te_loaders = []
    corrs_arr = []
    te_loss_arr = []
    skill_losses = []
    for i in range(skill_cnt):
        tr_loader, te_loader, _, _ = load_creator.get_with_skill(skill_idx=i, train_cnt=100, test_cnt=20000,
                                                                      batch_size=20000)
        skill_tr_loaders.append(tr_loader)
        skill_te_loaders.append(te_loader)
    train_loader, test_loader, _, _ = load_creator.get(train_cnt=data_cnt, test_cnt=1000, batch_size=int(4*data_cnt // batch_mul))
    model = model.to(device=args.device)
    for epo in range(300000):
        train_loader, test_loader, _, _ = load_creator.get(train_cnt=20000, test_cnt=5000, batch_size=int(args.batchsize))
        print('main', epo)
        criterion = nn.MSELoss()
        tr_acc,te_acc, tr_loss, te_loss = train_loop(args, model, train_loader, test_loader, optimizer, report=False, epochs=1, criterion=criterion, m=nn.Identity())      
        if epo % 100 == 0:
            check_corr = True #True
            if check_corr:
                te_loss_arr.append(te_loss)
                with torch.no_grad():
                    model_ = copy.deepcopy(model)
                    corrs, skill_loss = check_corrs(args, model_, skill_tr_loaders, skill_te_loaders)
                corrs_arr.append(corrs)
                skill_losses.append(skill_loss)
                write2file(f'data_{epo}', 1, [skill_losses], [corrs_arr], [te_loss_arr], param_dict, zero=args.zero_mean, filename=filename)


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
    parser.add_argument("-lr", "--lr", help="zero_mean", type=float, default=0.00001)
    parser.add_argument("-beta", "--beta", help="zero_mean", type=float, default=0.0)
    parser.add_argument("-wd", "--wd", help="zero_mean", type=float, default=0.0)
    parser.add_argument("-init", "--init", help="zero_mean", type=float, default=1.0)
    parser.add_argument("-model", "--model", help="zero_mean", type=str, default='transformer')
    parser.add_argument("-batchsize", "--batchsize", help="zero_mean", type=int, default=5000)
    parser.add_argument("-numskills", "--numskills", help="zero_mean", type=int, default=8)
    parser.add_argument("-numbits", "--numbits", help="zero_mean", type=int, default=32)
    parser.add_argument("-heads", "--heads", help="zero_mean", type=int, default=4)
    parser.add_argument("-ex", "--ex", help="zero_mean", type=int, default=512)
    parser.add_argument("-d", "--depth", help="zero_mean", type=int, default=1)
    parser.add_argument("-y", "--y_scale", help="zero_mean", type=int, default=1)
    parser.add_argument("-alpha", "--alpha", help="zero_mean", type=float, default=1.6)

    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(args)

    d = Dispenser(args.worker_cnt, dir=args.output, single_mode=args.worker_cnt == 1)
    d.add(ListSampler([5]), 'batch_mul')
    d.add(ListSampler([args.lr]), 'lr')
    d.add(ListSampler([0.001]), 'init')
    d.add(ListSampler([1]), 'y_scale')
    d.add(ListSampler([1.9]), 'alpha')

    if args.iter_dict:
        data_list = list(np.arange(500,10500,500)) + list(np.arange(10000,40000,5000))
    else:
        data_list = [2000,4000]

    data_list = [int(1e4)]
    print(data_list)


    d.add(ListSampler(data_list), 'data_cnt')
    d.add(ListSampler(np.arange(5)), 'try')
    zero_mean_str = 'zero_' if args.zero_mean else ''

    bits, skill_cnt = args.numbits, args.numskills

    for d_args in d:
        print(d_args)
        param_dict = {'bits': args.numbits, 'skill_cnt': args.numskills, 'batch_mul': max(1, 50), 'lr':args.lr,'alpha':args.alpha, 'init':args.init,
                        'skill_bit_cnt':3, 'y_scale': args.y_scale, 'opt':args.opt, 'act':args.act}
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
