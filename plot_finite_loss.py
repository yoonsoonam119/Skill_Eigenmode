import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import os
import copy
from save_load import YamlReader
from executor import Dispensor, RangeSampler, ChopSampler, ListSampler
from functools import partial
from collections import OrderedDict
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rc('font', **{'size':11})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def exp_format(cka):
    s = f"{cka: .1e}"
    return s[:-2] + s[-1]

def latex_transform(title, i = 0, alp=True):
    alphabet = list(string.ascii_lowercase)
    title = ['$\mathrm{' + s + '}$' for s in title.split()]
    if alp:
        title = " ".join(['$\mathrm{(' +alphabet[i] +')}$'] + title)
    else:
        roman = ['i', 'ii', 'iii', 'iv', 'v', 'vi']
        title = " ".join(['$\mathrm{(' + roman[i] + ')}$'] + title)
    print(title)
    return rf'{title}'

def cumul(arr):
    return np.array([np.sum(arr[:i]) for i in range(1,len(arr)+1)])
def cumul_std(arr):
    return np.array([np.sqrt(np.sum(np.power(arr[:i],2))) for i in range(1,len(arr)+1)])
def trunc(values, decs=3):
    return np.trunc(values*10**decs)/(10**decs)


def name2dict(string):
    d = {}
    args = string.rstrip('.npy').split('_')
    d['bits'] = int(args[0])
    d['skill_cnt'] = int(args[1])
    d['batch_mul'] = int(args[2])
    d['lr'] = float(args[3][0] + '.' + args[3][1:])
    d['alpha'] = float(args[4][0] + '.' + args[4][1:])
    d['init'] = float(args[5][0] + '.' + args[5][1:])
    d['skill_bit_cnt'] = int(args[6])
    d['y_scale'] = int(args[7])
    print(d)
    return d

def sigmoid_old(x, eig, init, s=1.5, lr=2e-1, batch_mul=1, a=0.001):
    c = 10/a -1
    #return 1 - 1/(1+c*np.exp(-eig*2*s*x*lr*batch_mul))
    return 1 - 1/(1+c*np.exp(-eig*s*x*lr*batch_mul))

def sigmoid(x, eig, init, s=1.5, lr=2e-1, batch_mul=1, a=0.001, free_p=40.0, init_val=None ,init_idxs=None):
    #c = s/a/np.power(eig,2)/200 -1
    #c = s/a -1
    if init_val is None:
        #c = s/np.power(a,1.0)*1.5 -1
        c = s/np.power(a,1.0)*5/3 -1
        #c = s/np.power(a,1.0)*0.3 -1 this is for the inner product at init?
        print('init', a)
        #free_p = 35
        free_p = 200
    else:
        c = s/np.clip(init_val, 1e-8, None)/2 -1
    #return 1 - 1/(1+c*np.exp(-eig*2*s*x*lr*batch_mul))
    return 1/(1+c*np.exp(-eig*4*s*x*lr*batch_mul*10/free_p))

def plot_theo_inners(xs, eigs, inners, func=sigmoid, init_vals=None, init_idxs =None, multi=1, style='dotted'):
    for i, (eig, inner) in enumerate(zip(eigs, inners)):
        init_val = init_vals[i][init_idxs[i]] if init_vals is not None else None
        ys = func(xs, eig, inner[0], init_val = init_val)
        if init_idxs is None:
            plt.plot(xs, multi*ys,color=f'C{i}', linestyle=style)
        else:
            plt.plot(xs[init_idxs[i]:], multi*ys[:-init_idxs[i]],color=f'C{i}', linestyle=style)

        plt.xlabel(r'$epochs*10$', fontdict={'size':15})
        plt.ylabel(r'$\theta^2/\theta^2_f$', fontdict={'size':15})

data_cnts = [1000,2000,5000,10000,20000,50000,100000]
name = '64_20_5_001_15_005_3_5'

try_str =''
rerun_str='finite_'
opt_str = '_adam'
for data_cnt in data_cnts:
    skills = np.load(f'data/{rerun_str}skill_losses_{data_cnt}_{name}{opt_str}.npy')
    print(skills.shape)
    skill_means = np.mean(skills, axis=0)
    skill_stds = np.std(skills, axis=0)
    inits_all = np.load(f'data/{rerun_str}init_vals_{data_cnt}_{name}{opt_str}.npy')
    #print(inits_all)
    #exit()
    inits_means = inits_all
    inits_stds = np.zeros_like(inits_all)
    #inits_means = np.mean(inits_all,axis=0).T
    inits_stds = np.std(inits_all,axis=0).T
    xs = np.arange(len(inits_means[0]))
    for i, (init_mean, init_std) in enumerate(zip(inits_means, inits_stds)):
        plt.plot(xs, np.abs(init_mean), color=f'C{i}')
        #plt.errorbar(xs, init_mean/d['y_scale']*2, init_std/d['y_scale']*2, color=f'C{i}')
        continue
    plt.plot(np.array(xs)+1, np.power(np.array(xs)+1,-1.5))
    plt.xscale('log')
    plt.yscale('log')

    plt.savefig(f'plot/finite_inners_{rerun_str}skills_{data_cnt}_{name}{opt_str}')
    plt.savefig(f'plot/finite_inners_{rerun_str}skills_{data_cnt}_{name}{opt_str}.pdf', format='pdf', dpi=300)
    plt.close()
