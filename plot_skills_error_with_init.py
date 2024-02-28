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

def sigmoid(x, eig, init, s=1.5, lr=2e-1, batch_mul=1, a=0.001, free_p=43.0, init_val=None ,init_idxs=None):
    #c = s/a/np.power(eig,2)/200 -1
    #c = s/a -1
    if init_val is None:
        c = s/np.power(a,1.0)*1.5 -1
        print('init', a)
    else:
        c = s/np.power(init_val, 1.5)*1.5 -1
    #return 1 - 1/(1+c*np.exp(-eig*2*s*x*lr*batch_mul))
    return np.power(1 - 1/(1+c*np.exp(-eig*4*s*x*lr*batch_mul/free_p)),2)

def plot_theo_skill(xs, eigs, skills, func=sigmoid, init_vals=None, init_idxs =None):
    if init_idxs is not None:
        init_vals = init_vals[init_idxs,np.arange(len(init_vals[0]))]
    for i, (eig, skill) in enumerate(zip(eigs, skills)):
        init_val = init_vals[i] if init_vals is not None else None
        ys = func(xs, eig, skill[0], init_val = init_val)
        if init_idxs is None:
            plt.plot(xs, ys,color=f'C{i}', linestyle='dashed')
        else:
            plt.plot(xs[10*init_idxs[i]:], ys[:-10*init_idxs[i]],color=f'C{i}', linestyle='dashed')


#names = ['16_5_50_0001_20_01_3_5',
#         '16_5_200_0001_20_01_3_5',
#         '16_5_100_0001_20_01_3_5',
#         '16_5_200_0005_20_005_3_5']
names = [
         '16_5_5_005_10_005_3_3']
names = [
    '16_5_5_005_10_001_3_3',
]
#tries = np.arange(2,4)
try_str =''
#try_str =''
#rerun_str='rerun_'
rerun_str=''
take_idx = 5
for name in names:
    init_idx_all = []
    d = name2dict(name)
    skills = np.load(f'data/{rerun_str}skill_losses_{name}.npy')
    inits_all = np.load(f'data/{rerun_str}init_vals_{name}.npy')
    print(np.mean(np.abs(inits_all[:,0,:]),axis=0))
    print(np.power(d['init'],2.0)*0.2)
    exit()
    eigs = np.power(np.arange(d['skill_cnt'])+1, -d['alpha'])
    eigs /= np.sum(eigs)
    for skill_means, inits in zip(skills, inits_all):
        init_idxs = []
        for j in range(5):
            for i in range(1000):
                if np.all(inits[i:,j]>0):
                    break
            init_idxs.append(i)
        print(init_idxs)
        init_idx_all.append(init_idxs)
    init_idx_all = np.array(init_idx_all)
    means = np.mean(init_idx_all,axis=0)
    stds = np.std(init_idx_all,axis=0)
    plt.errorbar(np.arange(5)+1, means, stds)
    pow = 1.5
    plt.plot(np.arange(5)+1, np.power(1/eigs,pow)*np.power(eigs[0],pow)*means[0])
    #plt.xscale('log')
    #plt.yscale('log')
    plt.savefig(f'plot/{rerun_str}inits_{name}_FUCK')
    plt.close()
