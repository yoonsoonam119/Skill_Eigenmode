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

def sigmoid(x, eig, init, s=1.5, lr=2e-1, batch_mul=1, a=0.001, free_p=33.0, init_val=None ,init_idxs=None):
    #c = s/a/np.power(eig,2)/200 -1
    #c = s/a -1
    if init_val is None:
        #c = s/np.power(a,1.0)*1.5 -1
        c = s/np.power(a,1.0)*5/3 -1
        print('init', a)
    else:
        c = s/np.abs(init_val) -1
    #return 1 - 1/(1+c*np.exp(-eig*2*s*x*lr*batch_mul))
    return np.power(1 - 1/(1+c*np.exp(-eig*4*s*x*lr*batch_mul/free_p)),2)

def plot_theo_skill(xs, eigs, skills, func=sigmoid, init_vals=None, init_idxs =None, multi=1):
    if init_idxs is not None:
        init_vals = init_vals[init_idxs,np.arange(len(init_vals[0]))]
    for i, (eig, skill) in enumerate(zip(eigs, skills)):
        init_val = init_vals[i] if init_vals is not None else None
        ys = func(xs, eig, skill[0], init_val = init_val)
        if init_idxs is None:
            plt.plot(xs, multi*ys,color=f'C{i}', linestyle='dashed')
        else:
            plt.plot(xs[10*init_idxs[i]:], multi*ys[:-10*init_idxs[i]],color=f'C{i}', linestyle='dashed')


#names = ['16_5_50_0001_20_01_3_5',
#         '16_5_200_0001_20_01_3_5',
#         '16_5_100_0001_20_01_3_5',
#         '16_5_200_0005_20_005_3_5']
names = [
         '16_5_5_005_10_005_3_3']
names = [
    '16_5_5_005_15_001_3_5',
    '16_5_5_005_20_005_3_5',
    '16_5_5_005_20_001_3_5',
    '16_5_5_005_15_005_3_5',
    '16_5_5_005_15_005_3_3',
    '16_5_5_005_10_005_3_3',
    '16_5_5_005_10_005_3_1',
    '16_5_5_005_15_001_3_3',
    '16_5_5_005_20_005_3_3',
    '16_5_5_005_10_01_3_5',
    '16_5_5_005_20_01_3_3',
    '16_5_5_005_20_01_3_1',
    '16_5_5_01_15_001_3_5',
    '16_5_5_01_20_005_3_5',
    '16_5_5_01_20_001_3_5',
    '16_5_5_01_15_005_3_5',
    '16_5_5_01_15_005_3_3',
    '16_5_5_01_10_005_3_3',
    '16_5_5_01_10_005_3_1',
    '16_5_5_01_15_001_3_3',
    '16_5_5_01_20_005_3_3',
    '16_5_5_01_10_01_3_5',
    '16_5_5_01_20_01_3_3',
    '16_5_5_01_20_01_3_1',
    '16_5_5_005_15_01_3_5',
]
#tries = np.arange(2,4)
try_str =''
#try_str =''
#rerun_str='rerun_'
rerun_str=''
for name in names:
    d = name2dict(name)
    skills = np.load(f'data/{rerun_str}skill_losses_{name}.npy')
    print(skills.shape)
    skill_means = np.mean(skills, axis=0)
    skill_stds = np.std(skills, axis=0)
    inits = np.mean(np.load(f'data/{rerun_str}init_vals_{name}.npy'),axis=0)
    init_idxs = np.array([1,2,3,4,6]).astype('int')
    inits = None
    init_idxs=None
    eigs = np.power(np.arange(len(skill_means))+1, -d['alpha'])
    eigs /= np.sum(eigs)
    print(eigs)
    xs = np.arange(len(skill_means[0]))
    #print('mean at init', np.mean(1 - skill_means[:,5]))
    if np.mean(skill_means[:,0] - skill_means[:,5]) > 0.05:
        multi = (np.mean(skill_means[:,5])/np.mean(skill_means[:,0]) + 0)/1
    else:
        multi = 1
    multi=1
    for skill_mean, skill_std in zip(skill_means, skill_stds):
        plt.errorbar(xs, skill_mean/np.power(d['y_scale']/2,2), skill_std/np.power(d['y_scale']/2,2))
        #plt.errorbar(xs, skill_mean / np.power(d['y_scale']/2,2)/multi, np.zeros_like(skill_mean))
        continue
    func = partial(sigmoid, s=d['y_scale'], lr=d['lr'], batch_mul=d['batch_mul'],a=np.power(d['init'],2.0))
    plot_theo_skill(xs, eigs, skill_means, func, init_vals=inits, init_idxs=init_idxs, multi=1)
    #plt.xlim(0,100)
    plt.ylim(0,1.2)
    plt.savefig(f'plot/{rerun_str}skills_{name}_FUCK')
    plt.close()
