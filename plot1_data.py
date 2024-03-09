import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import file2mem, dict2str
import argparse
import os

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

def theo(xs, param_dict, args):
    ys = np.power(param_dict['y_scale'],2)*(1-xs/args.theparam)
    ys *= ys >0
    return ys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="outputpath", type=str, default=os.getcwd())
    parser.add_argument("-n", "--worker_cnt", help="worker_cnt", type=int, default=1)
    parser.add_argument("-i", "--iter_dict", help="do_iteration", type=int, default=0)
    parser.add_argument("-b", "--bits", help="bits", type=int, default=16)
    parser.add_argument("-p", "--opt", help="opts", type=str, default='sgd')
    parser.add_argument("-a", "--act", help="activation", type=str, default='relu')
    #parser.add_argument("-w", "--width", help="width", type=int, default=100)
    parser.add_argument("-z", "--zero_mean", help="zero_mean", type=int, default=1)
    parser.add_argument("-c", "--skill_cnt", help="skill_cnt", type=int, default=5)
    parser.add_argument("-s", "--theparam", help="theparam", type=float, default=15)
    args = parser.parse_args()
    name = '64_20_5_001_15_005_3_5'

    param_dict = {'bits': 16, 'skill_cnt': 1, 'batch_mul': 5, 'lr': 0.05, 'alpha': 1.6,
                  'init': 0.001,
                  'skill_bit_cnt': 3, 'y_scale': 5, 'opt': args.opt, 'act': args.act}

    xs = list(np.arange(1000,6001,100))
    corrs_mean, corrs_std, skills_mean, skills_std, te_mean, te_std = file2mem('data', xs, param_dict, zero=True)
    plt.errorbar(xs, te_mean/2, te_std/2, label='NN')
    plt.plot(xs, theo(xs, param_dict, args)/2, label='extended toy', linestyle='dashed', color='C0')
    plt.legend()
    #plt.plot(data_cnts, sigmoid(data_cnts, 0.002, 1000))
    #plt.axvline(5000, color='black', linestyle='dotted')
    plt.xlabel(R'$D$')
    plt.ylabel(r'$\mathcal{L}$')
    #plt.xscale('log')
    #plt.yscale('log')

    plt.savefig(f'plot/data/data1_{dict2str(**param_dict)}', bbox_inches='tight')
    plt.savefig(f'plot/data/data1_{dict2str(**param_dict)}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
