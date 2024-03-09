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

def theo(xs, param_dict, args, eig):
    c = param_dict['y_scale']/np.power(param_dict['init'],2.0)*3.6 -1
    free_p = args.theparam
    mul = eig*param_dict['y_scale']*param_dict['lr']*param_dict['batch_mul']*4*10/free_p
    return 1/(1+c*np.exp(-mul*xs))

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
    parser.add_argument("-s", "--theparam", help="theparam", type=float, default=18000)
    args = parser.parse_args()
    name = '64_20_5_001_15_005_3_5'

    param_dict = {'bits': 16, 'skill_cnt': 5, 'batch_mul': 5, 'lr': 0.05, 'alpha': 1.6,
                  'init': 0.001,
                  'skill_bit_cnt': 3, 'y_scale': 5, 'opt': args.opt, 'act': args.act}

    xs = np.arange(1000,6001,100)
    xs = np.arange(1000,6001,1000)
    corrs_mean, corrs_std, skills_mean, skills_std, te_mean, te_std = file2mem('time', xs, param_dict, zero=True)
    eigs = np.power(np.arange(param_dict['skill_cnt'])+1, -param_dict['alpha'])
    eigs /= np.sum(eigs)
    for i, (c_mean, c_std) in enumerate(zip(corrs_mean, corrs_std)):
        plt.errorbar(xs, c_mean, c_std, label=rf'$k={i}$', color=f'C{i}')
        plt.plot(xs, theo(xs, param_dict, args, eigs[i]), linestyle='dashed', color=f'C{i}')
    plt.legend()
    plt.xlabel(R'$T$')
    plt.ylabel(r'$\mathcal{R}_k$')
    plt.savefig(f'plot/time/time_corr_{dict2str(**param_dict)}', bbox_inches='tight')
    plt.savefig(f'plot/time/time_corr_{dict2str(**param_dict)}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    for i, (s_mean, s_std) in enumerate(zip(skills_mean, skills_std)):
        plt.errorbar(xs, s_mean, s_std, label=rf'$k={i}$', color=f'C{i}')
        plt.plot(xs, np.power(param_dict['y_scale']*(1-theo(xs, param_dict, args, eigs[i])),2), linestyle='dashed', color=f'C{i}')
    plt.legend()
    plt.xlabel(R'$T$')
    plt.ylabel(r'$\mathcal{L}_k$')
    plt.savefig(f'plot/time/time_skill_{dict2str(**param_dict)}', bbox_inches='tight')
    plt.savefig(f'plot/time/time_skill_{dict2str(**param_dict)}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

