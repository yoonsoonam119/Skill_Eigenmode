import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import file2mem, dict2str
import argparse
import os

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rc('font', **{'size':15})
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

def theo(xs, args, eig):
    lk = 1-xs*eig/args.theparam
    lk *= lk >0
    return 1- np.sqrt(lk)

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
    parser.add_argument("-s", "--theparam", help="theparam", type=float, default=800)
    args = parser.parse_args()
    name = '64_20_5_001_15_005_3_5'
    alphas = [1.3, 1.6, 1.9]
    for alpha in alphas:
        param_dict = {'bits': 32, 'skill_cnt': 5, 'batch_mul': 5, 'lr': 0.05, 'alpha': alpha,
                      'init': 0.001,
                      'skill_bit_cnt': 3, 'y_scale': 5, 'opt': args.opt, 'act': args.act}
        eigs = np.power(np.arange(param_dict['skill_cnt'])+1, -param_dict['alpha'])
        eigs /= np.sum(eigs)

        #xs = np.array(list(np.arange(500, 10500, 500)) + list(np.arange(10000, 40000, 5000)))
        xs = np.array(list(np.arange(500, 10500, 500)) + list(np.arange(10000, 30000, 5000)))
        xs = np.array(list(np.arange(500, 10500, 500)) + list(np.arange(10000, 16000, 1000)) + list(np.arange(20000, 40000, 5000)))
        corrs_mean, corrs_std, skills_mean, skills_std, te_mean, te_std = file2mem('data', xs, param_dict, zero=True)
        for i, (c_mean, c_std, eig) in enumerate(zip(corrs_mean, corrs_std, eigs)):
            #plt.errorbar(xs, c_mean/param_dict['y_scale'], c_std/param_dict['y_scale'], label=rf'$I={i+1}$', color=f'C{i}')
            plt.plot(xs, c_mean/param_dict['y_scale'], label=rf'$k={i+1}$', linestyle='dashed', color=f'C{i}')
            plt.fill_between(xs, (c_mean + c_std) / param_dict['y_scale'],
                             (c_mean - c_std) / param_dict['y_scale'],
                             color=f'C{i}', alpha=0.2)
            plt.plot(xs, theo(xs, args, eig), linestyle='solid', color=f'C{i}')
            if i == 4:
                break
        #plt.legend()
        ps = [plt.plot([0], [0], color=f'C{i}', linestyle='solid')[0] for i in range(5)]
        ps = [plt.plot([0], [0], color='white', linestyle='solid')[0]] + ps
        #legend1 = plt.legend(ps, [title for title in titles], ncol=len(titles), loc=(-0.55,1.1))
        legend_ = plt.legend(ps, [r'$k=$' if i ==0 else rf'${i}$' for i in range(6)], ncol=6, loc='lower center',
                             fontsize=20,
                             columnspacing=1, handlelength=0.7, bbox_to_anchor=(0.45, 0.97), frameon=False)
        plt.xlabel(R'$D$', fontdict={'fontsize':20})
        plt.ylabel(r'$\mathcal{R}_k/S$', fontdict={'fontsize':20})
        plt.xscale('log')
        #plt.xlim(0,5000)
        plt.savefig(f'plot/data/data_corr_{dict2str(**param_dict)}', bbox_inches='tight')
        plt.savefig(f'plot/data/data_corr_{dict2str(**param_dict)}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        for i, (s_mean, s_std, eig) in enumerate(zip(skills_mean, skills_std, eigs)):
            plt.errorbar(xs, s_mean/2, s_std/2, label=rf'$k={i}$', color=f'C{i}')
            plt.plot(xs, np.power(param_dict['y_scale'] * (1 - theo(xs, args, eig)), 2), linestyle='dashed',
                     color=f'C{i}')
            if i == 4:
                break
        plt.legend()
        plt.xlabel(R'$D$')
        plt.ylabel(r'$\mathcal{L}_k$')
        plt.savefig(f'plot/data/data_skill_{dict2str(**param_dict)}', bbox_inches='tight')
        plt.savefig(f'plot/data/data_skill_{dict2str(**param_dict)}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()