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

def theo_loss(xs, args, idx):
    val = (-xs + (idx+1)*args.theparam)/args.theparam
    return np.clip(val, 0, 1)

def theo(xs, args, idx):
    return 1-np.sqrt(theo_loss(xs, args, idx))

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
    parser.add_argument("-s", "--theparam", help="theparam", type=float, default=4)
    args = parser.parse_args()

    #param_dict = {'bits': 32, 'skill_cnt': 5, 'batch_mul': 5, 'lr': 0.05, 'alpha': 1.9,
    #              'init': 0.001,
    #              'skill_bit_cnt': 3, 'y_scale': 5, 'opt': args.opt, 'act': args.act}

    param_dict = {'bits': 32, 'skill_cnt': 5, 'batch_mul': 5, 'lr': 0.05, 'alpha': 1.5,
                  'init': 0.05,
                  'skill_bit_cnt': 3, 'y_scale': 5, 'opt': 'adam', 'act': args.act}
    #param_dict = {'bits': 16, 'skill_cnt': 5, 'batch_mul': 5, 'lr': 0.05, 'alpha': 1.7,
    #              'init': 0.00001,
    #              'skill_bit_cnt': 3, 'y_scale': 5, 'opt': 'adam', 'act': 'relu'}
    #param_dict = {'bits': 32, 'skill_cnt': 5, 'batch_mul': 5, 'lr': 0.05, 'alpha': 1.9,
    #              'init': 0.05,
    #              'skill_bit_cnt': 4, 'y_scale': 5, 'opt': 'adam', 'act': args.act}

    xs = np.arange(2,50,2)
    #xs = np.arange(2,34,2)
    #xs = np.arange(10,41,10)
    xs = np.arange(1,21)
    #xs = np.arange(1,41,2)
    #xs = np.arange(2,61,2)
    eigs = np.power(np.arange(param_dict['skill_cnt'])+1, -param_dict['alpha'])
    eigs /= np.sum(eigs)
    corrs_mean, corrs_std, skills_mean, skills_std, te_mean, te_std = file2mem('parameter', xs, param_dict, zero=True, median=False)
    for i, (c_mean, c_std) in enumerate(zip(corrs_mean, corrs_std)):
        #plt.errorbar(xs, c_mean/param_dict['y_scale'], c_std/param_dict['y_scale'], label=rf'$k={i}$', color=f'C{i}')
        plt.plot(xs, c_mean/param_dict['y_scale'], label=rf'${i + 1}$', color=f'C{i}',
                 linestyle='dashed')
        plt.fill_between(xs, (c_mean + c_std) / param_dict['y_scale'],
                         (c_mean - c_std) / param_dict['y_scale'],
                         color=f'C{i}', alpha=0.2)
        plt.plot(xs, theo(xs, args, i), linestyle='solid', color=f'C{i}')
    #plt.legend()
    ps = [plt.plot([0], [0], color=f'C{i}', linestyle='solid')[0] for i in range(5)]
    ps = [plt.plot([0], [0], color='white', linestyle='solid')[0]] + ps
    # legend1 = plt.legend(ps, [title for title in titles], ncol=len(titles), loc=(-0.55,1.1))
    legend_ = plt.legend(ps, [r'$I=$' if i == 0 else rf'${i}$' for i in range(6)], ncol=6, loc='lower center',
                         fontsize=16,
                         columnspacing=1, handlelength=1, bbox_to_anchor=(0.5, 0.97), frameon=False)
    plt.xlim(1,21)
    #plt.xlim(1,61)
    plt.xlabel(R'$N$')
    plt.ylabel(r'$\mathcal{R}_k$')
    plt.savefig(f'plot/parameter/parameter_corr_{dict2str(**param_dict)}', bbox_inches='tight')
    plt.savefig(f'plot/parameter/parameter_corr_{dict2str(**param_dict)}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    for i, (s_mean, s_std) in enumerate(zip(skills_mean, skills_std)):
        plt.errorbar(xs, s_mean, s_std, label=rf'$k={i}$', color=f'C{i}')
        plt.plot(xs, np.power(param_dict['y_scale'] * (1 - theo(xs, args, i)), 2), linestyle='dashed',
                 color=f'C{i}')
    plt.legend()
    plt.xlabel(R'$N$')
    plt.ylabel(r'$\mathcal{L}_k$')
    plt.savefig(f'plot/parameter/parameter_skill_{dict2str(**param_dict)}', bbox_inches='tight')
    plt.savefig(f'plot/parameter/parameter_skill_{dict2str(**param_dict)}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
