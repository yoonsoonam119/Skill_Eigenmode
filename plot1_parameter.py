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

def theo(xs, param_dict, args):
    ys = np.power(param_dict['y_scale'],2)*(1-xs/args.theparam)
    ys *= ys >0
    return ys

def theo_corrs(xs, param_dict, args):
    ys = np.power(param_dict['y_scale'],2)*(1-xs/args.theparam)
    ys *= ys >0
    return (param_dict['y_scale'] - np.sqrt(theo(xs, param_dict,args)))/param_dict['y_scale']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="outputpath", type=str, default=os.getcwd())
    parser.add_argument("-n", "--worker_cnt", help="worker_cnt", type=int, default=1)
    parser.add_argument("-i", "--iter_dict", help="do_iteration", type=int, default=0)
    parser.add_argument("-b", "--bits", help="bits", type=int, default=16)
    parser.add_argument("-p", "--opt", help="opts", type=str, default='adam')
    parser.add_argument("-a", "--act", help="activation", type=str, default='relu')
    #parser.add_argument("-w", "--width", help="width", type=int, default=100)
    parser.add_argument("-z", "--zero_mean", help="zero_mean", type=int, default=1)
    parser.add_argument("-c", "--skill_cnt", help="skill_cnt", type=int, default=5)
    parser.add_argument("-s", "--theparam", help="theparam", type=float, default=4)
    args = parser.parse_args()

    param_dict = {'bits': 32, 'skill_cnt': 1, 'batch_mul': 5, 'lr': 0.05, 'alpha': 1.6,
                  'init': 0.05,
                  'skill_bit_cnt': 3, 'y_scale': 5, 'opt': args.opt, 'act': args.act}

    xs = np.arange(1,32,2)
    #xs = np.arange(1,30,2)
    #xs = np.arange(1,42,5)
    xs = np.arange(1,10)
    xs = np.arange(1,13)
    corrs_mean, corrs_std, skills_mean, skills_std, te_mean, te_std = file2mem('parameter', xs, param_dict, zero=True)
    plt.plot(xs, theo(xs, param_dict, args)/2, label='extended toy', linestyle='dashed', color='C0')
    ps = [plt.plot([0], [0], color='C0', linestyle='solid')[0], plt.plot([0], [0], color='C0', linestyle='dashed')[0]]
    legend_ = plt.legend(ps, [rf'${i}$' for i in ['NN', 'extended~model']], ncol=2, loc='lower center',
                         fontsize=16,
                         columnspacing=1, handlelength=1, bbox_to_anchor=(0.5, 0.97), frameon=False)
    plt.xlim(1,9)
    #plt.plot(data_cnts, sigmoid(data_cnts, 0.002, 1000))
    #plt.axvline(5000, color='black', linestyle='dotted')
    plt.xlabel(R'$N$')
    plt.ylabel(r'$\mathcal{L}$')
    #plt.xscale('log')
    #plt.yscale('log')

    plt.savefig(f'plot/parameter/parameter1_{dict2str(**param_dict)}', bbox_inches='tight')
    plt.savefig(f'plot/parameter/parameter1_{dict2str(**param_dict)}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    #plt.errorbar(xs, corrs_mean[0]/param_dict['y_scale'], corrs_std[0]/param_dict['y_scale'], label='NN')
    #plt.plot(xs, theo_corrs(xs, param_dict, args), label='extended toy', linestyle='dashed', color='C0')
    plt.plot(xs, corrs_mean[0]/param_dict['y_scale'], linestyle='dashed',label='NN')
    plt.fill_between(xs, (corrs_mean[0] + corrs_std[0]) / param_dict['y_scale'],
                     (corrs_mean[0] - corrs_std[0]) / param_dict['y_scale'],
                     color=f'C{0}', alpha=0.2)
    plt.plot(xs, theo_corrs(xs, param_dict, args), label='extended toy', linestyle='solid', color='C0')
    ps = [plt.plot([0], [0], color='C0', linestyle='solid')[0], plt.plot([0], [0], color='C0', linestyle='dashed')[0]]
    # legend1 = plt.legend(ps, [title for title in titles], ncol=len(titles), loc=(-0.55,1.1))
    legend_ = plt.legend(ps, [rf'${i}$' for i in ['extended~model','NN']], ncol=2, loc='lower center',
                         fontsize=16,
                         columnspacing=1, handlelength=1, bbox_to_anchor=(0.5, 0.97), frameon=False)
    #plt.legend()
    #plt.plot(data_cnts, sigmoid(data_cnts, 0.002, 1000))
    #plt.axvline(5000, color='black', linestyle='dotted')
    plt.xlim(1,9)
    plt.xlabel(R'$D$')
    plt.ylabel(r'$\mathcal{R}$')
    #plt.xscale('log')
    #plt.yscale('log')

    plt.savefig(f'plot/parameter/parameter1_corrs_{dict2str(**param_dict)}', bbox_inches='tight')
    plt.savefig(f'plot/parameter/parameter1_corrs_{dict2str(**param_dict)}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
