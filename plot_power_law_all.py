import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import copy
from save_load import YamlReader
from executor import Dispensor, RangeSampler, ChopSampler, ListSampler
from functools import partial
from collections import OrderedDict
import yaml
import string

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rc('font', **{'size':11})

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

def data_power(x, alpha, eigs, s=1):
    return np.sum(eigs*(1-eigs))*np.power(s,2)*np.power(x,-(alpha)/(alpha+1))/2

def temp_power(x, alpha, eigs, s=1,init=0.1):
    return np.power(s-np.power(init,2),2)*np.power(x,-(alpha)/(alpha+1))/2

def param_power(x, alpha, eigs, s=1):
    return np.power(s,2)/(alpha+1)*np.power(x,-alpha)/2

def data_theo(ns, eigs, s=1):
    return [np.sum(np.power((1 - eigs), n)*np.power(s,2)*eigs)/2 for n in ns]

def temp_theo(x, eigs, s=1, lr=1, init=0.1):
    c = s/np.power(init,2)-1
    ret = 0
    for eig in eigs:
        ret += np.power(s-s/(1+c*np.exp(-eig*2*s*x*lr)),2)*eig
    return ret/2

def param_theo(ps, eigs, s):
    ret = [np.power(s,2)/2 * np.sum(eigs[p:]) for p in ps]
    return ret


def plot_ax(ax, theo_func, power_func, name, eigss, alphas, idx, max_range=10000):
    ts = np.arange(1,max_range)
    for i, eigs in enumerate(eigss):
        ax.plot(ts, theo_func(ts, eigs,s=1), color=f'C{i}')
        ax.plot(ts, power_func(ts, alphas[i], eigs, s=1), color=f'C{i}', linestyle='dotted')
    ax.set_title(latex_transform(f'{name} Scaling',idx,alp=True), y=-0.4)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0,
                     box.width, box.height * 0.9])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_ylabel(r'$\mathcal{L}$')


def plot_all(alphas, p = 1):
    fig, axs = plt.subplots(1, 3, figsize=(10,2.5),gridspec_kw={'top':0.95, 'bottom': 0.15})
    plt.subplots_adjust(wspace=0.32)

    eigss = []
    for alpha in alphas:
        eigs = np.power(np.arange(p) + 1, -float(alpha)-1)
        eigs /= np.sum(eigs)
        eigss.append(eigs)
    linst = ['solid', 'dashed', 'dotted']
    plot_ax(axs[0], temp_theo, temp_power, 'Dynamic', eigss, alphas, 0,10000)
    axs[0].set_xlabel('$T$')
    ps0 = [axs[0].plot([0], [0], color='black', linestyle='dotted')[0]]
    legend0 = plt.legend(ps0, [r'$\mathcal{L}\propto T^{-\alpha/(\alpha+1)}$'],bbox_to_anchor=(-2.0, 0.3),
                         handlelength=1, frameon=False)
    fig.add_artist(legend0, )

    plot_ax(axs[1], data_theo, data_power, 'Data', eigss, alphas, 1, 1000)
    axs[1].set_xlabel('$D$')
    ps1 = [axs[1].plot([0], [0], color='black', linestyle='dotted')[0]]
    legend1 = plt.legend(ps1, [r'$\mathcal{L}\propto D^{-\alpha/(\alpha+1)}$'],bbox_to_anchor=(-0.65, 0.3),
                         handlelength=1, frameon=False)
    fig.add_artist(legend1, )

    plot_ax(axs[2], param_theo, param_power, 'Parameter', eigss, alphas, 2, 100)
    axs[2].set_xlabel('$N$')
    ps2 = [axs[2].plot([0], [0], color='black', linestyle='dotted')[0]]
    legend2 = plt.legend(ps2, [r'$\mathcal{L}\propto N^{-\alpha}$'],bbox_to_anchor=(0.5, 0.3),
                         handlelength=1, frameon=False)
    fig.add_artist(legend2, )

    handles, labels = axs[0].get_legend_handles_labels()
    ps = [axs[0].plot([0], [0], color=f'C{i}', linestyle='solid')[0] for i, tit in enumerate(alphas)]
    #legend1 = plt.legend(ps, [title for title in titles], ncol=len(titles), loc=(-0.55,1.1))
    legend_ = plt.legend(ps, [r'$\alpha=$' + f'{alpha}' for alpha in alphas], ncol=len(alphas), loc='lower center',
                         columnspacing=1, handlelength=1, bbox_to_anchor=(-1.9, 1.00, 2.0, 0.3), frameon=False)
    fig.add_artist(legend_)
    #plt.gca().add_artist(legend1)

    fig.savefig(f'plot/power_law_all', dpi=300, bbox_inches='tight')
    fig.savefig(f'plot/power_law_all.pdf',format="pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--worker_cnt", help="total workers", type=int, default=1)
    parser.add_argument("-o", "--output", help="outputpath", type=str, default=os.getcwd())
    parser.add_argument("-m", "--modelfile", help="model_file", type=str, default='base.data')
    parser.add_argument("-d", "--d", help="plot_dnn", type=bool, default=True)
    parser.add_argument("-t", "--use_train", help="use_train_set", type=bool, default=False)
    parser.add_argument("-p", "--use_prev", help="use_train_set", type=bool, default=False)
    parser.add_argument("-r", "--repeats", help="repeats", type=int, default=2)
    parser.add_argument("-e", "--exp_name", help="exp_name", type=str, default='ww')
    parser.add_argument("-s", "--train_set", help="train_set", type=bool, default=True)
    args = parser.parse_args()
    plot_all([0.3 ,0.6, 0.9], p = 10000)
