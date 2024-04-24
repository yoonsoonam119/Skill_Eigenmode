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
from utils import file2mem

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


#def theo_param(xs, param_dict, theparam, idx):
#    val = (xs - idx*theparam)/theparam
#    return np.clip(val, 0, 1)

def theo_param_loss(xs, param_dict, theparam,  idx):
    val = (-xs + (idx+1)*theparam)/theparam
    return np.clip(val, 0, 1)

#def theo_param(xs, param_dict, the_param, idx):
#    return 1-np.sqrt(theo_param_loss(xs, param_dict, the_param, idx))

def theo_param(xs, param_dict, the_param, idx):
    val = (xs + -(idx)*the_param)/the_param
    return np.clip(val, 0, 1)

def theo_time(xs, param_dict, theparam, idx):
    eigs = np.power(np.arange(param_dict['skill_cnt'])+1, -param_dict['alpha'])
    eigs /= np.sum(eigs)
    c = param_dict['y_scale']/np.power(param_dict['init'],2.0)*3.6 -1
    free_p = theparam
    mul = eigs[idx]*param_dict['y_scale']*param_dict['lr']*param_dict['batch_mul']*4*10/free_p/50
    return 1/(1+c*np.exp(-mul*xs))

def theo_data(xs, param_dict, theparam, idx):
    eigs = np.power(np.arange(param_dict['skill_cnt'])+1, -param_dict['alpha'])
    eigs /= np.sum(eigs)
    lk = 1-xs*eigs[idx]/theparam
    lk *= lk >0
    return 1- np.sqrt(lk)

def plot_ax(ax, type_name, xs, theo, param_dict, theparam, sel_idxs=None):
    corrs_mean, corrs_std, skills_mean, skills_std, te_mean, te_std = file2mem(type_name, xs, param_dict, zero=True)
    for i, (c_mean, c_std) in enumerate(zip(corrs_mean, corrs_std)):
        if sel_idxs is not None:
            c_mean = c_mean[sel_idxs]
            c_std = c_std[sel_idxs]
            if i == 0:
                xs = xs[sel_idxs]
        #ax.errorbar(xs, c_mean/param_dict['y_scale'], c_std/param_dict['y_scale'], label=rf'$k={i}$', color=f'C{i}', linestyle='dashed')
        ax.plot(xs, c_mean/param_dict['y_scale'], label=rf'$k={i}$', color=f'C{i}', linestyle='dashed')
        ax.fill_between(xs, (c_mean + c_std)/param_dict['y_scale'], (c_mean - c_std)/param_dict['y_scale'],  color=f'C{i}', alpha=0.2)
        ax.plot(xs, theo(xs, param_dict, theparam, i), linestyle='solid', color=f'C{i}')
        if i == 4:
            break
    if type_name == 'time':
        name = 'Time'
        idx = 0
        #ax.set_xscale('log')
    elif type_name == 'data':
        name = 'Data'
        idx = 1
        #ax.set_xscale('log')
    elif type_name == 'parameter':
        name = 'Parameter'
        idx = 2
        ax.set_xlim(0,11)
        #ax.set_xscale('log')
    ax.set_title(latex_transform(f'{name} Calibration',idx,alp=False), y=-0.4, fontdict={'size':12})
    box = ax.get_position()
    ax.set_position([box.x0, box.y0,
                     box.width, box.height * 0.9])
    ax.set_yticks([0,0.2, 0.4, 0.6,0.8,1])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r'$Skill$' +' ' + r'$strength$' + ' ' + r'$\mathcal{R}_k$')
    #ax.xscale('log')


def plot_all(alphas, p = 1):
    fig, axs = plt.subplots(1, 3, figsize=(10,2.5),gridspec_kw={'top':0.95, 'bottom': 0.15})
    plt.subplots_adjust(wspace=0.32)

    eigss = []
    for alpha in alphas:
        eigs = np.power(np.arange(p) + 1, -float(alpha)-1)
        eigs /= np.sum(eigs)
        eigss.append(eigs)
    linst = ['solid', 'dashed', 'dotted']
    #param_dict = {'bits': 32, 'skill_cnt': 5, 'batch_mul': 5, 'lr': 0.05, 'alpha': 1.6,
    #              'init': 0.001,
    #              'skill_bit_cnt': 3, 'y_scale': 5, 'opt': 'sgd', 'act': 'relu'}
    #theparam =23.5
    param_dict = {'bits': 32, 'skill_cnt': 1, 'batch_mul': 5, 'lr': 0.02, 'alpha': 1.6,
                  'init': 0.01,
                  'skill_bit_cnt': 3, 'y_scale': 5, 'opt': 'sgd', 'act': 'relu'}
    theparam =22.0
    xs = np.arange(0,50,1)
    plot_ax(axs[0], 'time', xs*50, theo_time, param_dict, theparam, sel_idxs=np.arange(0,50,1))
    axs[0].set_xlabel('$T$')

    param_dict = {'bits': 32, 'skill_cnt': 1, 'batch_mul': 5, 'lr': 0.05, 'alpha': 1.6,
                  'init': 0.001,
                  'skill_bit_cnt': 3, 'y_scale': 5, 'opt': 'sgd', 'act': 'relu'}
    theparam =800
    xs = np.arange(100,2001,100)
    #xs = np.array(list(np.arange(500, 10500, 500)) + list(np.arange(10000, 40000, 5000)))
    plot_ax(axs[1], 'data', xs, theo_data, param_dict, theparam)
    axs[1].set_xlabel('$D$')

    param_dict = {'bits': 32, 'skill_cnt': 1, 'batch_mul': 5, 'lr': 0.05, 'alpha': 1.6,
                  'init': 0.05,
                  'skill_bit_cnt': 3, 'y_scale': 5, 'opt': 'adam', 'act': 'relu'}
    theparam =4
    #xs = np.arange(1,21)
    xs = np.arange(1,13)
    plot_ax(axs[2], 'parameter', xs, theo_param, param_dict, theparam)
    axs[2].set_xlabel('$N$')

    handles, labels = axs[0].get_legend_handles_labels()
    #legend1 = plt.legend(ps, [title for title in titles], ncol=len(titles), loc=(-0.55,1.1))
    #legend_ = plt.legend(ps, [rf'$k={i}$' for i in range(1,6)], ncol=5, loc='lower center', fontsize=13,
    #                     columnspacing=1, handlelength=1, bbox_to_anchor=(-1.9, 1.00, 2.0, 0.3), frameon=False)
    ps = [plt.plot([0], [0], color='C0', linestyle='solid')[0], plt.plot([0], [0], color='C0', linestyle='dashed')[0]]
    legend_ = plt.legend(ps, [rf'${i}$' for i in ['NN', 'extended~model']], ncol=2, loc='lower center',
                         fontsize=16,
                         columnspacing=1, handlelength=1, bbox_to_anchor=(-1.0, 0.97), frameon=False)
    fig.add_artist(legend_)
    #plt.gca().add_artist(legend1)

    fig.savefig(f'plot/emergence_calibration_all', dpi=300, bbox_inches='tight')
    fig.savefig(f'plot/emergence_calibration_all.pdf',format="pdf", dpi=300, bbox_inches='tight')
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
    exit()
