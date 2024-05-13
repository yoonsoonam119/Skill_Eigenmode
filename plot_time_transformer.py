import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import torch
# from utils_agd2 import file2mem, file2mem_single
import argparse
import os
import string
import statistics
from glob import glob
from utils import *

def file2mem(type_name, variables, parameters, zero=True, dict_str = None):
    if dict_str is None:
        dict_str = dict2str(**parameters)
    assert type_name in ['time', 'data', 'parameter']
    zero_mean_str = 'zero' if zero else ''
    if type_name == 'time':
        # repeat X Timesteps X skill_cnt
        files = glob(f'data/{zero_mean_str}/time/skill_losses_{dict_str}*.npy')
        skill_losses, corrs, te_losses = [], [], []
        for file in files:
            f_ = file.split(f'data/{zero_mean_str}/time/skill_losses_{dict_str}')[-1]
            skill_loss = np.load(f'data/{zero_mean_str}/time/skill_losses_{dict_str}{f_}')
            if skill_loss.shape[1]<1000:
                continue
            skill_losses+=[skill_loss]
            corrs += [np.load(f'data/{zero_mean_str}/time/corr_{dict_str}{f_}')]
            te_losses +=   [np.load(f'data/{zero_mean_str}/time/te_loss_{dict_str}{f_}')]

        lens = min([i.shape[1] for i in skill_losses])
        skill_losses = np.stack([i[0,:lens,:] for i in skill_losses], axis=0)
        corrs = np.stack([i[0,:lens,:] for i in corrs], axis=0)
        te_losses = np.stack([i[0,:lens] for i in te_losses], axis=0)

        skills_mean = np.mean(skill_losses,axis=0).T
        skills_std = np.std(skill_losses,axis=0).T
        corrs_mean = np.mean(corrs,axis=0).T
        corrs_std = np.std(corrs,axis=0).T
        te_mean = np.mean(te_losses,axis=0)
        te_std = np.std(te_losses,axis=0)

        # skill_cnt X timesteps
        return corrs_mean, corrs_std, skills_mean, skills_std, te_mean, te_std
    else:
        skills_mean = []
        skills_std = []
        corrs_mean = []
        corrs_std = []
        te_mean = []
        te_std = []
        for variable in variables:
            print('FILE I/O:', type_name, variable, dict_str, flush=True)
            with open(f'data/{zero_mean_str}/{type_name}/skill_losses_{variable}_{dict_str}', 'r') as f:
                skills = np.array([[float(x) for x in line.split()] for line in f])
                skills_mean.append(np.mean(skills,axis=0))
                skills_std.append(np.std(skills,axis=0))
            with open(f'data/{zero_mean_str}/{type_name}/corr_{variable}_{dict_str}', 'r') as f:
                corrs = np.array([[float(x) for x in line.split()] for line in f])
                corrs_mean.append(np.mean(corrs, axis=0))
                corrs_std.append(np.std(corrs, axis=0))
            with open(f'data/{zero_mean_str}/{type_name}/te_loss_{variable}_{dict_str}', 'r') as f:
                te = np.array([[float(x) for x in line.split()] for line in f])
                te_mean.append(np.mean(te, axis=0).item())
                te_std.append(np.std(te, axis=0).item())

        return np.stack(corrs_mean).T, np.stack(corrs_std).T, np.stack(skills_mean).T, np.stack(skills_std).T, np.array(te_mean), np.array(te_std)

def file2mem_single(type_name, variables, parameters, zero=True, dict_str = None):
    if dict_str is None:
        dict_str = dict2str(**parameters)
    assert type_name in ['time', 'data', 'parameter']
    zero_mean_str = 'zero' if zero else ''
    if type_name == 'time':
        # repeat X Timesteps X skill_cnt
        skill_losses = np.load(f'data/{zero_mean_str}/time/skill_losses_{dict_str}.npy')
        corrs = np.load(f'data/{zero_mean_str}/time/corr_{dict_str}.npy')
        te_losses = np.load(f'data/{zero_mean_str}/time/te_loss_{dict_str}.npy')

        skills_mean = np.mean(skill_losses,axis=0).T
        skills_std = np.std(skill_losses,axis=0).T
        corrs_mean = np.mean(corrs,axis=0).T
        corrs_std = np.std(corrs,axis=0).T
        te_mean = np.mean(te_losses,axis=0)
        te_std = np.std(te_losses,axis=0)

        # skill_cnt X timesteps
        return corrs_mean, corrs_std, skills_mean, skills_std, te_mean, te_std
    else:
        skills_mean = []
        skills_std = []
        corrs_mean = []
        corrs_std = []
        te_mean = []
        te_std = []
        for variable in variables:
            print('FILE I/O:', type_name, variable, dict_str, flush=True)
            with open(f'data/{zero_mean_str}/{type_name}/skill_losses_{variable}_{dict_str}', 'r') as f:
                skills = np.array([[float(x) for x in line.split()] for line in f])
                skills_mean.append(np.mean(skills,axis=0))
                skills_std.append(np.std(skills,axis=0))
            with open(f'data/{zero_mean_str}/{type_name}/corr_{variable}_{dict_str}', 'r') as f:
                corrs = np.array([[float(x) for x in line.split()] for line in f])
                corrs_mean.append(np.mean(corrs, axis=0))
                corrs_std.append(np.std(corrs, axis=0))
            with open(f'data/{zero_mean_str}/{type_name}/te_loss_{variable}_{dict_str}', 'r') as f:
                te = np.array([[float(x) for x in line.split()] for line in f])
                te_mean.append(np.mean(te, axis=0).item())
                te_std.append(np.std(te, axis=0).item())

        return np.stack(corrs_mean).T, np.stack(corrs_std).T, np.stack(skills_mean).T, np.stack(skills_std).T, np.array(te_mean), np.array(te_std)

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rc('font', **{'size':12})
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
    parser.add_argument("-c", "--skill_cnt", help="skill_cnt", type=int, default=6)
    parser.add_argument("-s", "--theparam", help="theparam", type=float, default=6)
    args = parser.parse_args()
    name = '64_20_5_001_15_005_3_5'
    for _ in range(1):
        dict_str = 'time_corr_32_5_50_5e-05_19_10_3_1'
        dict_str = dict_str.split("corr_")[-1][:-4]
        NAME = dict_str
        n_skills = 5

        param_dict = {'bits': 16, 'skill_cnt': n_skills, 'batch_mul': 1, 'lr': 0.0005, 'alpha': 1.9,
                    'init': 0.001,
                    'skill_bit_cnt': 5, 'y_scale': 1, 'opt': args.opt, 'act': args.act}

        xs = np.arange(1000,6001,100)
        xs = np.arange(1000,6001,1000)

        dict_str = '32_5_50_5e-05_19_10_3_1_14_00_48_072044'

        corrs_mean, corrs_std, skills_mean, skills_std, te_mean, te_std = file2mem_single('time', xs, param_dict, zero=True, dict_str=dict_str)
        eigs = np.power(np.arange(param_dict['skill_cnt'])+1, -param_dict['alpha'])
        eigs /= np.sum(eigs)
        fig, ax = plt.subplots(1,2)
        for i, (c_mean, c_std) in enumerate(zip(corrs_mean, corrs_std)):
            xs = np.array([(100*i)+1 for i in range(len(c_mean))])
            ax[0].plot(xs, c_mean/param_dict['y_scale'], label=rf'$k={i+1}$', color=f'C{i}')
            ax[0].fill_between(xs, (c_mean + c_std)/param_dict['y_scale'], (c_mean - c_std)/param_dict['y_scale'],  color=f'C{i}', alpha=0.2)
        ax[0].set_xlabel(R'$T$')
        ax[0].set_ylabel(r'$\mathcal{R}_k/S$')
        ax[0].set_xscale('log')
        ax[0].set_xlim([1e3, 1e6])

        # Plot 2

        dict_str = 'time_corr_32_5_50_5e-05_19_10_3_1'
        dict_str = dict_str.split("corr_")[-1][:-4]

        NAME = dict_str

        n_skills = int(dict_str.split('_')[1])

        param_dict = {'bits': 16, 'skill_cnt': n_skills, 'batch_mul': 1, 'lr': 0.0005, 'alpha': 1.9,
                    'init': 0.001,
                    'skill_bit_cnt': 5, 'y_scale': 1, 'opt': args.opt, 'act': args.act}

        corrs_mean, corrs_std, skills_mean, skills_std, te_mean, te_std = file2mem('time', xs, param_dict, zero=True, dict_str=dict_str)
        eigs = np.power(np.arange(param_dict['skill_cnt'])+1, -param_dict['alpha'])
        eigs /= np.sum(eigs)
        xs = np.array([(100*i)+1 for i in range(len(c_mean))])

        targs = [0.05, 0.25, 0.45, 0.65, 0.85]
        mids = {i:[] for i in targs}
        for targ in targs:
            for i in range(corrs_mean.shape[0]):
                for j in range(corrs_mean.shape[1]):
                    if corrs_mean[i,j]>targ:
                        mids[targ].append(j)
                        break

        print(mids)
        means = {i:statistics.mean([j[i] for j in list(mids.values())[:4]]) for i in range(5)}
        stds = {i:statistics.stdev([j[i] for j in list(mids.values())[:4]]) for i in range(5)}
        x = [1+i for i in range(5)]

        color_cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        print(color_cycle)
        means_, stds_ = list(means.values()), list(stds.values())
        for kk in range(5):
            ax[1].errorbar(x[kk],[100*i for i in [means_[kk]]], yerr=[100*i for i in [stds_[kk]]], label='__nolabel__', marker='x', linestyle = 'none', color=color_cycle[kk])
        c=4000
        ax[1].plot(x, [c*i**1.9 for i in x], linestyle = 'dashed', color='grey', label=r'$\tau_{emerge}(k) \propto 1.9^k$')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_xlabel(r'$\mathrm{Skill~index~}k$')
        ax[1].set_ylabel(r'$\tau_{emerge}(k)$')

        handles, labels = ax[0].get_legend_handles_labels()
        order = [0, 1, 2, 3, 4]
        ax[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                    loc="upper center", bbox_to_anchor=(1.15, 1.3), ncol=5, frameon=False, fontsize=12)
        ax[1].set_xticks([1,2,3,4,5])
        ax[1].set_xticklabels([1,2,3,4,5])

        ax[1].legend(fontsize=12, frameon=False)

        ax[0].set_box_aspect(0.66)
        ax[1].set_box_aspect(0.66)
        fig.set_size_inches(7.5,2)
        plt.savefig(f'plot/transformer.pdf', bbox_inches='tight', format='pdf', dpi=300)
        plt.close()
