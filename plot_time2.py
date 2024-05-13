import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils_agd2 import file2mem, dict2str, file2mem_single
import argparse
import os
import string

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
    import glob 
    # for dict_str in glob.glob('data/zero/time/corr_*'):
    #     if '5_50_5e-05_19' not in dict_str:
    #         continue
    for _ in range(1):
        dict_str = 'time_corr_32_5_50_5e-05_19_10_3_1'
        
        dict_str = dict_str.split("corr_")[-1][:-4]
        
        # NAME = 'transformer' #dict2str(**param_dict)
        # dict_str = '32_8_50_0001_15_10_3_5'
        # dict_str = '32_6_50_0001_15_10_3_5_09_50_41_227136'
        # dict_str = '32_8_50_00001_15_10_3_5_18_19_36_530533'
        NAME = dict_str

        n_skills = 5

        param_dict = {'bits': 16, 'skill_cnt': n_skills, 'batch_mul': 1, 'lr': 0.0005, 'alpha': 1.9,
                    'init': 0.001,
                    'skill_bit_cnt': 5, 'y_scale': 1, 'opt': args.opt, 'act': args.act}

        xs = np.arange(1000,6001,100)
        xs = np.arange(1000,6001,1000)
        dict_str = '32_5_50_5e-05_19_10_3_1_13_33_35_272148'
        dict_str = '32_5_50_5e-05_19_10_3_1_13_34_15_564566'
        dict_str = '32_5_50_5e-05_19_10_3_1_14_00_48_072044'
        # dict_str = '32_5_50_5e-05_19_10_3_1_16_08_52_154161'

        corrs_mean, corrs_std, skills_mean, skills_std, te_mean, te_std = file2mem_single('time', xs, param_dict, zero=True, dict_str=dict_str)
        eigs = np.power(np.arange(param_dict['skill_cnt'])+1, -param_dict['alpha'])
        eigs /= np.sum(eigs)
        fig, ax = plt.subplots(1,2)
        for i, (c_mean, c_std) in enumerate(zip(corrs_mean, corrs_std)):
            xs = np.array([(100*i)+1 for i in range(len(c_mean))])
            ax[0].plot(xs, c_mean/param_dict['y_scale'], label=rf'$k={i}$', color=f'C{i}')
            ax[0].fill_between(xs, (c_mean + c_std)/param_dict['y_scale'], (c_mean - c_std)/param_dict['y_scale'],  color=f'C{i}', alpha=0.2)
            # ax.plot(xs, theo(xs, param_dict, args, i), linestyle='solid', color=f'C{i}')
            # xs = np.array([(100*i)+1 for i in range(len(c_mean))])
            # ax.errorbar(xs, c_mean, c_std, label=rf'$k={i}$', color=f'C{i}', alpha=0.2)
            # ax[0].plot(xs, theo(xs, param_dict, args, eigs[i]), linestyle='dashed', color=f'C{i}')
        # ax[0].legend()
        ax[0].set_xlabel(R'$T$')
        ax[0].set_ylabel(r'$\mathcal{R}_k$')
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

        import statistics
        # print([[j[i] for j in mids.values()] for i in range(5)])
        means = {i:statistics.mean([j[i] for j in mids.values()]) for i in range(5)}
        stds = {i:statistics.stdev([j[i] for j in mids.values()]) for i in range(5)}

        print(mids)
        print(means)
        print(stds)

        # mids = []
        # for i in range(corrs_mean.shape[0]):
        #     print(i)
        #     for j in range(corrs_mean.shape[1]):
        #         if corrs_mean[i,j]>0.5:
        #             mids.append(j)
        #             break

        # lower = []
        # for i in range(corrs_mean.shape[0]):
        #     print(i)
        #     for j in range(corrs_mean.shape[1]):
        #         if corrs_mean[i,j]>0.4:
        #             lower.append(j)
        #             break

        # upper = []
        # for i in range(corrs_mean.shape[0]):
        #     print(i)
        #     for j in range(corrs_mean.shape[1]):
        #         if corrs_mean[i,j]>0.6:
        #             upper.append(j)
        #             break

        # std = 

        print(mids)
        # stds = [corrs_std[i][j] for i,j in zip(range(5), mids)]
        # print(stds)
        x = [1+i for i in range(5)]
        # ax[1].plot(x,[100*i for i in means.values()], label='')

        color_cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        print(color_cycle)
        means_, stds_ = list(means.values()), list(stds.values())
        for kk in range(5):
            ax[1].errorbar(x[kk],[100*i for i in [means_[kk]]], yerr=[100*i for i in [stds_[kk]]], label='__nolabel__', marker='x', linestyle = 'none', color=color_cycle[kk])
        # ax[1].fill_between(x, [100*i for i in mids[0.05]], [100*i for i in mids[0.85]], alpha=0.5)
        # ax[1].errorbar(x, [100*i for i in lower], [100*i for i in upper], alpha=0.5)
        # print(len(xs), len(stds), len(lower))
        # ax[1].errorbar(x, [100*i for i in lower], yerr=[100*i for i in stds], label=rf'$k={i}$', color=f'C{i}')
        c=4000
        ax[1].plot(x, [c*i**1.9 for i in x], linestyle = 'dashed', color='grey', label='$\propto 1.9^k$')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_xlabel('Skill')
        ax[1].set_ylabel(r'$\tau_{Emergance}$')

        # for i, (c_mean, c_std) in enumerate(zip(corrs_mean, corrs_std)):
        #     xs = np.array([(100*i)+1 for i in range(len(c_mean))])
        #     ax[1].plot(xs, c_mean/param_dict['y_scale'], label=rf'$k={i}$', color=f'C{i}')
        #     ax[1].fill_between(xs, (c_mean + c_std)/param_dict['y_scale'], (c_mean - c_std)/param_dict['y_scale'],  color=f'C{i}', alpha=0.2)
        #     ax[1].plot(xs, theo(xs, param_dict, args, eigs[i]), linestyle='dashed', color=f'C{i}')
        # ax[1].legend()
        # ax[1].set_xlabel(R'$T$')
        # ax[1].set_ylabel(r'$\mathcal{R}_k$')
        # ax[1].set_xscale('log')
        # ax[1].set_xlim([1e3, 1e6])


        handles, labels = ax[0].get_legend_handles_labels()
        order = [0, 1, 2, 3, 4]
        ax[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                    loc="upper center", bbox_to_anchor=(1.05, 1.25), ncol=5, frameon=False, fontsize=12)
        ax[1].set_xticks([1,2,3,4,5])
        ax[1].set_xticklabels([1,2,3,4,5])

        ax[1].legend(fontsize=12, frameon=False)

        ax[0].set_box_aspect(0.75)
        ax[1].set_box_aspect(0.75)
        fig.set_size_inches(8,2.5)
        plt.savefig(f'plot/main.png', bbox_inches='tight')
        # plt.savefig(f'plot/time/time_corr_{NAME}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # fig, ax = plt.subplots()
        # for i, (s_mean, s_std) in enumerate(zip(skills_mean, skills_std)):
        #     xs = np.array([(100*i)+1 for i in range(len(s_mean))])
        #     ax.errorbar(xs, s_mean, s_std, label=rf'$k={i}$', color=f'C{i}')
        #     ax.plot(xs, np.power(param_dict['y_scale']*(1-theo(xs, param_dict, args, eigs[i])),2), linestyle='dashed', color=f'C{i}')
        # ax.legend()
        # ax.set_xlabel(R'$T$')
        # ax.set_ylabel(r'$\mathcal{L}_k$')
        # # ax.set_xlim([1e3, 1e6])
        # # ax.set_xscale('log')
        # plt.savefig(f'plot/time/time_skill_{NAME}', bbox_inches='tight')
        # # plt.savefig(f'plot/time/time_skill_{NAME}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        # plt.close()

