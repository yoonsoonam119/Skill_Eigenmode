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

def theo(xs, param_dict, args):
    c = param_dict['y_scale']/np.power(param_dict['init'],2.0)*3.3 -1
    free_p = args.theparam
    mul = param_dict['y_scale']*param_dict['lr']*param_dict['batch_mul']*4*10/free_p
    return np.power(param_dict['y_scale'],2)*np.power(1-1/(1+c*np.exp(-mul*xs)),2)

def theo_corrs(xs, param_dict, args):
    c = param_dict['y_scale']/np.power(param_dict['init'],2.0)*3.3 -1
    free_p = args.theparam
    mul = param_dict['y_scale']*param_dict['lr']*param_dict['batch_mul']*4*10/free_p
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
    parser.add_argument("-s", "--theparam", help="theparam", type=float, default=22)
    args = parser.parse_args()
    name = '64_20_5_001_15_005_3_5'

    param_dict = {'bits': 32, 'skill_cnt': 1, 'batch_mul': 5, 'lr': 0.02, 'alpha': 1.6,
                  'init': 0.01,
                  'skill_bit_cnt': 3, 'y_scale': 5, 'opt': args.opt, 'act': args.act}

    #xs = np.arange(0,200,1)
    xs = np.arange(0,100,1)
    corrs_mean, corrs_std, skills_mean, skills_std, te_mean, te_std = file2mem('time', xs, param_dict, zero=True)
    print(corrs_mean.shape)
    print(te_mean/2)
    print(te_mean.shape)
    print(corrs_mean[0][25])
    lims = np.arange(0,max(xs)//2,2)
    plt.errorbar(xs[lims]*50, te_mean[lims]/2, te_std[lims]/2, label='NN')
    plt.plot(xs[:max(xs)//2]*50, theo(xs[:max(xs)//2], param_dict, args)/2, label='extended toy', linestyle='dashed', color='C0')
    ps = [plt.plot([0], [0], color='C0', linestyle='solid')[0], plt.plot([0], [0], color='C0', linestyle='dashed')[0]]
    # legend1 = plt.legend(ps, [title for title in titles], ncol=len(titles), loc=(-0.55,1.1))
    legend_ = plt.legend(ps, [rf'${i}$' for i in ['NN', 'extended~model']], ncol=2, loc='lower center',
                         fontsize=16,
                         columnspacing=1, handlelength=1, bbox_to_anchor=(0.5, 0.97), frameon=False)
    #plt.legend()
    #plt.plot(data_cnts, sigmoid(data_cnts, 0.002, 1000))
    #plt.axvline(5000, color='black', linestyle='dotted')
    plt.xlabel(R'$T$')
    plt.ylabel(r'$\mathcal{L}$')
    #plt.xscale('log')
    #plt.yscale('log')

    plt.savefig(f'plot/time/time1_{dict2str(**param_dict)}', bbox_inches='tight')
    plt.savefig(f'plot/time/time1_{dict2str(**param_dict)}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


    plt.plot(xs[:max(xs)//2]*50, theo_corrs(xs[:max(xs)//2], param_dict, args), label='extended toy', linestyle='solid', color='C0')
    #plt.errorbar(xs[lims]*50, corrs_mean[0][lims]/param_dict['y_scale'], corrs_std[0][lims]/param_dict['y_scale'], label='NN')
    plt.plot(xs[lims]*50, corrs_mean[0][lims]/param_dict['y_scale'], label='NN', linestyle='dashed')
    plt.fill_between(xs[lims] * 50, (corrs_mean[0][lims] + corrs_std[0][lims]) / param_dict['y_scale'],
                     (corrs_mean[0][lims] - corrs_std[0][lims]) / param_dict['y_scale'],
                     color=f'C{0}', alpha=0.2)
    ps = [plt.plot([0], [0], color='C0', linestyle='solid')[0], plt.plot([0], [0], color='C0', linestyle='dashed')[0]]
    # legend1 = plt.legend(ps, [title for title in titles], ncol=len(titles), loc=(-0.55,1.1))
    legend_ = plt.legend(ps, [rf'${i}$' for i in ['extended~model','NN']], ncol=2, loc='lower center',
                         fontsize=20,
                         columnspacing=1, handlelength=0.7, bbox_to_anchor=(0.5, 0.97), frameon=False)
    #plt.legend()
    #plt.plot(data_cnts, sigmoid(data_cnts, 0.002, 1000))
    #plt.axvline(5000, color='black', linestyle='dotted')
    plt.xlabel(R'$T$', fontdict={'fontsize':20})
    plt.ylabel(r'$\mathcal{R}_1/S$', fontdict={'fontsize':20})
    plt.xlim(0,2400)

    #plt.xscale('log')
    #plt.yscale('log')

    plt.savefig(f'plot/time/time1_corrs_{dict2str(**param_dict)}', bbox_inches='tight')
    plt.savefig(f'plot/time/time1_corrs_{dict2str(**param_dict)}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
