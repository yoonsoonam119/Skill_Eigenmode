import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import string
from utils import file2mem

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rc('font', **{'size':12})

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

def theo_param_loss(xs, param_dict, theparam,  idx):
    val = (-xs + (idx+1)*theparam)/theparam
    return np.clip(val, 0, 1)

def theo_param(xs, param_dict, the_param, idx):
    val = (xs + -(idx)*the_param)/the_param
    return np.clip(val, 0, 1)

def theo_time(xs, param_dict, theparam, idx):
    probs = np.power(np.arange(param_dict['skill_cnt'])+1, -param_dict['alpha'])
    probs /= np.sum(probs)
    # the following equation for initial $\mathcal{R}_k(0)$ from initial standard deviation
    # matches the empirical measurement of $\mathcal{R}_k(0)$
    c = param_dict['y_scale']/np.power(param_dict['init'],2.0)*3.6 -1
    free_p = theparam
    # For the scaling constant, we multiply 4 instead of 2: as we empirically used L^2 and not L^2/2.
    mul = probs[idx]*param_dict['y_scale']*param_dict['lr']*4/free_p
    return 1/(1+c*np.exp(-mul*xs))

def theo_data(xs, param_dict, theparam, idx):
    probs = np.power(np.arange(param_dict['skill_cnt'])+1, -param_dict['alpha'])
    probs /= np.sum(probs)
    lk = 1-xs*probs[idx]/theparam
    lk *= lk >0
    return 1- np.sqrt(lk)

def plot_ax(ax, type_name, xs, theo, param_dict, theparam, sel_idxs=None, alp=True):
    corrs_mean, corrs_std, skills_mean, skills_std, te_mean, te_std = file2mem(type_name, xs, param_dict, zero=True)
    for i, (c_mean, c_std) in enumerate(zip(corrs_mean, corrs_std)):
        if sel_idxs is not None:
            c_mean = c_mean[sel_idxs]
            c_std = c_std[sel_idxs]
            if i == 0:
                xs = xs[sel_idxs]
        ax.plot(xs, c_mean/param_dict['y_scale'], label=rf'$k={i}$', color=f'C{i}', linestyle='dashed')
        ax.fill_between(xs, (c_mean + c_std)/param_dict['y_scale'], (c_mean - c_std)/param_dict['y_scale'],  color=f'C{i}', alpha=0.2)
        ax.plot(xs, theo(xs, param_dict, theparam, i), linestyle='solid', color=f'C{i}')
        if i == 4:
            break
    if type_name == 'time':
        name = 'Time'
        idx = 0
        ax.set_xscale('log')
    elif type_name == 'data':
        name = 'Data'
        idx = 1
        ax.set_xscale('log')
    elif type_name == 'parameter':
        name = 'Parameter'
        idx = 2
        ax.set_xscale('log')
    ax.set_title(latex_transform(f'{name} emergence',idx,alp=alp), y=-0.4)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0,
                     box.width, box.height * 0.9])
    ax.set_yticks([0,0.2, 0.4, 0.6,0.8,1])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r'$\mathrm{Skill}$' + ' '+ r'$\mathrm{strength}$' + '  ' + r'$\mathcal{R}_k/S$')


def plot_all(alp=True):
    fig, axs = plt.subplots(1, 3, figsize=(10,2.5),gridspec_kw={'top':0.95, 'bottom': 0.15})
    plt.subplots_adjust(wspace=0.32)

    param_dict = {'bits': 32, 'skill_cnt': 5, 'batch_mul': 5, 'lr': 0.02, 'alpha': 1.6,
                  'init': 0.01,
                  'skill_bit_cnt': 3, 'y_scale': 5, 'opt': 'sgd', 'act': 'relu'}
    theparam =22.0
    xs = np.arange(0,400,1)
    #scale up by 50 as the measurements are performed at every 50 steps
    plot_ax(axs[0], 'time', xs*50, theo_time, param_dict, theparam, sel_idxs=np.arange(0,400,10), alp=alp)
    axs[0].set_xlabel('$T$')

    param_dict = {'bits': 32, 'skill_cnt': 5, 'batch_mul': 5, 'lr': 0.05, 'alpha': 1.6,
                  'init': 0.001,
                  'skill_bit_cnt': 3, 'y_scale': 5, 'opt': 'sgd', 'act': 'relu'}
    theparam =800
    xs = np.array(list(np.arange(500, 10500, 500)) + list(np.arange(10000, 16000, 1000)) + list(np.arange(20000, 40000, 5000)))
    plot_ax(axs[1], 'data', xs, theo_data, param_dict, theparam, alp=alp)
    axs[1].set_xlabel('$D$')

    param_dict = {'bits': 32, 'skill_cnt': 5, 'batch_mul': 5, 'lr': 0.05, 'alpha': 1.6,
                  'init': 0.05,
                  'skill_bit_cnt': 3, 'y_scale': 5, 'opt': 'adam', 'act': 'relu'}
    theparam =4
    xs = np.arange(1,21)
    plot_ax(axs[2], 'parameter', xs, theo_param, param_dict, theparam, alp=alp)
    axs[2].set_xlabel('$N$')

    ps = [axs[0].plot([0], [0], color=f'C{i}', linestyle='solid')[0] for i in range(5)]
    legend_ = plt.legend(ps, [rf'$k={i}$' for i in range(1,6)], ncol=5, loc='lower center', fontsize=13,
                         columnspacing=1, handlelength=1, bbox_to_anchor=(-1.9, 1.00, 2.0, 0.3), frameon=False)
    fig.add_artist(legend_)
    alp_str = '' if alp else '_alp'

    fig.savefig(f'plot/emergence_all{alp_str}', dpi=300, bbox_inches='tight')
    fig.savefig(f'plot/emergence_all{alp_str}.pdf',format="pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    plot_all(alp=True)
