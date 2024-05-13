import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import argparse
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

def data_power(x, alpha, probs, s=1):
    return np.sum(probs*(1-probs))*np.power(s,2)*np.power(x,-(alpha)/(alpha+1))/2

def temp_power(x, alpha, probs, s=1,init=0.1):
    return np.power(s-np.power(init,2),2)*np.power(x,-(alpha)/(alpha+1))/2

def param_power(x, alpha, probs, s=1):
    return np.power(s,2)/(alpha+1)*np.power(x,-alpha)/2

def data_theo(ns, probs, s=1):
    return [np.sum(np.power((1 - probs), n)*np.power(s,2)*probs)/2 for n in ns]

def temp_theo(x, probs, s=1, lr=1, init=0.1):
    c = s/np.power(init,2)-1
    ret = 0
    for prob in probs:
        ret += np.power(s-s/(1+c*np.exp(-prob*2*s*x*lr)),2)*prob
    return ret/2

def param_theo(ps, probs, s):
    ret = [np.power(s,2)/2 * np.sum(probs[p:]) for p in ps]
    return ret

def theo(ax, xs, probs, lr, s, init, idx):
    c = s/np.power(init,2)-1
    loss_diffs = 0
    for i, prob in enumerate(probs):
        rks = 1/(1+c*np.exp(-2*s*xs*lr*prob))
        loss_diffs += s*s*(1-np.power(1-rks,2))*prob/2
        if i in [10,20,50,70,100,200,500,700,1000,2000,5000,10000]:
            ax.plot(xs*(i+1), np.power(s,2)/2-loss_diffs, color=f'C{idx}')

def plot_ax(ax, alpha, idx, n_s, ratio=1):
    init = 0.1
    lr = 1e0
    s = 5
    probs = np.power(np.arange(n_s) + 1, -float(alpha) - 1)
    probs /= np.sum(probs)
    xs = np.logspace(1,8,1000)
    theo(ax, xs, probs, lr, s, init, idx)
    qs = np.logspace(1,10,1000)
    ax.plot(qs, s*s/2*ratio*np.power(qs, -alpha/(alpha+2)), linestyle='dotted', color=f'C{idx}')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0,
                     box.width, box.height * 0.9])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e2,1e9)
    ax.set_ylabel(r'$\mathcal{L}$')


def plot_all(alphas, n_s = 1):
    fig, axs = plt.subplots(1, 3, figsize=(10,2.5),gridspec_kw={'top':0.95, 'bottom': 0.15})
    plt.subplots_adjust(wspace=0.36)

    #integral constants computed by mathematica
    ratios = [0.68, 0.70, 0.64]

    for i, (alpha,ratio) in enumerate(zip(alphas,ratios)):
        plot_ax(axs[i], alpha, i, n_s, ratio=ratio)
        axs[i].set_xlabel('$C$')
        ps0 = [axs[i].plot([0], [0], color='black', linestyle='dotted')[0]]

    ps = [axs[0].plot([0], [0], color=f'C{i}', linestyle='solid')[0] for i, tit in enumerate(alphas)]
    legend_ = plt.legend(ps, [r'$\alpha=$' + f'{alpha}' for alpha in alphas], ncol=len(alphas), loc='lower center',
                         fontsize=15,
                         columnspacing=1, handlelength=1, bbox_to_anchor=(-1.9, 1.00, 2.0, 0.3), frameon=False)
    fig.add_artist(legend_)

    fig.savefig(f'plot/compute_power_law', dpi=300, bbox_inches='tight')
    fig.savefig(f'plot/compute_power_law.pdf',format="pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--n_s", help="total number of skills", type=int, default=50000)
    args = parser.parse_args()
    plot_all([0.3, 0.6, 0.9], n_s=args.n_s)
