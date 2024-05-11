import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import argparse
import mpmath
import string

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

def param_const(S, init, alpha, zeta):
    const = 1/(alpha * zeta)
    print('param:', const)
    return const * np.power(S,2)/2

def data_const(S, init, alpha, zeta):
    const1 = mpmath.gamma(alpha/(alpha+1))
    const1 /= (alpha+1)*np.power(zeta,1/(alpha+1))
    const2 = 0
    const = const1+const2
    print('data:', const)
    return const * np.power(S-np.power(init,2),2)/2

def time_const(S, init, alpha, zeta):
    lr = 1
    #integral constants calculated by Mathematica
    integral_consts = {0.3: 1.29, 0.6:1.20, 0.9:1.06}
    const = integral_consts[alpha]
    const /= np.power(lr*S, alpha/(alpha+1))
    print('time: ', const)
    return const * np.power(S,2)/2

def data_power(x, S, init, alpha, zeta):
    return data_const(S, init, alpha, zeta)*np.power(x,-(alpha)/(alpha+1))

def temp_power(x, S, init, alpha, zeta):
    return time_const(S, init, alpha, zeta)*np.power(x,-(alpha)/(alpha+1))

def param_power(x, S, init, alpha, zeta):
    return param_const(S, init, alpha, zeta)*np.power(x,-alpha)

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


def plot_ax(ax, theo_func, power_func, name, probss, alphas, zetas, idx, init, s=1, max_range=10000):
    ts = np.arange(1,max_range)
    for i, (probs, zeta) in enumerate(zip(probss,zetas)):
        ax.plot(ts, theo_func(ts, probs,s=s), color=f'C{i}')
        ax.plot(ts, power_func(ts, s, init, alphas[i], zeta), color=f'C{i}', linestyle='dotted')
    ax.set_title(latex_transform(f'{name} scaling',idx,alp=True), y=-0.4)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0,
                     box.width, box.height * 0.9])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_ylabel(r'$\mathcal{L}$')


def plot_all(alphas, n_s = 1, init=0.1):
    fig, axs = plt.subplots(1, 3, figsize=(10,2.5),gridspec_kw={'top':0.95, 'bottom': 0.15})
    plt.subplots_adjust(wspace=0.36)

    probss = []
    zetas = []
    for alpha in alphas:
        probs = np.power(np.arange(n_s) + 1, -float(alpha)-1)
        zetas.append(np.sum(probs))
        probs /= zetas[-1]
        probss.append(probs)

    plot_ax(axs[0], temp_theo, temp_power, 'Time', probss, alphas, zetas, 0, init=init, max_range=10000)
    axs[0].set_xlabel('$T$')
    ps0 = [axs[0].plot([0], [0], color='black', linestyle='dotted')[0]]
    legend0 = plt.legend(ps0, [r'$\mathcal{L} = \mathcal{A}_T T^{-\alpha/(\alpha+1)}$'],bbox_to_anchor=(-1.875, 0.3),
                         handlelength=1, frameon=False)
    legend00 = plt.legend(ps0, [r'$D,N\rightarrow \infty$'],bbox_to_anchor=(-2.1, 0.4),
                         handlelength=0, frameon=False)
    fig.add_artist(legend0, )
    fig.add_artist(legend00 )

    plot_ax(axs[1], data_theo, data_power, 'Data', probss, alphas, zetas, 1, init=init, max_range= 1000)
    axs[1].set_xlabel('$D$')
    ps1 = [axs[1].plot([0], [0], color='black', linestyle='dotted')[0]]
    legend1 = plt.legend(ps1, [r'$\mathcal{L} =\mathcal{A}_D D^{-\alpha/(\alpha+1)}$'],bbox_to_anchor=(-0.5, 0.3),
                         handlelength=1, frameon=False)
    legend10 = plt.legend(ps1, [r'$N,T\rightarrow \infty$'],bbox_to_anchor=(-0.77, 0.4),
                          handlelength=0, frameon=False)
    fig.add_artist(legend1, )
    fig.add_artist(legend10 )

    plot_ax(axs[2], param_theo, param_power, 'Parameter', probss, alphas, zetas, 2, init=init, max_range= 100)
    axs[2].set_xlabel('$N$')
    ps2 = [axs[2].plot([0], [0], color='black', linestyle='dotted')[0]]
    legend2 = plt.legend(ps2, [r'$\mathcal{L} =\mathcal{A}_N N^{-\alpha}$'],bbox_to_anchor=(0.66, 0.3),
                         handlelength=1, frameon=False)
    legend20 = plt.legend(ps2, [r'$T,D\rightarrow \infty$'],bbox_to_anchor=(0.6, 0.4),
                          handlelength=0, frameon=False)
    fig.add_artist(legend2, )
    fig.add_artist(legend20 )

    ps = [axs[0].plot([0], [0], color=f'C{i}', linestyle='solid')[0] for i, tit in enumerate(alphas)]
    legend_ = plt.legend(ps, [r'$\alpha=$' + f'{alpha}' for alpha in alphas], ncol=len(alphas), loc='lower center',
                         fontsize=13,
                         columnspacing=1, handlelength=1, bbox_to_anchor=(-1.9, 1.00, 2.0, 0.3), frameon=False)
    fig.add_artist(legend_)

    fig.savefig(f'plot/power_law_all', dpi=300, bbox_inches='tight')
    fig.savefig(f'plot/power_law_all.pdf',format="pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--n_s", help="total number of skills", type=int, default=100000)
    args = parser.parse_args()
    plot_all([0.3 ,0.6, 0.9], n_s = args.n_s)
