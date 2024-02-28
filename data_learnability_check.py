import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rc('font', **{'size':20})

def get_info(n,p, alpha, repeats = 100):
    eigs = np.power(np.arange(p)+1, -float(alpha))
    M = 0
    for _ in range(repeats):
        X = np.random.randn(n,p)
        u,s,v = np.linalg.svd(X * eigs, full_matrices=False)
        M += v.T@v
    M /= repeats
    L = np.diag(M)
    kappa = eigs/L - eigs
    print(L)
    kappas = []
    hit=0
    for idx, val in enumerate(L):
        if val < 0.95 and val> 0.06:
            kappas.append( eigs[idx]/L[idx]-eigs[idx])
        if val < 0.505 and val>0.495:
            hit = eigs[idx]

    kappa = np.mean(np.array(kappas))
    return eigs, L, kappa

def get_kappa(n, p, alpha, repeats=100):
    _, _, kappa = get_info(n, p, alpha, repeats=repeats)
    return kappa

def get_gen(n, p, alpha, repeats=100):
    eigs, L, kappa = get_info(n, p, alpha, repeats=repeats)
    return np.sum(np.power(1-L,2)*eigs)/(1-1/n*np.sum(np.power(L,2)))

def gen_from_L(n, L, alpha):
    eigs = np.power(np.arange(p)+1, -float(alpha))
    eigs /= np.sum(eigs)
    return np.sum(np.power(1 - L, 2) * eigs) / (1 - 1 / n * np.sum(np.power(L, 2)))

def get_discrete(n, p, alpha, repeats=100):
    eigs = np.power(np.arange(p) + 1, -float(alpha))
    eigs /= np.sum(eigs)
    probs = np.zeros(p)
    loss = 0
    for _ in range(repeats):
        idxs = set(np.random.choice(np.arange(p), n, replace=True, p=eigs))
        probs[np.array(list(idxs))] +=1
        l = np.ones(p)
        l[np.array(list(idxs))] -= 1
        loss += np.sum(l*eigs)

    probs /= repeats
    loss /= repeats
    return probs, loss

def power(n, p, alpha, repeats=100):
    eigs = np.power(np.arange(p) + 1, -float(alpha))
    eigs /= np.sum(eigs)
    return eigs

def theo(n, p, alpha, repeats=100):
    eigs = np.power(np.arange(p) + 1, -float(alpha))
    eigs /= np.sum(eigs)
    return 1-np.power((1-eigs),n)

if __name__ == '__main__':
    ns = np.array([100,200,300])
    p = 3000
    alpha = 1.5
    losses = []
    losses2 = []
    fig, axs = plt.subplots()
    ps = [axs.plot([0], [0], color=f'black', linestyle='dotted')[0]]
    legend0 = plt.legend(ps, [r'$Dk^{-(\alpha+1)}$'], ncol=1, loc='lower center',
                         columnspacing=1, handlelength=1, bbox_to_anchor=(0.8, 0.55), frameon=False)
    ps1 = [axs.plot([0], [0], color=f'black', linestyle='dotted')[0]]
    legend1 = plt.legend(ps1, [r'$k_D = D^{1/(\alpha+1)}$'], ncol=1, loc='lower center',
                         columnspacing=1, handlelength=0, bbox_to_anchor=(0.5, 0.8), frameon=False)
    fig.add_artist(legend0)
    fig.add_artist(legend1)
    for i, n in enumerate(ns):
        probs, loss = get_discrete(n,p, alpha, 10000)
        probs_theo = theo(n,p, alpha, 1000)
        #plt.plot(np.arange(p)+1, probs, label='empirical ' + rf'$n={n}$', color=f'C{i}')
        plt.plot(np.arange(p)+1, theo(n,p,alpha), label=r'$D=$'+f'{int(n)}', linestyle='solid', color=f'C{i}')
        plt.plot(np.arange(p)+1, n*power(n,p,alpha), linestyle='dotted', color=f'C{i}')
        plt.scatter([np.power(n/np.sum(np.power(np.arange(p) + 1, -float(alpha))), 1/alpha)],[1], color=f'C{i}')
        losses.append(loss)
        #losses2.append(loss2)
    plt.axhline(1,color='black',linestyle='dashed')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$P_{learn}(k)$')
    plt.xlabel(r'$k$')
    plt.ylim(1e-4,1e1)
    plt.legend(handlelength=1, loc='lower left')

    #legend1 = plt.legend(ps, [title for title in titles], ncol=len(titles), loc=(-0.55,1.1))
    plt.savefig(f'plot/data_learnability{int(alpha*10)}', bbox_inches='tight')
    plt.savefig(f'plot/data_learnability{int(alpha*10)}.pdf',dpi=300, format='pdf', bbox_inches='tight')
    exit()
