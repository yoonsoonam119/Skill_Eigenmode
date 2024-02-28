import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def theo(x, eigs, lr, s, init):
    c = s/np.power(init,2)-1
    ret = 0
    for eig in eigs:
        ret += np.power(s-s/(1+c*np.exp(-eig*2*s*x*lr)),2)*eig
    return ret/2

if __name__ == '__main__':
    ts = np.array([100,200,300])
    p = 3000
    alpha = 1.3
    losses = []
    losses2 = []
    for i, t in enumerate(ts):
        plt.plot(np.arange(p)+1, theo(t,alpha,p), label=r'$t=$'+f'{t}', linestyle='dashed', color=f'C{i}')
    plt.axhline(1,color='black',linestyle='dotted')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$\mathcal{R}_k$')
    plt.xlabel(r'$k''(skill_idx)')
    plt.legend()
    plt.savefig(f'plot/temporal_learnability{int(alpha*10)}', bbox_inches='tight')
    exit()
