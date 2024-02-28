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

if __name__ == '__main__':
    ns = np.array([100,200,300,400])
    alpha = 1.5
    kappas = [get_gen(n,2000,alpha,100) for n in ns]
    plt.plot(ns,kappas)
    plt.plot(ns,kappas[0]/np.power(ns[0],-1.0) * np.power(ns, -1.0))
    plt.plot(ns,kappas[0]/np.power(ns[0],-alpha)* np.power(ns, -alpha))
    plt.plot(ns,kappas[0]/np.power(ns[0],-(alpha-1)/alpha)* np.power(ns, -(alpha-1)/alpha))
    plt.plot(ns,kappas[0]/np.power(ns[0],-(alpha-1))* np.power(ns, -(alpha-1)))
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('plot/kappa')
