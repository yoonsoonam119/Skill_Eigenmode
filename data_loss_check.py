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

def threshold_cnt(idxs, threshold=1):
    items = set(idxs)
    d = {}
    for elem in items:
        d[elem] = 0
    for i in idxs:
        d[i] += 1

    ret = []
    for k,v in d.items():
        if v >= threshold:
            ret.append(k)
    return np.array(ret)

def get_discrete(n, p, alpha, repeats=100, threshold=0):
    eigs = np.power(np.arange(p) + 1, -float(alpha))
    eigs /= np.sum(eigs)
    probs = np.zeros(p)
    loss = 0
    for _ in range(repeats):
        idxs = np.random.choice(np.arange(p), n, replace=True, p=eigs)
        if threshold == 0:
            idxs = np.array(list(set(idxs)))
        else:
            idxs = threshold_cnt(idxs, threshold)
        probs[idxs] +=1
        l = np.ones(p)
        l[idxs] -= 1
        loss += np.sum(l*eigs)

    probs /= repeats
    loss /= repeats
    return probs, loss

if __name__ == '__main__':
    ns = np.array([100,200,300,400,500,1000,2000,3000,4000,5000])
    #ns = np.array([100,200,300,400,500,1000])
    alphas = [1.3, 1.5, 1.9, 2.8]
    #alphas = [1.3]
    p = 50000
    for i, alpha in enumerate(alphas):
        losses = []
        losses2 = []
        for n in ns:
            probs, loss = get_discrete(n,p, alpha, 1000)
            losses.append(loss)
        plt.plot(ns, losses/losses[0], label='empirical ' + rf'$alpha+1={alpha}$', color=f'C{i}')
        plt.plot(ns, np.power(ns, -(alpha-1)/alpha)/np.power(ns[0], -(alpha-1)/alpha), linestyle='dashed',color= f'C{i}',
                 label='power law: ' +r'$k^{-\alpha/(\alpha+1)}$')
        #plt.plot(ns,losses[0]/np.power(ns[0],-(alpha-1))* np.power(ns, -(alpha-1)))
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('n(datapoint)')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('plot/data_loss', bbox_inches='tight')
