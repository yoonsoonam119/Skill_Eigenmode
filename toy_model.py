import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.cm as cm
from train_package.train2 import train_loop
import matplotlib.collections as mcoll
from train_package.train2 import train_loop
from torch.utils.data import Dataset, DataLoader, TensorDataset

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rc('font', **{'size':20})

class Toy(nn.Module):
    def __init__(self, init=0.1, p=100):
        super(Toy, self).__init__()
        self.layer1 = nn.Parameter(torch.ones(p)*init)
        self.layer2 = nn.Parameter(torch.ones(p)*init)
        self.p = p

    def forward(self, x):
        x = x[:,:self.p]
        x = torch.sum(self.layer1*self.layer2*x, dim=1)
        return x

    def get_results(self):
        with torch.no_grad():
            return (self.layer1*self.layer2).detach().flatten().numpy()

def create_guassian_data(cnt, batch_size, s):
    x = torch.randn([cnt, 1])
    y = x*s
    dataset = TensorDataset(x,y)
    return DataLoader(dataset, batch_size=batch_size)

def create_parity_data(cnt, n_skills, p, eigs, scale, batch_size=None, check_appearance=True):
    if batch_size is None:
        batch_size = cnt
    arr = np.random.choice(np.arange(n_skills),size=cnt,p=eigs)
    if p > n_skills:
        arr = np.concatenate(arr, np.zeros(cnt,p-n_skills),axis=1)
    # arr = cnt X max(n_skills, p)
    # model can handle p < n_skills case

    y = np.random.randint(0,2,cnt)*np.sqrt(2)
    # sqrt(2) is needed for the of g_k(i,x) to be equal to 1
    # Note that we have outputs 0, \sqrt(2)*y_scale with equal chances.

    appearance = np.array(list(set(arr * y))) if check_appearance else None

    #convert into torch format
    y = torch.from_numpy(y).float()
    x = torch.nn.functional.one_hot(torch.from_numpy(arr).long(), n_skills).float()
    x *= y.unsqueeze(1)
    y *= scale

    return DataLoader(TensorDataset(x.float(), y), batch_size=batch_size), appearance

def loss_from_rk(eigs,rk,y_scale, init=0.1):
    if len(rk) < len(eigs):
        rk = np.concatenate([rk, np.ones(len(eigs)-len(rk))*np.power(init,2)])
    return 1/2*np.sum(np.power(y_scale-rk,2)*eigs)

def one_loop(model, loader, opt, criterion, device='cpu'):
    for x, y in loader:
        opt.zero_grad()
        loss = criterion(model(x.to(device)),y)/2
        loss.backward()
        opt.step()

def train_multi_epoch( lr=0.5, alpha=1.5, p=100, cnt=1000, p_task=100, epochs=100, init=0.1, y_scale=5, opt='adam', verb=False):
    eigs = np.power(np.arange(p_task) +1, -alpha)
    eigs = eigs/np.sum(eigs)
    model = Toy(init=init)
    if opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')
    results = []
    train_loader, appearance = create_parity_data(cnt, p_task, p, eigs, y_scale, batch_size=100)
    for _ in range(epochs):
        if verb:
            print(model.get_results())
            print(loss_from_rk(eigs,model.get_results(), y_scale))
        one_loop(model, train_loader, optimizer, criterion)
    print(loss_from_rk(eigs, model.get_results(), y_scale))
    exit()
    return results

def train_single_epoch( lr=0.1, alpha=1.5, p=100, p_task=100, cnt=10000, s=None, epochs=100, init=0.1, y_scale=5, verb=False, opt='sgd'):
    eigs = np.power(np.arange(p_task) +1, -alpha)
    eigs = eigs/np.sum(eigs)
    model = Toy(init=init,p=p)
    if opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')
    losses = []
    for _ in range(epochs):
        train_loader, appearance = create_parity_data(cnt, p_task, p, eigs, y_scale, batch_size=10000, check_appearance=False)
        if verb:
            print(model.get_results())
            print(loss_from_rk(eigs,model.get_results(), y_scale))
        losses.append(loss_from_rk(eigs,model.get_results(), y_scale, init))
        one_loop(model, train_loader, optimizer, criterion)
    return losses

def iter(s,w, eig, power=2.0, lr=1e-2, error=8e-1):
    #w += 2*lr * eig *np.power(w,power-1)*(s-eig*np.power(w,power))
    w0_grad = 2*lr *( eig *np.power(w[1],power-1)*(s-w[0]*w[1]) + error*np.random.randn(1).item())
    w1_grad = 2*lr *( eig *np.power(w[0],power-1)*(s-w[0]*w[1]) + error*np.random.randn(1).item())
    w += [w0_grad, w1_grad]

if __name__ == '__main__':
    #train_single_epoch(verb=True, p=10,p_task=1000,opt='adam')
    train_multi_epoch(verb=False, p=10000,p_task=10000,opt='adam')
    exit()
    #train_multi_epoch()
    #eigs = np.power(np.arange(10) + 1,- 1.0)
    #create_parity_data(100, 10, 10, eigs, 3,5)
    s = 5
    alpha = 1.5
    epochs= 10000
    #init = 4.4e-2
    init = 2
    lr = 7e-3
    #for power, linestyle in zip([1.0, 2.0, 3.0], ['dotted', 'dashed', 'solid']):
    for i, alpha in enumerate([0.3, 0.6, 0.9]):
        eigs /= np.sum(eigs)
        x = np.arange(epochs)+1
        #plt.plot(x, 40*np.power(x,-alp), linestyle='solid')
        plt.plot(x, np.power(s-np.power(init,2),2)*np.power(x,-(alpha)/(alpha+1)), linestyle='dashed', color=f'C{i}')
        plt.plot(x, theo(x, eigs, lr, s, init), linestyle='solid', color=f'C{i}', label=r'$\alpha=$'+f'{alpha}')

    #ps = [plt.plot([0], [0], color='black', linestyle=style)[0] for style in ['dotted', 'dashed', 'solid']]
    plt.legend(ncol=len(eigs), loc=(-0.05,1.1), handlelength=1)
    plt.ylabel(r'$Loss$')
    plt.xlabel(r'$t$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([1,10,100,1000,10000])

    #plt.legend(legend1)
    plt.savefig(f'plot/time_learning_curve_slow', dpi=300, bbox_inches='tight')
    plt.savefig(f'plot/time_learning_curve_slow.pdf', format='pdf', dpi=300, bbox_inches='tight')
