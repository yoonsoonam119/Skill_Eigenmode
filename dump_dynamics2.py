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

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rc('font', **{'size':20})

class FCN(nn.Module):
    def __init__(self, init=0.1):
        super(FCN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(1,10,bias=False), nn.Linear(10,1, bias=False)])
        with torch.no_grad():
            self.layers[0].weight *= 0
            self.layers[0].weight += init
            self.layers[1].weight *= 0
            self.layers[1].weight += init
            torch.nn.init.normal_(self.layers[0].weight, std=init/np.sqrt(10)*1.5)
            torch.nn.init.normal_(self.layers[-1].weight, std=init/np.sqrt(10)*1.5)

    def forward(self, x):
        x = self.layers[1](self.layers[0](x))
        return x

    def get_result(self):
        a1 = self.layers[0].weight.detach().numpy()
        a2 = self.layers[1].weight.detach().numpy()
        return (a2@a1).item()

def create_guassian_data(cnt, batch_size, s):
    x = torch.randn([cnt, 1])
    y = x*s
    dataset = TensorDataset(x,y)
    return DataLoader(dataset, batch_size=batch_size)

def train_dnn(init, lr, epochs, eig, batch_size=1000):
    model = FCN(init=init)
    #learning rate multiplier = 1-momentum WHY WORK?
    optimizer = torch.optim.SGD(model.parameters(), lr=lr*eig, momentum=0.0)
    results = []
    for _ in range(epochs):
        results.append(model.get_result())
        train_loader = create_guassian_data(batch_size, batch_size, s)
        test_loader = create_guassian_data(batch_size, batch_size, s)
        train_loop(model, train_loader, test_loader, optimizer, report=False, epochs=1, criterion=nn.MSELoss(reduction='mean'),
                   m=nn.Identity())
    print(results)
    return results

def iter(s,w, eig, power=2.0, lr=1e-2, error=8e-1):
    #w += 2*lr * eig *np.power(w,power-1)*(s-eig*np.power(w,power))
    w0_grad = 2*lr *( eig *np.power(w[1],power-1)*(s-w[0]*w[1]) + error*np.random.randn(1).item())
    w1_grad = 2*lr *( eig *np.power(w[0],power-1)*(s-w[0]*w[1]) + error*np.random.randn(1).item())
    w += [w0_grad, w1_grad]

def theo(x, eig, lr, s, init):
    c = s/np.power(init,2)-1
    return np.power(1/(1+c*np.exp(-eig*4*s*x*lr)),1)

s = 5
eigs = np.array([0.8, 0.16, 0.04])
#eigs /= np.sum(eigs)
epochs= 100
#init = 4.4e-2
init = 0.1
lr = 1e-1
#for power, linestyle in zip([1.0, 2.0, 3.0], ['dotted', 'dashed', 'solid']):
for power, linestyle in zip([2.0], ['solid']):
    results = [[] for _ in eigs]
    dnn_results = [[] for _ in eigs]

    for i,(eig, result, dnn_result) in enumerate(zip(eigs, results, dnn_results)):
        x = np.arange(epochs)
        #plt.plot(x, np.array(result)/s, linestyle=linestyle, color=f'C{i}', label=rf'$s_{i+1}=$'+f'{eig}')
        plt.plot(x, eig*np.power(1-theo(x, eig, lr, s, init),2), linestyle='solid', color=f'C{i}', label=rf'$s_{i+1}=$'+f'{eig}')
       # plt.plot(x, np.array(dnn_result)/s, linestyle='dotted',color=f'C{i}')

#ps = [plt.plot([0], [0], color='black', linestyle=style)[0] for style in ['dotted', 'dashed', 'solid']]
plt.legend(ncol=len(eigs), loc=(-0.05,1.1), handlelength=1)
plt.ylabel(r'$\mathcal{L}$')
plt.xlabel(r'$t$')
#legend1 = plt.legend(ps, [name for name in ['power = 1', '2', '3']], ncol=3,
#                     columnspacing=1, handlelength=1, bbox_to_anchor=(-0.5, 0.8, 1.0, 0.3))
                     #loc='lower left',
                     #columnspacing=1, handlelength=1, )

#plt.legend(legend1)
plt.savefig(f'plot/loss_dynamics', dpi=300, bbox_inches='tight')
plt.savefig(f'plot/loss_dynamics.pdf', format='pdf', dpi=300, bbox_inches='tight')
