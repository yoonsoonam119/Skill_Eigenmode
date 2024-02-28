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
from toy_model import train_single_epoch

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

def theo(ps, eigs):
    ret = [np.power(s,2)/2 * np.sum(eigs[p+1:]) for p in ps]
    return ret
s = 5
alpha = 1.5
epochs= 100
#init = 4.4e-2
init = 0.1
lr = 5e-2
p_task = 10000
#for power, linestyle in zip([1.0, 2.0, 3.0], ['dotted', 'dashed', 'solid']):
for i, alpha in enumerate([0.3, 0.5, 0.7]):
    eigs = np.power(np.arange(p_task) + 1, -alpha-1.0)
    eigs /= np.sum(eigs)
    ps = np.array([10,20,30,50,100,200,300,500])
    losses = []
    for p in ps:
        loss = train_single_epoch(lr=lr, alpha=alpha+1.0, p=p, p_task=p_task, cnt=10000, s=None, epochs=epochs, init=init, y_scale=s, opt='adam')
        losses.append(loss[-1])
        #plt.plot(x, 40*np.power(x,-alp), linestyle='solid')
        #plt.plot(x, np.power(s-np.power(init,2),2)*np.power(x,-(alpha)/(alpha+1)), linestyle='dashed', color=f'C{i}')
    plt.plot(ps, losses, linestyle='dotted', color=f'C{i}', linewidth=2)
    plt.plot(ps, theo(ps, eigs), linestyle='solid', color=f'C{i}', label=r'$\alpha=$'+f'{alpha}')
    plt.plot(ps, np.power(s,2)/2/(1+alpha)*np.power(ps,-alpha), linestyle='dashed', color=f'C{i}')

#ps = [plt.plot([0], [0], color='black', linestyle=style)[0] for style in ['dotted', 'dashed', 'solid']]
plt.legend(ncol=len(eigs), loc=(-0.05,1.1), handlelength=1)
plt.ylabel(r'$Loss$')
plt.xlabel(r'$P$')
plt.xscale('log')
plt.yscale('log')
#plt.xticks([1,10,100,1000,10000])

#plt.legend(legend1)
plt.savefig(f'plot/simul_parameter_learning_curve', dpi=300, bbox_inches='tight')
plt.savefig(f'plot/simul_parameter_learning_curve.pdf', format='pdf', dpi=300, bbox_inches='tight')
