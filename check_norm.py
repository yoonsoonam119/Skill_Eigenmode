import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from train_package.train2 import train_loop
from data_generator import Loader
import argparse
import os
from executor import Dispensor, RangeSampler, ChopSampler, ListSampler

class FCN(nn.Module):
    def __init__(self, bits=16, middle=1000, out=1, init=0.1, skill_cnt=5, init_fixed=False):
        super(FCN, self).__init__()
        self.bits = bits
        self.nonlin = nn.ReLU()
        #self.nonlin = nn.Tanh()
        self.layers = nn.ModuleList([nn.Linear(bits+skill_cnt,middle), nn.Linear(middle,out)])
        with torch.no_grad():
            if init_fixed:
                torch.nn.init.constant_(self.layers[0].weight, init)
                torch.nn.init.constant_(self.layers[0].bias, init)
                torch.nn.init.constant_(self.layers[-1].weight, init)
                torch.nn.init.constant_(self.layers[-1].bias, init)
            else:
                torch.nn.init.normal_(self.layers[0].weight, std=init)
                torch.nn.init.normal_(self.layers[0].bias, std=init)
                torch.nn.init.normal_(self.layers[-1].weight, std=init)
                torch.nn.init.normal_(self.layers[-1].bias, std=init)

    def forward(self, x):
        x = self.nonlin(self.layers[0](x))
        #x = self.nonlin(self.layers[1](x))
        return self.layers[-1](x).flatten()

def dict2str(bits, skill_cnt, batch_mul, lr,alpha, init, skill_bit_cnt, y_scale):
    return '_'.join([str(bits), str(skill_cnt), str(batch_mul), str(lr).replace('.',''), str(alpha).replace('.',''),
                     str(init).replace('.',''), str(skill_bit_cnt), str(y_scale)])

def check_init_vals(skill_cnt, load_creator, model, optimizer):
    init_vals = []
    for i in range(skill_cnt):
        train_loader, test_loader, _, _ = load_creator.get_with_skill(skill_idx=i, train_cnt=100, test_cnt=20000,
                                                                      batch_size=20000)
        for x, y in test_loader:
            outs = model(x).detach().numpy()
            outs = (outs - np.mean(outs))/np.sqrt(20000)
            y = y.detach().numpy()
            y = (y-np.mean(y))/np.sqrt(20000)
            y /= np.linalg.norm(y)
            init_vals.append(np.abs(np.inner(y, outs).item()))
            #init_vals.append(np.power(np.inner(outs, outs).item(), 0.5))
        #_, _, _, skill_loss = train_loop(model, train_loader, test_loader, optimizer, report=False, epochs=0,
        #                                 criterion=nn.MSELoss(), m=nn.Identity())
        #losses.append(skill_loss)
    #train_loader, test_loader, _, _ = load_creator.get(train_cnt=100, test_cnt=20000, batch_size=20000)
    return init_vals

def run(bits, skill_cnt=5, batch_mul=200, lr=0.001,alpha=2.0, skill_bit_cnt=3, init=0.1, y_scale=3):
    load_creator = Loader(bits=bits,skill_cnt=skill_cnt,skill_bit_cnt=skill_bit_cnt, alpha=alpha, y_scale=y_scale)
    model = FCN(bits=bits, init=init, skill_cnt=skill_cnt)
    train_loader, test_loader, _, _ = load_creator.get(train_cnt=20000, test_cnt=1000, batch_size=20000)
    #train_loader, test_loader, _, _ = load_creator.get_with_skill(skill_idx=0, train_cnt=100, test_cnt=10000,
    #                                                              batch_size=10000)
    for x, y in train_loader:
        outs = model(x).detach().numpy()
    init_vals = check_init_vals(skill_cnt, load_creator, model,_)

    return np.mean(outs), np.std(outs), init_vals

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="outputpath", type=str, default=os.getcwd())
    parser.add_argument("-n", "--worker_cnt", help="worker_cnt", type=int, default=1)
    parser.add_argument("-i", "--iter_dict", help="do_iteration", type=int, default=0)
    args = parser.parse_args()

    d = Dispensor(args.worker_cnt, dir=args.output, single_mode=args.worker_cnt == 1)
    d.add(ListSampler([5,10,20]), 'batch_mul')
    d.add(ListSampler([0.05,0.1, 0.2]), 'lr')
    d.add(ListSampler([0.01, 0.05, 0.1]), 'init')
    d.add(ListSampler([1, 3, 5]), 'y_scale')
    d.add(ListSampler([1.0, 1.5, 2.0]), 'alpha')
    #d.add(ListSampler([10]), 'batch_mul')
    #d.add(ListSampler([0.1]), 'lr')
    #d.add(ListSampler([0.05]), 'init')
    #d.add(ListSampler([5]), 'y_scale')
    #d.add(ListSampler([2.0]), 'alpha')
    # d.add(ListSampler([10,50,100,500,1000]), 'batch_size')
    try_cnt = 5

    mean_plots = []
    std_plots = []
    init_plots = [[] for _ in range(5)]
    init_stds = [[] for _ in range(5)]
    inits = np.logspace(-2,-1,4)

    for init in inits:
        print(init)
        param_dict = {'bits': 16, 'skill_cnt': 5, 'batch_mul': 10, 'lr':0.1,'alpha':2.0, 'init': init, 'skill_bit_cnt':3, 'y_scale': 5}
        dict_str = dict2str(**param_dict)
        mean_arr = []
        std_arr = []
        init_vals_arr = []
        for _ in range(try_cnt):
            mean, std, init_vals = run(**param_dict)
            mean_arr.append(mean)
            std_arr.append(std)
            init_vals_arr.append(init_vals)
        init_stds_arr = np.power(np.std(np.abs(np.array(init_vals_arr)),axis=0),1)
        init_vals_arr = np.power(np.mean(np.abs(np.array(init_vals_arr)),axis=0),1)
        mean_arr = np.array(mean_arr)
        std_arr = np.array(std_arr)
        print(np.mean(mean_arr))
        print(np.mean(std_arr))
        mean_plots.append(np.abs(np.mean(mean_arr)))
        std_plots.append(np.mean(std_arr))
        for i in range(5):
            init_plots[i].append(init_vals_arr[i])
            init_stds[i].append(init_stds_arr[i])
    #plt.plot(inits, mean_plots)
    #plt.plot(inits, std_plots)
    #plt.plot(inits, 40*np.power(inits,2), linestyle='dashed')
    plt.plot(inits, 0.3*np.power(inits,2), linestyle='dashed', label=r'0.3$\sigma^2$')
    #plt.plot(inits, 400*np.power(inits,1.5), linestyle='dotted')
    print(init_plots)
    for i in range(5):
        plt.errorbar(inits, np.array(init_plots[i]), np.array(init_stds[i]), label=f'k={i}')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel(r'$\sigma^2$', fontdict={'size':15})
    plt.ylabel(r'$|\mathcal{R}_k(0)|$', fontdict={'size':15})
    plt.savefig('plot/norm_plot')
    plt.savefig(f'plot/norm_plot.pdf', format='pdf', dpi=300)
