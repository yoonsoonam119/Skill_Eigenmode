import numpy as np
import os
import torch
import torch.nn as nn

def arr2str(arr):
    return ' '.join([str(x) for x in arr])

def dict2str(bits, skill_cnt, batch_mul, lr,alpha, init, skill_bit_cnt, y_scale, opt=None, act='relu'):
    if opt in ['sgd',None]:
        opt = []
    else:
        opt = [opt]

    act = [act] if act != 'relu' else []
    return '_'.join([str(bits), str(skill_cnt), str(batch_mul), str(lr).replace('.',''), str(alpha).replace('.',''),
                     str(init).replace('.',''), str(skill_bit_cnt), str(y_scale)] + opt + act)

def write2file(type_name, variable, skill_loss, corr, te_loss, parameters, zero=True):
    dict_str = dict2str(**parameters)
    assert type_name in ['time', 'data', 'parameter']
    zero_mean_str = 'zero' if zero else ''
    print('FILE I/O:', type_name, variable, dict_str, flush=True)
    if type_name == 'time':
        # repeat X Timesteps X skill_cnt
        assert len(skill_loss) > 1
        assert len(skill_loss[0][0]) == parameters['skill_cnt']
        assert len(corr[0][0]) == parameters['skill_cnt']
        np.save(f'data/{zero_mean_str}/time/skill_losses_{dict_str}', np.array(skill_loss))
        np.save(f'data/{zero_mean_str}/time/corr_{dict_str}', np.array(corr))
        np.save(f'data/{zero_mean_str}/time/te_loss_{dict_str}', np.array(te_loss))
    else:
        assert len(skill_loss) == parameters['skill_cnt']
        assert len(corr) == parameters['skill_cnt']
        skill_loss = arr2str(skill_loss)
        corr = arr2str(corr)
        te_loss = str(te_loss)
        with open(f'data/{zero_mean_str}/{type_name}/skill_losses_{variable}_{dict_str}', 'a') as f:
            print(skill_loss, file=f, flush=True)
        with open(f'data/{zero_mean_str}/{type_name}/corr_{variable}_{dict_str}', 'a') as f:
            print(corr, file=f, flush=True)
        with open(f'data/{zero_mean_str}/{type_name}/te_loss_{variable}_{dict_str}', 'a') as f:
            print(te_loss, file=f, flush=True)

def file2mem(type_name, variables, parameters, zero=True):
    dict_str = dict2str(**parameters)
    assert type_name in ['time', 'data', 'parameter']
    zero_mean_str = 'zero' if zero else ''
    if type_name == 'time':
        # repeat X Timesteps X skill_cnt
        skill_losses = np.load(f'data/{zero_mean_str}/time/skill_losses_{dict_str}.npy')
        corrs = np.load(f'data/{zero_mean_str}/time/corr_{dict_str}.npy')
        te_losses = np.load(f'data/{zero_mean_str}/time/te_loss_{dict_str}.npy')

        skills_mean = np.mean(skill_losses,axis=0).T
        skills_std = np.std(skill_losses,axis=0).T
        corrs_mean = np.mean(corrs,axis=0).T
        corrs_std = np.std(corrs,axis=0).T
        te_mean = np.mean(te_losses,axis=0)
        te_std = np.std(te_losses,axis=0)

        # skill_cnt X timesteps
        return corrs_mean, corrs_std, skills_mean, skills_std, te_mean, te_std
    else:
        skills_mean = []
        skills_std = []
        corrs_mean = []
        corrs_std = []
        te_mean = []
        te_std = []
        for variable in variables:
            print('FILE I/O:', type_name, variable, dict_str, flush=True)
            with open(f'data/{zero_mean_str}/{type_name}/skill_losses_{variable}_{dict_str}', 'r') as f:
                skills = np.array([[float(x) for x in line.split()] for line in f])
                skills_mean.append(np.mean(skills,axis=0))
                skills_std.append(np.std(skills,axis=0))
            with open(f'data/{zero_mean_str}/{type_name}/corr_{variable}_{dict_str}', 'r') as f:
                corrs = np.array([[float(x) for x in line.split()] for line in f])
                corrs_mean.append(np.mean(corrs, axis=0))
                corrs_std.append(np.std(corrs, axis=0))
            with open(f'data/{zero_mean_str}/{type_name}/te_loss_{variable}_{dict_str}', 'r') as f:
                te = np.array([[float(x) for x in line.split()] for line in f])
                te_mean.append(np.mean(te, axis=0).item())
                te_std.append(np.std(te, axis=0).item())

        return np.stack(corrs_mean).T, np.stack(corrs_std).T, np.stack(skills_mean).T, np.stack(skills_std).T, np.array(te_mean), np.array(te_std)

class FCN(nn.Module):
    def __init__(self, bits=16, middle=1000, out=1, init=0.1, skill_cnt=5, init_fixed=False, act='relu'):
        super(FCN, self).__init__()
        self.bits = bits
        if act == 'relu':
            self.nonlin = nn.ReLU()
        elif act == 'tanh':
            self.nonlin = nn.Tanh()
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

def check_corrs(model, tr_loaders, te_loaders, cnt=20000):
    corrs = []
    skill_losses = []
    with torch.no_grad():
        for i, (train_loader, test_loader) in enumerate(zip(tr_loaders, te_loaders)):
            #train_loader, test_loader, _, _ = load_creator.get_with_skill(skill_idx=i, train_cnt=100, test_cnt=20000,
            #                                                              batch_size=20000)
            for x, y in test_loader:
                outs = model(x).detach().numpy()
                y = y.detach().numpy()
                skill_losses.append(np.mean(np.power(y-outs,2)).item())
                outs = (outs - np.mean(outs))/np.sqrt(cnt)
                y = (y-np.mean(y))/np.sqrt(cnt)
                #print(i, np.inner(y,y))
                y /= np.linalg.norm(y)
                #print(i, np.inner(y,y))
                #print(i, np.inner(outs,outs))
                #print(i, np.inner(y, outs))
                corrs.append(np.inner(y, outs).item())
    print('corrs: ', corrs)
    print('skill losses: ', skill_losses)
    return corrs, skill_losses

def train(model, train_loader, optimizer, criterion, m):
    tr_loss = 0
    tr_acc = 0
    cnt = 0
    for x, y in train_loader:
        model.zero_grad()
        logit = m(model(x))
        loss = criterion(logit,y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            tr_acc += torch.sum(logit * y > 0).detach().item()
            tr_loss += loss.detach().item() *len(y)
            cnt += len(y)
    tr_loss /= cnt
    tr_acc /= cnt
    return tr_acc, tr_loss

def test(model, test_loader, criterion, m):
    te_loss = 0
    te_acc = 0
    cnt = 0
    with torch.no_grad():
        for x, y in test_loader:
            logit = m(model(x))
            loss = criterion(logit,y)
            te_acc += torch.sum(logit * y > 0).detach().item()
            te_loss += loss.detach().item() *len(y)
            cnt += len(y)
    te_loss /= cnt
    te_acc /= cnt
    return te_acc, te_loss

def train_loop(model, train_loader, test_loader, optimizer, report=False, epochs=1,
               criterion=nn.MSELoss(), m=nn.Identity(), verbose=True):
    for i in range(epochs):
        tr_acc, tr_loss = train(model, train_loader, optimizer, criterion, m)
        te_acc, te_loss = test(model, test_loader, criterion, m)
        if verbose:
            print('train acc/loss', tr_acc, tr_loss)
            print('test acc/loss', te_acc, te_loss)
    if epochs == 0:
        tr_acc, tr_loss = train(model, train_loader, criterion, m)
        te_acc, te_loss = test(model, test_loader, criterion, m)
        if verbose:
            print('train acc/loss', tr_acc, tr_loss)
            print('test acc/loss', te_acc, te_loss)
    return tr_acc,te_acc, tr_loss, te_loss