import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


class Loader():
    def __init__(self, bits=16, skill_cnt=3, skill_bit_cnt=2, alpha=2.0, y_scale=3, zero_mean=False, gpu_load=False,
                 skill_mask=None, one_skill_idx=None, large=False, agd=False):
        if large:
            self.skill_key, self.skill_mask = self.get_skill_info_duplicate(bits, skill_cnt, skill_bit_cnt)
        else:
            self.skill_key, self.skill_mask = self.get_skill_info(bits, skill_cnt, skill_bit_cnt)
        if skill_mask is not None:
            self.skill_mask = skill_mask
        self.bits = bits
        self.prob_dist = np.power(np.arange(skill_cnt) + 1, -alpha)
        self.prob_dist /= np.sum(self.prob_dist)
        self.skill_cnt = skill_cnt
        self.y_scale = y_scale
        self.zero_mean = zero_mean
        self.gpu_load = gpu_load
        self.one_skill_idx = one_skill_idx
        self.agd = agd

    def get(self, train_cnt, test_cnt, batch_size=100):
        x = self.do_generate_data(train_cnt + test_cnt, self.bits)
        skill_idxs = self.generate_skill_idxs(train_cnt + test_cnt, self.skill_cnt, self.prob_dist)
        y = np.mod(np.sum(x * (self.skill_mask[skill_idxs]), axis=1), 2)
        if self.one_skill_idx is not None:
            y *= (skill_idxs ==  self.one_skill_idx).astype('float32')
        if self.zero_mean:
            y = 2*(y -0.5)
        #y= np.expand_dims(y,1)
        x = np.concatenate([self.skill_key[skill_idxs], x], axis=1)

        #x_train = torch.split(torch.from_numpy(x[:train_cnt]).float(), batch_size)
        #y_train = torch.split(torch.from_numpy(y[:train_cnt]).float(), batch_size)
        #x_test = torch.split(torch.from_numpy(x[train_cnt:]).float(), batch_size)
        #y_test = torch.split(torch.from_numpy(y[train_cnt:]).float(), batch_size)
        x_train = torch.from_numpy(x[:train_cnt]).float()
        y_train = self.y_scale*torch.from_numpy(y[:train_cnt]).float()
        x_test = torch.from_numpy(x[train_cnt:]).float()
        y_test = self.y_scale*torch.from_numpy(y[train_cnt:]).float()
        if self.gpu_load:
            x_train = x_train.to('cuda')
            x_test = x_test.to('cuda')
            y_train = y_train.to('cuda')
            y_test = y_test.to('cuda')
        if self.agd:
            x_test *= np.sqrt(self.skill_cnt/self.bits+0.5)
            y_test *= np.sqrt(self.skill_cnt/self.bits+0.5)
        train_loader = DataLoader(TensorDataset(x_train, y_train),batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_test, y_test),batch_size=batch_size)
        #return zip(x_train, y_train), zip(x_test, y_test), skill_idxs[:train_cnt], skill_idxs[train_cnt:]
        return train_loader, test_loader, skill_idxs[:train_cnt], skill_idxs[train_cnt:]

    def get_with_skill(self, skill_idx, train_cnt, test_cnt, batch_size=100):
        x = self.do_generate_data(train_cnt + test_cnt, self.bits)
        skill_idxs = (np.ones(train_cnt + test_cnt)*skill_idx).astype('int')
        y = np.mod(np.sum(x * (self.skill_mask[skill_idxs]), axis=1), 2)
        if self.zero_mean:
            y = 2*(y -0.5)
        #y= np.expand_dims(y,1)
        x = np.concatenate([self.skill_key[skill_idxs], x], axis=1)

        #x_train = torch.split(torch.from_numpy(x[:train_cnt]).float(), batch_size)
        #y_train = torch.split(torch.from_numpy(y[:train_cnt]).float(), batch_size)
        #x_test = torch.split(torch.from_numpy(x[train_cnt:]).float(), batch_size)
        #y_test = torch.split(torch.from_numpy(y[train_cnt:]).float(), batch_size)
        x_train =torch.from_numpy(x[:train_cnt]).float()
        y_train =self.y_scale*torch.from_numpy(y[:train_cnt]).float()
        x_test = torch.from_numpy(x[train_cnt:]).float()
        y_test = self.y_scale*torch.from_numpy(y[train_cnt:]).float()
        train_loader = DataLoader(TensorDataset(x_train, y_train),batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(x_test, y_test),batch_size=batch_size)
        #return zip(x_train, y_train), zip(x_test, y_test), skill_idxs[:train_cnt], skill_idxs[train_cnt:]
        return train_loader, test_loader, skill_idxs[:train_cnt], skill_idxs[train_cnt:]

    def do_generate_data(self, cnt, bits=16):
        assert bits%8 == 0
        n_bits = bits//8
        randns = [np.expand_dims(np.random.randint(0,255, cnt),1).astype('uint8') for _ in range(n_bits)]
        #randns = [np.binary_repr(np.random.randint(0,255, cnt)) for _ in range(n_bits)]
        randns = [np.unpackbits(rand, axis=1) for rand in randns]
        x = np.concatenate(randns,axis=1)
        return x

    def generate_skill_idxs(self, cnt, skill_cnt, prob_dist=None):
        if prob_dist is None:
            prob_dist = np.power(np.arange(skill_cnt)+1, -1.0)
            prob_dist /= np.sum(prob_dist)
        skill_idxs = np.random.choice(np.arange(skill_cnt), size=cnt, replace=True, p=prob_dist)
        return skill_idxs

    def get_skill_info(self, bits=16, skill_cnt=3, skill_bit_cnt=2):
        skill_key = np.eye(skill_cnt)
        skill_mask_idx = np.random.choice(np.arange(bits), size=skill_bit_cnt * skill_cnt, replace=False)
        skill_mask_idx = skill_mask_idx.reshape([skill_cnt, skill_bit_cnt])
        skill_mask = np.zeros([skill_cnt, bits])
        for i in range(skill_cnt):
            skill_mask[i][skill_mask_idx[i]] += 1
        return skill_key, skill_mask

    def get_skill_info_duplicate(self, bits=16, skill_cnt=3, skill_bit_cnt=2):
        skill_key = np.eye(skill_cnt)
        a = True
        cnt = 0
        while a:
            cnt += 1
            skill_mask_idx = [np.random.choice(np.arange(bits), size=skill_bit_cnt, replace=False) for _ in range(skill_cnt)]
            t = [str(np.sort(x)) for x in skill_mask_idx]
            a = len(set(t)) != skill_cnt
            if cnt == 100:
                assert 0
        skill_mask_idx = np.stack(skill_mask_idx)
        skill_mask_idx = skill_mask_idx.reshape([skill_cnt, skill_bit_cnt])
        skill_mask = np.zeros([skill_cnt, bits])
        for i in range(skill_cnt):
            skill_mask[i][skill_mask_idx[i]] += 1
        return skill_key, skill_mask

if __name__ == '__main__':
    loader = Loader(bits=16,skill_cnt=5, skill_bit_cnt=3)
    train_loader, test_loader, _, _ = loader.get(20,5, 5)
    print(loader.skill_key)
    print(loader.skill_mask)
    train_loader, test_loader, skill_idx1, skill_idx2 = loader.get_with_skill(0, 20,5, 5)
    print(skill_idx1)
    print(skill_idx2)
    for x, y in train_loader:
        print(x)
        print(y)
        exit()