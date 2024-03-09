from sampler import *
from collections import OrderedDict
import os, sys
import shutil

class Dispenser:
    def __init__(self, worker_cnt, dir=os.getcwd(), name='data/name', single_mode=False):
        self.worker_cnt = worker_cnt

        try:
            from mpi4py import MPI
        except ImportError:
            assert single_mode

        if not single_mode:
            comm = MPI.COMM_WORLD
            self.cur_worker = comm.Get_rank()
            print('worker:', self.cur_worker)
            if self.cur_worker == 0:
                cur_exec_filename = __file__.split('/')[-1]

                if cur_exec_filename not in os.listdir(dir):
                    print(f'copying file: {cur_exec_filename}')
                    shutil.copy(__file__, dir)
        else:
            self.cur_worker = 0
        name += f'_{self.cur_worker}'
        self.f = open(os.path.join(dir, name), 'w')
        #self.cur_worker=0
        self.cur_elem = self.cur_worker - self.worker_cnt
        self.samplers = OrderedDict()
        self.idx_shape = []
        self.tot_cnt = 0
        self.cur = {}

    def add(self, sampler, name=None):
        if name is None:
            name = len(self.samplers)
        self.samplers[name]=sampler
        self.idx_shape.append(sampler.cnt)
        if self.tot_cnt == 0:
            self.tot_cnt = 1
        self.tot_cnt *= sampler.cnt
        self.idx_ndarr = np.arange(0,self.tot_cnt)
        self.idx_ndarr.shape = self.idx_shape

    def __iter__(self):
        return self

    def __next__(self):
        self.cur_elem += self.worker_cnt
        if self.cur_elem >= self.tot_cnt:
            raise StopIteration
        #idxs = [int(elem) for elem in np.where(self.idx_ndarr == self.cur_elem)]
        idxs = self.cur_idx2idxs(self.cur_elem)
        self.cur = OrderedDict()
        for (name, sampler), idx in zip(self.samplers.items(), idxs):
            self.cur[name] = sampler.get(idx)
        return self.cur

    def cur_idx2idxs(self, idx):
        idx_arr = []
        div = np.prod(np.array(self.idx_shape))
        for i in self.idx_shape:
            div = div//i
            idx_arr.append(idx//div)
            idx = idx % div
        return idx_arr

    def print_(self, *args, novalues=False, verbose=False, report=False):
        if verbose:
            print(f"printing to {self.f}")

        if novalues and not report:
            print(*args, file=self.f, flush=True)
        elif report:
            #args is the report
            if novalues:
                for arg in args:
                    print(*arg, file=self.f, flush=True)
            else:
                for arg in args:
                    print(*self.cur.values(), *arg, file=self.f, flush=True)
        else:
            print(*self.cur.values(), *args, file=self.f, flush=True)

    def read(self, line):
        items = line.strip().split()
        d = {}
        for name, elem in zip(self.samplers, items[:len(self.samplers)]):
            d[name] = self.samplers[name].load(elem)
        return d, items[len(self.samplers):]

    def close(self):
        self.f.close()

    @abstractmethod
    def load(self, item):
        return

    @abstractmethod
    def range_func(self, *args):
        return

if __name__ == '__main__':
    a = Dispenser(1,'data', 'temp',single_mode=True)
    a.add(IntSampler(2,[0,10]),'hi')
    a.add(IntSampler(3,[0,10]),'hi2')
    a.add(IntSampler(5,[0,10]),'hi3')

    '''
    for i in range(30):
        print(a.cur_idx2idxs(i))
    exit()
    '''
    print(a)
    for _ in a:
        a.print_('test')

    print(__file__)
    with open('data/name','r') as f:
        for line in f:
            print(a.read(line))