import numpy as np
from functools import partial
from abc import ABC
from abc import abstractmethod

class Sampler(ABC):
    def __init__(self, cnt, range_func):
        self.ranges = range_func()
        self.cnt = cnt
        self.cur = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.cur += 1
        if self.cnt < self.cur:
            raise StopIteration

        return self.ranges[self.cur-1]

    def print_(self, f):
        print(self.ranges[self.cur], file=f, flush=True)

    def get(self, idx):
        assert idx < self.cnt
        return self.ranges[idx]

    @abstractmethod
    def load(self, item):
        return

    @abstractmethod
    def range_func(self, *args):
        return

class LinSampler(Sampler):
    def __init__(self, cnt, ranges):
        super(LinSampler, self).__init__(cnt, partial(self.range_func, cnt, ranges))

    def range_func(self, cnt, ranges):
        return np.linspace(start = ranges[0], stop= ranges[1], num=cnt)

    def load(self, item):
        return float(item)

class IntSampler(Sampler):
    def __init__(self, cnt, ranges):
        super(IntSampler, self).__init__(cnt, partial(self.range_func, cnt, ranges))

    def range_func(self, cnt, ranges):
        assert isinstance(ranges[0], int) and isinstance(ranges[1], int)
        assert np.abs(ranges[1] - ranges[0]) >= cnt
        arrs = np.array_split(np.arange(ranges[0], ranges[1]), cnt)
        return [arr[0] for arr in arrs]

    def load(self, item):
        return int(item)

class LogSampler(Sampler):
    def __init__(self, cnt, ranges):
        super(LogSampler, self).__init__(cnt, partial(self.range_func, cnt, ranges))

    def range_func(self, cnt, ranges):
        assert ranges[1] > 0 and ranges[0] > 0
        arrs = np.logspace(np.log10(ranges[0]), np.log10(ranges[1]), cnt)
        return arrs

    def load(self, item):
        return float(item)

class RangeSampler(Sampler):
    def __init__(self, ranges):
        super(RangeSampler, self).__init__(len(ranges), partial(self.range_func, ranges))

    def range_func(self, ranges):
        return ranges

    def load(self, item):
        return int(item)

class ListSampler(Sampler):
    def __init__(self, l):
        super(ListSampler, self).__init__(len(l), partial(self.range_func, l))
        self.type = type(list(l)[0])

    def range_func(self, l):
        return list(l)

    def load(self, item):
        return self.type(item)

class ChopSampler(Sampler):
    def __init__(self, cnt, ranges):
        super(ChopSampler, self).__init__(cnt, partial(self.range_func, cnt, ranges))

    def range_func(self, cnt, ranges):
        return np.array_split(np.array(list(ranges)),cnt)

    def load(self, item):
        return int(item)

if __name__ == '__main__':
    A= ChopSampler(4,range(10,20))
    for _ in A:
        print(_)