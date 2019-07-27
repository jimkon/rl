import numpy as np


# by LazyProgrammer
def running_average(arr, frame=-1):

    arr = np.array(arr)
    N = len(arr)

    if frame == -1:
        frame = int(N*.1)
    frame = np.min([frame, N])
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = arr[max(0, t - frame):(t + 1)].mean()

    return running_avg


def epsilon(n):
    if n<0:
        return .0
    return  1.0 / np.sqrt(n + 1)


class BatchBuffer:

    def __init__(self, maxsize):
        self.__maxsize = maxsize
        self.__buffer = []

    def push(self, items):
        self.__buffer.append(items)
        self.__pop_extra()

    def __pop_extra(self):
        extra = len(self.__buffer)-self.__maxsize
        for _ in range(extra):
            self.__buffer.pop(0)

    def get_random(self, count=1):
        indexes = np.random.randint(0, len(self.__buffer), count)
        return [self.__buffer[index] for index in indexes]


class Mapper:

    def map(self, state):
        return state

class StandardMapper(Mapper):

    def __init__(self, low, high):
        self.low = np.array(low)
        self.high = np.array(high)
        self.lenght = self.high-self.low

    def map(self, state):
        mapped = (state-self.low)/self.lenght
        return mapped


class UnitMapper(StandardMapper):

    def map(self, state):
        return 2*super().map(state)-1
