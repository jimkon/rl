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
