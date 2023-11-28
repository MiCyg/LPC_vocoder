import numpy as np
import scipy.io.wavfile
from scipy.signal.windows import hann
from scipy.signal import resample
from numpy import floor
from scipy.signal import lfilter
from numpy.random import randn
import matplotlib.pyplot as plt

def create_overlapping_blocks(x, w, R = 0.5):
    n = len(x)
    nw = len(w)
    step = int(floor(nw * (1 - R)))
    nb = int(floor((n - nw) / step) + 1)

    B = np.zeros((nb, nw))

    for i in range(nb):
        offset = i * step
        B[i, :] = w * x[offset : nw + offset]

    return B

def add_overlapping_blocks(B, w, R = 0.5):
    [count, nw] = B.shape
    step = floor(nw * R)

    n = int((count-1) * step + nw)

    x = np.zeros((n, ))

    for i in range(count):
        offset = int(i * step)
        x[offset : nw + offset] += B[i, :]

    return x

def make_matrix_X(x, p):
    n = len(x)
    # [x_n, ..., x_1, 0, ..., 0]
    xz = np.concatenate([x[::-1], np.zeros(p)])

    X = np.zeros((n - 1, p))
    for i in range(n - 1):
        offset = n - 1 - i
        X[i, :] = xz[offset : offset + p]
    return X

def solve_lpc(x, p, ii):
    b = x[1:]

    X = make_matrix_X(x, p)

    a = np.linalg.lstsq(X, b.T)[0]

    e = b - np.dot(X, a)
    g = np.var(e)

    return [a, g]

def lpc_encode(x, p, w):
    B = create_overlapping_blocks(x, w)
    [nb, nw] = B.shape

    A = np.zeros((p, nb))
    G = np.zeros((1, nb))

    for i in range(nb):
        [a, g] = solve_lpc(B[i, :], p, i)

        A[:, i] = a
        G[:, i] = g

    return [A, G]

def run_source_filter(a, g, block_size):
    src = np.sqrt(g)*randn(block_size, 1) # noise

    b = np.concatenate([np.array([-1]), a])

    x_hat = lfilter([1], b.T, src.T).T

    # convert Nx1 matrix into a N vector
    return np.squeeze(x_hat)

def lpc_decode(A, G, w, lowcut = 0):

    [ne, n] = G.shape
    nw = len(w)
    [p, _] = A.shape

    B_hat = np.zeros((n, nw))

    for i in range(n):
        B_hat[i,:] = run_source_filter(A[:, i], G[:, i], nw)

    # recover signal from blocks
    x_hat = add_overlapping_blocks(B_hat, np.ones([1,n]))
    
    return x_hat


downsampling_rate = 8000

[fs, raw_samples] = scipy.io.wavfile.read('odbiemrz.wav')
samples = np.array(raw_samples)
samples = 0.9*samples/max(abs(samples))

target_size = int(len(samples)*downsampling_rate/fs)
samples = resample(samples, target_size)
fs = downsampling_rate

sym = False # periodic
w = hann(int(floor(0.03 * fs)), sym) # 30ms window

p = 12 # number of poles
[A, G] = lpc_encode(samples, p, w)

# Print stats
original_size = len(samples)
model_size = A.size + G.size
print('Original signal size:', original_size)
print('Encoded signal size:', model_size)
print('Data reduction:', original_size/model_size)

xhat = lpc_decode(A, G, w)



scipy.io.wavfile.write("example2.wav", fs, xhat)
print('done')

plt.figure(1)
plt.subplot(1,2,1)
plt.plot(xhat)
plt.subplot(1,2,2)
plt.plot(samples)
plt.show()