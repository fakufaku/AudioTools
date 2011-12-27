
import sys
import math
import numpy as np
import scipy as sp
#import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
from scipy.signal import correlate as correlate


# open raw PCM audio file, return float array
def read_raw_pcm_16bits(filename):
  fd = open(filename, 'rb')
  data = np.fromfile(file = fd, dtype = np.int16)
  fd.close()
  return np.array(data, dtype=float)


# write a raw PCM audio file
def write_raw_pcm_16bits(filename, data):
  fd = open(filename, 'wb')
  towrite = np.array(data, dtype=np.int16)
  towrite.tofile(fd)
  fd.close()


# convert X to decibels
def dB(X,floor=None):
  if (floor != None):
    return 10.*np.log10(np.maximum(floor, X))
  else:
    return 10.*np.log10(X)

 
# DCT-II of 1D signal (type 2) [equivalent to scipy.fftpack.dct]
def dct(din):
  # dct size
  N = len(din)
  # create output array
  dout = np.zeros(din.shape)
  # rearrange input in output array
  for i in range(0,N/2):
    dout[i] = din[2*i]
    dout[-i-1] = din[2*i+1]
  # twiddle
  W = np.exp(-1j*2*np.pi*np.arange(0,N)/(4*N))
  # return DCT
  return 2*np.real(W * np.fft.fft(dout))

    
# iDCT-II of 1D signal (type 2) [equivalent to scipy.fftpack.idct divided by 2N]
def idct(din):
  # idct size
  N = len(din)
  # create output array
  temp = np.zeros(din.shape, dtype=complex)
  dout = np.zeros(din.shape)
  # twiddle
  W = np.exp(1j*2*np.pi*np.arange(0,N)/(4*N))
  # create output array
  # first point
  temp[0]  = 0.5 * W[0] * din[0]
  temp[1:] = 0.5 * W[1:] * (din[1:] - 1j*din[:0:-1])
  # compute iFFT
  temp = np.real(np.fft.ifft(temp))
  # output mapping
  for i in range(0,N/2):
    dout[2*i] = temp[i]
    dout[2*i+1] = temp[-i-1] 
  # return iDCT
  return dout


# the input mapping for the DCT-II
def dct_II_mapping(d):
  N = len(d)
  # first reverse odd numbers
  for i in range(1,N/2,2):
    d[i], d[N-i] = d[N-i], d[i]
  # then exchange odd and even elements
  for i in range(1,N/2,2):
    d[i], d[i+N/2-1] = d[i+N/2-1], d[i]


# cosine window function
def cosine(N, flag='asymmetric', length='full'):

  # first choose the indexes of points to compute
  if (length == 'left'):     # left side of window
    t = np.arange(0,N/2)
  elif(length == 'right'):   # right side of window
    t = np.arange(N/2,N)
  else:                   # full window by default
    t = np.arange(0,N)

  # if asymmetric window, denominator is N, if symmetric it is N-1
  if (flag == 'symmetric'):
    t = t/float(N-1)
  else:
    t = t/float(N)

  # compute window
  return np.cos(np.pi*(t - 0.5))


# root triangular window function
def triang(N, flag='asymmetric', length='full'):

  # first choose the indexes of points to compute
  if (length == 'left'):     # left side of window
    t = np.arange(0,N/2)
  elif(length == 'right'):   # right side of window
    t = np.arange(N/2,N)
  else:                   # full window by default
    t = np.arange(0,N)

  # if asymmetric window, denominator is N, if symmetric it is N-1
  if (flag == 'symmetric'):
    t = t/float(N-1)
  else:
    t = t/float(N)

  # compute window
  return np.sqrt(1. - np.abs(2.*t - 1.))


# root hann window function
def hann(N, flag='asymmetric', length='full'):

  # first choose the indexes of points to compute
  if (length == 'left'):     # left side of window
    t = np.arange(0,N/2)
  elif(length == 'right'):   # right side of window
    t = np.arange(N/2,N)
  else:                   # full window by default
    t = np.arange(0,N)

  # if asymmetric window, denominator is N, if symmetric it is N-1
  if (flag == 'symmetric'):
    t = t/float(N-1)
  else:
    t = t/float(N)

  # compute window
  return np.sqrt(0.5*(1-np.cos(2*np.pi*t)))


# Rectangular window function
def rect(N):
  return np.ones(N)


# DCT spectrogram
def spectrogram(x, N, L, D, transform=np.fft.fft, win=hann):

  # pad if necessary
  if (len(x)%L != 0):
    x = np.hstack((x, np.zeros(L-(len(x)%L))))

  # compute number of frames
  F = len(x)/L

  # compute left and right windows
  if (D != 0):
    W = win(2*D)
    winL = np.tile(W[0:D], (F, 1)).T  # type DxF
    winR = np.tile(W[D:], (F, 1)).T   # type DxF

  # frames, matrix type LxF
  X = x.reshape(F, L).T.copy()

  if (D != 0):
    # overlap, matrix type DxF , multiply by left window function
    O = np.hstack((np.zeros((D,1)), X[-D:,0:-1]))*winL
    # multiply frames by right window function
    X[-D:,:] *= winR

  # zero-padding, matrix type (N-L-D)xF
  ZP = np.zeros((N-L-D, F))

  # stack frames, overlap and zero-padding: matrix type NxF
  if (D != 0):
    Y = np.vstack((O, X, ZP))
  else:
    Y = np.vstack((X, ZP))

  if (transform == np.fft.fft):
    Z = np.zeros(Y.shape, dtype=complex)
  elif (transform == dct):
    Z = np.zeros(Y.shape, dtype=float)

  # compute DCT
  for i in range(0,F):
    Z[:,i] = transform(Y[:,i])

  return Z


# inverse spectrogram
def margortceps(Y, L, D, transform=np.fft.ifft, win=hann):

  N,F = Y.shape

  if (transform == np.fft.ifft):
    Z = np.zeros(Y.shape, dtype=complex)
  elif (transform == idct):
    Z = np.zeros(Y.shape, dtype=float)

  # compute left and right windows
  if (D != 0):
    W = win(2*D)
    winL = np.tile(W[0:D], (F, 1)).T  # type DxF
    winR = np.tile(W[D:], (F, 1)).T   # type DxF

  # compute iTransform
  for i in range(0,F):
    Z[:,i] = transform(Y[:,i])

  # overlap-add
  if (D != 0):
    O = np.real(Z[0:D,:])*winL  # DxF, overlap, left window is applie

  X = np.real(Z[D:D+L,:])       # LxF, frames
  if (D != 0):
    X[-D:,:] *= winR                    # apply window
    X[-D:,:-1] = O[:,1:] + X[-D:,:-1]   # overlap-add

  return X.T.reshape(L*F)


# Solves the Yule-Walker equations and returns the linear prediction coefficients using Levinson-Durbin algorithm
def yule_walker(X, M):

  if (len(X.shape) > 1):
    print 'Warning: multidimensional arrays not supported yet. May (will) fail.'

  N = len(X)
  
  A = np.zeros(M+1)
  E = 0

  R = correlate(X, X)/N
  R = R[N-1:]

  a = np.zeros(M)
  e = R[0]

  # first loop
  k = R[1]/e
  a[0] = k
  e = e*(1 - k**2)

  # rest of the loop
  for l in range(2,M+1):
    k = (R[l] - (a[0:l-1]*R[l-1:0:-1]).sum())/e
    a[0:l] = np.hstack((a[0:l-1] , 0)) - k*np.hstack((a[l-2::-1] , -1))
    e = e*(1 - k**2)

  A = np.hstack((1, -a))

  return A, e


# auto-regressive spectrum
def auto_regressive_spectrum(x, M, f):

  if (len(x.shape) > 1):
    print 'Warning: multidimensional arrays not supported yet. May (will) fail.'
  a, e = yule_walker(x, M)

  N = len(f)

  I = np.outer(np.arange(0,M+1), f)
  A = np.tile(a, (N, 1)).T

  return e/np.abs((A*np.exp(-2.j*np.pi*I)).sum(axis=0))**2


def mse(x1, x2):
  return (np.abs(x1-x2)**2).sum()/len(x1)

# Itakura-Saito distance function
def itakura_saito(X1,X2):

  P1 = np.abs(X1)**2
  P2 = np.abs(X2)**2

  if (len(P1) != len(P2)):
    print 'Error: Itakura-Saito requires both array to have same length'

  R = P1/P2

  return (R - np.log(R) - 1.).sum()/len(P1)

# compute error frame by frame in spectral domain
def spectral_error_function(x1, x2, N, error_func=mse):

  X1 = spectrogram(x1, N)
  X2 = spectrogram(x2, N)

  F = X1.shape[1]

  E = np.zeros(F)

  # get reference noise power
  P = (np.abs(X1)**2).sum(axis=0)
  P0 = P[X1.shape[1]/2]

  for i in range(0,F):
    E[i] = error_func(X1[:,i], X2[:,i])

  E[P <= 200*P0] = 1

  return E, np.arange(0,F)


# compute SNR frame by frame
def segmental_snr(noisy, clean, N):

  Y = spectrogram(noisy, N) # noisy speech spectrum
  X = spectrogram(clean, N) # clean speech spectrum
  W = Y-X                   # noise spectrum

  return (np.abs(X)**2).sum(axis=0)/(np.abs(W)**2).sum(axis=0)


# Compute the Critical bands as defined in the book:
# Psychoacoustics by Zwicker and Fastl. Table 6.1 p. 159
# with respect to a given sampling frequency Fs and a transform size N
# an optional transform type is used to handle DCT case.
def critical_bands(Fs, N, transform='dft'):
  
  # Those are from the book Psychoacoustics by Zwicker and Fastl. Table 6.1 p. 159
  # center frequencies
  fc = [50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850,
        2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500, 10500, 13500];
  # boundaries of the bands (e.g. the first band is from 0Hz to 100Hz with center 50Hz, fb[0] to fb[1], center fc[0]
  fb = [0,  100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720,
        2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500];

  # set the bin width
  if (transform == 'dct'):
    B = float(Fs)/2./N
  else:
    B = float(Fs)/N


  F = len(fb)

  cb = np.zeros(N)

  i = 0;
  while (fb[i+1] < min(Fs/2, fb[-1])):
    cb[i] = int(np.around(fb[i]/B))
    i += 1

  return cb[:i]


# implement a time-domain circular convolution
# this is not the efficient way to compute it but is sometimes nevertheless usefull
def circconv_td(x, y):

  N = len(x)
  M = len(y)

  if (N > M):
    L = x
    S = y
  else:
    L = y
    S = x

  convmtx = sp.linalg.toeplitz(L, np.hstack((L[0], L[-1:-M:-1])))

  return np.dot(convmtx, S)


## Return the DFT to DCT transform matrix
def spectral_dct_mtx(N):

  D2C = np.zeros((N,N), dtype=complex)

  # DC component
  D2C[0,0] = 2*N

  # l even, right part
  k = np.arange(N/2+1,N)
  l = 2*(N-k)
  D2C[l,k] = N*np.exp(1j*np.pi*l/2./N)

  # l even, left part
  k = np.arange(1, N/2)
  l = 2*k
  D2C[l,k] = N*np.exp(-1j*np.pi*l/2./N)

  # l odd
  K, L = np.meshgrid(np.arange(N), np.arange(1,N,2))
  D2C[1::2,:] = 2*np.exp(1j*np.pi*L/2./N)/(1. - np.exp(1j*np.pi*(2*K+L)/N)) \
              + 2*np.exp(-1j*np.pi*L/2./N)/(1. - np.exp(1j*np.pi*(2*K-L)/N))

  return D2C


