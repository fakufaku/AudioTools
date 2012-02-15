
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
    return 10.*np.log10(np.maximum(floor, np.abs(X)))
  else:
    return 10.*np.log10(np.abs(X))

 
# DCT-II of 1D signal (type 2) [equivalent to scipy.fftpack.dct]
def dctII(din):
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

    
# DCT-III of 1D signal (type 2) [equivalent to scipy.fftpack.idct divided by 2N]
def dctIII(din):
  # dct-III size
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
  # return DCT-III
  return dout

# DST-III of 1D signal (type 3) [see Shao and Johnson, Type-IV DCT, DST and MDCT algorithms with reduced numbers of arithmetic operations, 2009]
def dstIII(x):
  # size and output array
  N = len(x)
  X = np.zeros(N)
  # reverse array
  X[0] = x[0]
  X[1:] = x[-1:0:-1]
  # perform DCT-III
  X = dctIII(X)
  # invert sign of odd indices
  X[1::2] *= -1
  # return DST-III
  return X


# DCT-IV of 1D signal (type 4) [see Z. Wang, On Computing the Discrete Fourier and Cosine Transforms, 1985]
def dctIV(x):
  # size and output array
  N = len(x)
  X = np.zeros(N)
  # Multiplication by matrix Bn from paper, combined with reordering from Pn, permutation matrix
  X[0] = x[0] # factor sqrt(2) cancelled by later DCT-III
  for n in range(1,N-1,2):
    X[N-1-(n-1)/2] = x[n] - x[n+1]    # odd indices
    X[(n+1)/2] = x[n] + x[n+1]        # even indices
  X[N/2] = x[N-1] # factor sqrt(2) cancelled by later DST-III
  # Half-length DCT-III
  X[0] *= 2     # to conform to our definition of DCT-III
  X[0:N/2] = dctIII(X[0:N/2])
  # Half-length DST-III of reversed data, in reversed order
  X[N/2] *= 2   # to conform to our definition of DST-III
  X[N/2+1:] = X[-1:N/2:-1]
  X[N-1:N/2-1:-1] = dstIII(X[N/2:])
  # Multiplication by Tn from paper (Rotation matrix)
  for n in range(0,N/2):
    c = np.cos((2*n+1)*np.pi/4./N)
    s = np.sin((2*n+1)*np.pi/4./N)
    X[[n, N-1-n]]  = np.dot(np.array([[c,s],[s,-c]]), X[[n,N-1-n]])
  # Scaling was adjusted compared to paper so that dctIV is orthonormal (its own inverse)
  X *= np.sqrt(N/2.)
  # return DCT-IV
  return X


# DCT-II of 1D signal (type 2) [equivalent to scipy.fftpack.dct]
def dct(din):
  return dctII(din)


# iDCT-II (<=> DCT-III) of 1D signal (type 2) [equivalent to scipy.fftpack.idct divided by 2N]
def idct(din):
  return dctIII(din)


# Modified Discrete Cosine Transform
# see ref: Shao and Johnson, Type-IV DCT, DST, and MDCT algo..., 2009
# uses DCT-IV to compute MDCT
# takes as input 2N array, outputs N array
def mdct(x):
  N = len(x)/2 # output is half length

  # create array
  X = np.zeros(N)

  # fill the array according to Shao and Johnson
  for n in range(N/2):
    X[n] = -x[3*N/2-1-n] - x[3*N/2+n]
  for n in range(N/2,N):
    X[n] = x[n-N/2] - x[3*N/2-1-n]

  # take the DCT-IV of this array
  X = dctIV(X)
  
  # return MDCT
  return X


# inverse Modified Discrete Cosine Transform
# see ref: Shao and Johnson, Type-IV DCT, DST, and MDCT algo..., 2009
# uses DCT-IV to compute iMDCT
# takes N array as input, outputs 2N array
def imdct(X):
  N = len(X)

  # create array
  x = np.zeros(2*N)

  # take DCT-IV
  x[:N] = dctIV(X)

  # we know DCT-IV output is mirror antisymmetric, the iMDCT is shifted by N/2
  x[N:3*N/2] = -x[N/2-1::-1]
  x[3*N/2:] = -x[:N/2]
  x[:N/2] = x[N/2:N]
  x[N/2:N] = -x[N/2-1::-1]

  # return iMDCT
  return x


# Discrete Hartley transform
def dht(din):
  # dht size
  N = len(din)
  # create output array
  H = np.zeros(N)
  # Fourier transform
  F = np.fft.fft(din)
  # mapping to DHT
  for k in range(N):
    H[k] = np.real(F[k]) - np.imag(F[k])
  # return DHT
  return H


# inverse Discrete Hartley transform
def idht(din):
  # dht size
  N = len(din)
  # create output array
  F = np.zeros(N,dtype=complex)
  # mapping to DFT
  F[0] = din[0]
  for k in range(1,N):
    F[k] = 0.5*(din[k] + din[N-k]) - 1j*(din[k] - din[N-k])
  # inverse Fourier transform
  F = np.fft.ifft(F)
  # return iDFT
  return np.real(F)


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
  if (flag == 'symmetric' or flag == 'mdct'):
    t = t/float(N-1)
  else:
    t = t/float(N)

  w = np.cos(np.pi*(t - 0.5))**2

  # make the window respect MDCT condition
  if (flag == 'mdct'):
    w **= 2
    d = w[:N/2]+w[N/2:]
    w[:N/2] *= 1./d
    w[N/2:] *= 1./d
    w = np.sqrt(w)

  # compute window
  return w


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
  if (flag == 'symmetric' or flag == 'mdct'):
    t = t/float(N-1)
  else:
    t = t/float(N)

  w = 1. - np.abs(2.*t - 1.)

  # make the window respect MDCT condition
  if (flag == 'mdct'):
    d = w[:N/2]+w[N/2:]
    w[:N/2] *= 1./d
    w[N/2:] *= 1./d

  w = np.sqrt(w)

  # compute window
  return w


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
  if (flag == 'symmetric' or flag == 'mdct'):
    t = t/float(N-1)
  else:
    t = t/float(N)

  w = 0.5*(1-np.cos(2*np.pi*t))

  # make the window respect MDCT condition
  if (flag == 'mdct'):
    d = w[:N/2]+w[N/2:]
    w[:N/2] *= 1./d
    w[N/2:] *= 1./d

  # take square root for perfect reconstruction
  w = np.sqrt(w)

  # compute window
  return w


# Rectangular window function
def rect(N):
  return np.ones(N)


# DCT spectrogram
def spectrogram(x, N, L, D, transform=np.fft.fft, win=hann):

  if ((L != D) and transform == mdct):
    print "Frame size and overlap must be equal for MDCT"
    sys.exit(-1)

  # pad if necessary
  if (len(x)%L != 0):
    x = np.hstack((x, np.zeros(L-(len(x)%L))))

  # compute number of frames
  F = len(x)/L

  # compute left and right windows
  if (D != 0):
    if (transform == mdct):
      W = win(2*D, flag='mdct')
    else:
      W = win(2*D)
    winL = np.tile(W[0:D], (F, 1)).T  # type DxF
    winR = np.tile(W[D:], (F, 1)).T   # type DxF

  # frames, matrix type LxF
  X = np.zeros((L,F), dtype=float)
  X = x.reshape(F, L).T.copy()*1.

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
  elif (transform == mdct):
    Z = np.zeros((N/2, F), dtype=float)
  else:
    Z = np.zeros(Y.shape, dtype=float)

  # compute DCT
  for i in range(0,F):
    Z[:,i] = transform(Y[:,i])

  return Z


# inverse spectrogram
def margortceps(Y, L, D, transform=np.fft.ifft, win=hann):

  if ((L != D) and transform == mdct):
    print "Frame size and overlap must be equal for MDCT"
    sys.exit(-1)

  N,F = Y.shape

  if (transform == np.fft.ifft):
    Z = np.zeros(Y.shape, dtype=complex)
  elif (transform == imdct):
    Z = np.zeros((Y.shape[0]*2,Y.shape[1]), dtype=float)
  else:
    Z = np.zeros(Y.shape, dtype=float)

  # compute left and right windows
  if (D != 0):
    if (transform == imdct):
      W = win(2*D, flag='mdct')
    else:
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


# AR spectrogram estimate
def ar_spectrogram(x, L, D, M, f):

  # pad if necessary
  if (len(x)%L != 0):
    x = np.hstack((x, np.zeros(L-(len(x)%L))))

  # compute number of frames
  F = len(x)/L

  # frames, matrix type LxF
  X = np.zeros((L,F), dtype=float)
  X = x.reshape(F, L).T.copy()*1.

  if (D != 0):
    # overlap, matrix type DxF , multiply by left window function
    O = np.hstack((np.zeros((D,1)), X[-D:,0:-1]))*winL
    # multiply frames by right window function
    X[-D:,:] *= winR

  # stack frames, overlap and zero-padding: matrix type NxF
  if (D != 0):
    Y = np.vstack((O, X, ZP))
  else:
    Y = np.vstack((X, ZP))

  Z = np.zeros((len(f), Y.shape[1]), dtype=float)
  AR = np.zeros((M+1, Y.shape[1]), dtype=float)
  E = np.zeros(Y.shape[1], dtype=float)

  # compute DCT
  for i in range(0,F):
    AR[:,i], E[i] = yule_walker(x, M)

    N = len(f)

    I = np.outer(np.arange(0,M+1), f)
    A = np.tile(AR[:,i], (N, 1)).T

    Z[:,i] =  E[i]/np.abs((A*np.exp(-2.j*np.pi*I)).sum(axis=0))**2

  return Z, AR, E


# Simple mean squared error function
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
  D2C[1::2,:] = 1j*np.exp(-1j*np.pi*K/N)*(1/np.sin(np.pi*(K+L/2.)/N) + 1/np.sin(np.pi*(K-L/2.)/N))

  return D2C


## A Naive Gram-Schmidt algo, for when I want to know what is happening
# makes the columns of the input matrix orthonormal to each other
def gram_schmidt(A):

  E = A.copy()

  E[:,0] /= np.sqrt(np.inner(A[:,0], A[:,0]))

  for j in range(1, A.shape[1]):

    E[:,j] = A[:,j]
    for k in range(1,j+1):
      p = np.inner(E[:,j-k], A[:,j])
      E[:,j] -= p*E[:,j-k]

    E[:,j] /= np.sqrt(np.inner(A[:,j], A[:,j]))

  return E


