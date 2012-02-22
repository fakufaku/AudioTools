'''Collection of spectral estimation methods.'''

import numpy as np
from windows import *
from scipy.signal import correlate as correlate

from transforms import *
from windows import *

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


# Compute singe-sided autocorrelation of a sequence using FFT method
# This is the >biased< estimate of the ACF and only returns the positive
# part of the ACF: R(n), n >= 0
def autocorr(X):

  N = X.shape[0]

  # fft based columnwise correlation
  R = np.zeros((N,1))
  R = np.real(np.fft.ifft(np.abs(np.fft.fft(X,n=2*N-1,axis=0))**2,n=None,axis=0))[:N,]/N

  return R


# Solves the Yule-Walker equations and returns the linear prediction coefficients using Levinson-Durbin algorithm
# If X is a matrix, the algorithm operates on the columns of the matrix
def yule_walker(X, M):

  N = X.shape[0]
  if (len(X.shape) > 1):
    F = X.shape[1]
  else:
    F = 1
  
  A = np.zeros(M+1)
  E = 0

  # FFT based columnwise correlation
  R = autocorr(X)

  if (F == 1):
    a = np.zeros(M)
  else:
    a = np.zeros((M,F))

  e = R[0,]

  # first loop
  k = R[1,]/e
  a[0,] = k
  e = e*(1 - k**2)

  # rest of the loop
  for l in range(2,M+1):
    k = (R[l,] - (a[0:l-1,]*R[l-1:0:-1,]).sum(axis=0))/e
    if (F == 1):
      a[0:l] = np.hstack((a[0:l-1] , 0)) - k*np.hstack((a[l-2::-1] , -1))
    else:
      a[0:l,] = np.vstack((a[0:l-1,] , np.zeros((1,F)))) - k*np.vstack((a[l-2::-1,] , -np.ones((1,F))))
    e = e*(1 - k**2)

  if (F == 1):
    A = np.hstack((1, -a))
  else:
    A = np.vstack((np.ones((1,F)), -a))

  if (F == 1):
    A = A.T

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
def autoregressive_spectrogram(x, L, D, M, f, window=hann):

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
    O = np.hstack((np.zeros((D,1)), X[-D:,0:-1]))

  # stack frames, overlap and zero-padding: matrix type NxF
  if (D != 0):
    Y = np.vstack((O, X))
  else:
    Y = X

  Z = np.zeros((len(f), Y.shape[1]), dtype=float)

  # precomputation of complex exponentials for later use
  I = np.outer(np.arange(0,M+1), f)
  EXP = np.exp(-2.j*np.pi*I)

  # window function
  W = np.tile(window(Y.shape[0]), (Y.shape[1], 1)).T

  # compute Yule-Walker coefficients (on columns)
  AR, E = yule_walker(W*Y, M)

  # compute DCT
  for i in range(0,F):
    N = len(f)
    A = np.tile(AR[:,i], (N, 1)).T
    Z[:,i] =  E[i]/np.abs((A*EXP).sum(axis=0))**2

  return Z, AR, E


