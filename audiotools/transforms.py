'''A Collection of trigonometric orthogonal transforms.'''

import numpy as np

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


