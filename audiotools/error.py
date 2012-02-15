'''Collection of functions related to error computations.'''

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


