
import numpy as np

# Trains a Gaussian Mixture according to sample set x
# using the Expectation Maximization algorithm
def em_gmm_training(x, M, T=10, model=0, theta0=None):

  # the samples are stored column-wise in x
  D = x.shape[0]  # dimension of data to model
  N = x.shape[1]  # number of samples

  # initialize parameters
  if (theta0 != None):
    mu = theta0[0]
    Sigma = theta0[1]
    w = theta0[2]
  else:
    # default to zero-mean, unit variance
    mu = np.zeros((D,M))
    if (constraint == 0):
      # Covariance matrix is diagonal
      Sigma = np.ones((D, M))
    else:
      # full free diagonal covariance matrix
      Sigma = np.zeros((D,D,M))
      for m in range(M):
        Sigma[:,:,m] = np.eye(D)
    w = np.ones(M)/float(M)

  # initialize variables
  T = np.zeros((M, N))
  F = np.zeros((M, N))

  # loop for optimization
  for t in range(T):
    # Expectation step:
    ###################

    # compute individual mixture components and total mixture
    gm = np.zeros(N)
    for m in range(M):
      if (model == 0):
        F[m,:] = normalpdf(x, mu[:,m], Sigma[:,m])
      else:
        F[m,:] = normalpdf(x, mu[:,m], Sigma[:,:,m])
      gm += w[m]*F[m,:]

    for m in range(M):
      for n in range(N):
        T[m,n] = w[m]*F[m,n]/(gm[n])

    # Maximization step:
    ####################
    
    # weights update
    w = T.mean(axis=1)

    # mean and covariance matrix update
    for m in range(M):

      # auxilliary
      S = T[m,:].sum()
      Tm = np.tile(T[m,:], (D,1))

      # mu (mean)
      mu[:,m] = (x*Tm).sum(axis=1)/S

      # sigma (cov matrix)
      xm = (x - np.tile(mu[:,m], (N,1)).T)
      if (model == 0):
        Sigma[:,m] = (Tm*xm**2).sum(axis=1)/S
      else:
        Sigma[:,:,m] = (Tm*np.outer(xm,xm)).sum(axis=1)/S

  # return the trained GMM
  return mu, Sigma, w   


# compute the value of the probability
# density function of a Gaussian mixture
def gmpdf(x, mu, Sigma, w):

  f = np.zeros(x.shape)

  M = w.shape[0]  # number of components in mixture
  D = x.shape[0]  # dimension of the samples
  N = x.shape[1]  # number of samples

  # compute pdf value for samples
  for m in range(M):
    if (len(Sigma.shape) == 2):
      f += w[m]*normalpdf(x, mu[:,m], Sigma[:,m])
    else:
      f += w[m]*normalpdf(x, mu[:,m], Sigma[:,:,m])

  return f

# compute the probability density function
# of a multivariate normal distribution
def normalpdf(x, mu, Sigma):

  D = mu.shape[0]
  if (len(x.shape) == 1 and x.shape[0] == D):
    N = 1
  elif (len(x.shape) == 2):
    N = x.shape[1]
  else:
    print 'Error! Dimension mismatch in normalpdf.'

  # remove mean
  xm = x - np.tile(mu[:,m], (N, 1)).T

  # compute determinant inverse of Sigma, and then exponent
  if (len(Sigma.shape) == 1):
    # diagonal covariance matrix (independent components)
    detSigma = Sigma.prod(axis=0)
    invSigma = 1./Sigma
    a = -0.5*(np.tile(invSigma, (N, 1)).T*xm**2).sum(axis=0)
  else:
    # full covariance matrix
    detSigma = np.linalg.det(Sigma[:,:])
    invSigma[:,:,m] = np.linalg.inv(Sigma[:,:,m])
    a = -0.5*(xm*np.dot(invSigma[:,:,m], xm)).sum(axis=0)

  # compute pdf
  return np.exp(a)/((2*np.pi)**(D/2.)*np.sqrt(detSigma[m]))


