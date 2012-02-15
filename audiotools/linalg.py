'''Some general linear algebra methods.'''

import numpy as np

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

