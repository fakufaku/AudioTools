'''Compute decibels dB(x) = 10.*log10(|x|).'''

import numpy as np
# convert X to decibels
def dB(X,floor=None):
  if (floor != None):
    return 10.*np.log10(np.maximum(floor, np.abs(X)))
  else:
    return 10.*np.log10(np.abs(X))

 
