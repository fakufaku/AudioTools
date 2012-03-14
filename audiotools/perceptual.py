'''Methods related to perception of speech.'''

import numpy as np

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

  # convert frequency boundaries into bins
  cb[0] = int(fb[0]/B)   # this is usually == 0
  i = 1;
  while (fb[i+1] < min(Fs/2, fb[-1])):
    cb[i] = int(np.around(fb[i]/B))
    i += 1

  # add the last boundary (transform size)
  if (cb[i-1] != N):
    cb[i] = N
    i += 1

  # remove duplicate, if any, (typically, if N is small and Fs is large)
  j = 0
  while (j < i):
    if (cb[j] == cb[j+1]):
      cb[j:i-1] = cb[j+1:i]
      i -= 1
    j += 1


  return cb[:i]


