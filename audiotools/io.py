'''Read and write PCM 16 bits files.'''

import numpy as np

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

