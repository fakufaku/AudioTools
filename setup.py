from distutils.core import setup

setup(
    name='AudioTools',
    version='0.3.0',
    author='Robin Scheibler',
    author_email='r-scheibler@zf.jp.nec.com',
    packages=['audiotools'],
    #scripts=['bin/spectrogram_distortion.py'],
    #url='http://pypi.python.org/pypi/TowelStuff/',
    #license='LICENSE.txt',
    description='Useful audio signal processing tools.',
    long_description=open('README.txt').read(),
    install_requires=[
      "numpy >= 1.5.1",
      "scipy >= 0.9.0",
      "matplotlib >= 1.0.1"
      ],
)
