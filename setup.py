from setuptools import setup
from setuptools import find_packages

setup(name='ivis-supervised',
      version='1.0',
      description='Artificial neural network-driven visualization of high-dimensional data using supervised-sampled triplets.',
      url='http://github.com/beringresearch/ivis-supervised',
      author='Benjamin Szubert, Ignat Drozdov',
      author_email='bszubert@beringresearch.com, idrozdov@beringresearch.com',
      license='Creative Commons Attribution-NonCommercial-NoDerivs 3.0',
      packages=find_packages(),
      install_requires=[
          'tensorflow',
          'keras',
          'numpy',
          'scikit-learn',
      ],
      zip_safe=False)
