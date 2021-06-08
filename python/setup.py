import setuptools
from numpy.distutils.core import setup, Extension

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
print(this_directory[:-5])
with open(path.join(this_directory[:-5], 'README.md')) as f:
    long_description = f.read()

ext1 = Extension(name='fiesta.src.grid', sources=['fiesta/src/grid.f90'])
exts = []

setup(name = 'shift',
      version = '0.0.0',
      description       = "SpHerIcal Fourier Transforms",
      long_description  = long_description,
      long_description_content_type = 'text/markdown',
      url               = 'https://github.com/knaidoo29/fiesta',
      author            = "Krishna Naidoo",
      author_email      = "krishna.naidoo.11@ucl.ac.uk",
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=['numpy', 'scipy'],
      #ext_modules = exts,
      python_requires = '>=3',
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Simulation',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      )
