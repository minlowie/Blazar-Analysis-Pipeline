from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name             = 'blazarkit',
    version          = '1.0.0',
    description      = 'Plotting and analysis utilities for Fermi/SDSS-V blazar spectral fitting',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author           = 'Mohammed Iddrisu Nlowie',
    author_email     = '',
    url              = 'https://github.com/minlowie/Blazar-Analysis-Pipeline',
    py_modules       = ['blazarkit'],
    python_requires  = '>=3.8', 
    install_requires = [
        'numpy>=1.24',
        'matplotlib>=3.7',
        'astropy>=5.3',
        'scipy>=1.11',
    ],
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
)
