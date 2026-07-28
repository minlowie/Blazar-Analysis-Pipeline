from setuptools import find_packages, setup

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
    package_dir      = {"": "src"},
    packages         = find_packages(where="src"),
    py_modules       = ['blazarkit'],
    package_data     = {'blazarkit': ['dropbox_links.json']},
    data_files            = [('', ['dropbox_links.json'])],
    include_package_data  = True,
    python_requires  = '>=3.10', 
    install_requires = [
        'numpy>=2.0',
        'matplotlib>=3.7',
        'astropy>=6.1',
        'scipy>=1.11',
        'pandas>=1.5',
    ],
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
)
