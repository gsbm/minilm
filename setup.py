import os

from setuptools import find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), 'r', encoding='utf-8') as f:
        return f.read()


VERSION = '0.1.0'
DESCRIPTION = 'A lightweight toolkit for experimenting with compact language models'
LONG_DESCRIPTION = read('README.md')

setup(
    name="minilm",
    version=VERSION,
    author="gsbm",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="GPL-3.0-or-later",
    keywords=[
        'nlp', 'machine-learning', 'deep-learning', 'language-model',
        'transformers', 'pytorch', 'natural-language-processing',
    ],
    url="https://github.com/gsbm/minilm",
    project_urls={
        'Bug Tracker': 'https://github.com/gsbm/minilm/issues',
        'Source Code': 'https://github.com/gsbm/minilm',
    },
    packages=find_packages(include=['minilm', 'minilm.*']),
    python_requires='>=3.8',
    install_requires=[
        "numpy>=1.24",
        "pyarrow>=14.0",
        "torch>=2.0",
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'ruff>=0.14.0',
            'hypothesis>=6.0',
            'twine>=4.0',
        ],
        'docs': [
            'sphinx>=5.0',
            'sphinx-rtd-theme>=1.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
    zip_safe=False,
)
