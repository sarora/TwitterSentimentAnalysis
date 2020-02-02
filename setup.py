#!/usr/bin/env python
import sys
from setuptools import setup, find_packages
import argparse

argparser = argparse.ArgumentParser(add_help=False)
argparser.add_argument('--gpu', default=False,
                       help='use to build gpu version of conda package',
                       action="store_true")
args, unknown = argparser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

NAME = "sentiment_analysis"

INSTALL_REQUIRES = [
    "numpy ==1.17.3",
    "pandas ==0.23.4",
    "scikit-learn ==0.21.3",
    "spacy ==2.0.16",
    "ftfy ==5.6.0",
    "imbalanced-learn ==0.5.0",
    "tqdm",
    'tensorflow-io==0.10.0',
    'tensorflow-addons==0.6.0',
    'tsfresh==0.13.0',
    'statsmodels==0.10.1',
]

# add development requires here
DEV_REQUIRES = [
    "pytest",
    "pytest-html",
    "pytest-xdist",
    "pytest-forked",
    "pytest-cov",
    "setuptools",
    "sphinx",
    "sphinx-gallery",
    "sphinx_rtd_theme",
    "matplotlib",
    "pillow",
    "responses",
    "pylint",
]

# TensorFlow dependencies

TF_VERSION = " ==2.0.0"
TF = "tensorflow" + TF_VERSION
if args.gpu:
    NAME += "-gpu"
    TF = 'tensorflow-gpu' + TF_VERSION
INSTALL_REQUIRES.append(TF)

setup(
    name=NAME,
    version='0.1.0',
    author="Sidd Arora",
    author_email='sidd.arora.05@gmail.com',
    packages=find_packages(),
    description="Perform Sentiment Analysis",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'dev': DEV_REQUIRES,
        'tf-gpu': 'tensorflow-gpu' + TF_VERSION,
        'tf-cpu': 'tensorflow' + TF_VERSION
    },
    include_package_data=True,
)
