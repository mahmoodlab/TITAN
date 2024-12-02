from setuptools import setup, find_packages
from codecs import open
from os import path

setup(
    name='titan',
    version='0.1.0',
    description='TITAN',
    url='https://github.com/mahmoodlab/TITAN',
    author='TD,SJW,AHS,RJC',
    author_email='',
    license='CC-BY-NC-ND 4.0',
    packages=find_packages(exclude=['__dep__', 'assets']),
    install_requires=["torch==2.0.1", "torchvision==0.15.2", "timm==1.0.3", "einops==0.6.1", "einops-exts==0.0.4", "tqdm==4.66.6", "h5py==3.8.0", "transformers==4.46.0", "pandas==2.2.3", "scikit-learn==1.5.2"],
    classifiers = [
    "Programming Language :: Python :: 3",
    "License :: CC-BY-NC-ND 4.0",
]
)