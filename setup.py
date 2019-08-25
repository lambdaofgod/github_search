from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='github_search',
    version='0.01',
    url='https://github.com/lambdaofgod/github_search',
    author='Jakub Bartczuk',
    packages=find_packages(),
    install_requires=requirements
)
