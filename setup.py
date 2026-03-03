from setuptools import setup, find_packages

setup(
    name='null-cone-sgd',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
        'openai>=1.0.0'
    ],
    author='Y.Y.N. Li',
    description='6.7x faster training and 1.6% forgetting with Null Cone geometry',
    url='https://github.com/papasop/null-cone-sgd'
}