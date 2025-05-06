
from setuptools import setup, find_packages



with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='factorio_balancers',
    version='1.0.0',
    description='A library for working with Factorio belt balancer graphs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='sOvr9000',
    author_email='jordanthegeek@gmail.com',
    url='https://github.com/sOvr9000/factorio_balancers',
    packages=find_packages(),
    install_requires=[
        'graph_tools',
        'sympy',
        'factorio_blueprints',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.6',
)


