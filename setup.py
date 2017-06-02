#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="benderthon",
    version="0.1.0",
    description="TODO",
    long_description="TODO",
    url="TODO",
    keywords=["Bender", "machine learning", "artificial intelligence", "freeze", "model", "utility", "utilities",
              "TensorFlow"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'benderthon = benderthon.cmdline:main',
        ],
    },
    install_requires=[
        'tensorflow>=1.2.0rc1',
    ],
    author="Xmartlabs",
    author_email="hi@xmartlabs.com",
    maintainer="Santiago Castro",
    maintainer_email="santiago@xmartlabs.com",
    license="Apache 2.0",
)
