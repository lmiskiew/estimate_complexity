#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'scipy>=0.18.0',
    'numpy>=1.11.0',
    'timeout-decorator>=0.3.3',
    'matplotlib>=2.0.0',
    'pynverse>=0.1.4.4'
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='estimate_complexity',
    version='0.1.0',
    description="Simple program to estimate the complexity of a function.",
    long_description=readme + '\n\n' + history,
    author="Łukasz Miśkiewicz",
    author_email='lukasz.miskiewicz20@gmail.com',
    url='https://github.com/lmiskiew/estimate_complexity',
    packages=[
        'estimate_complexity',
    ],
    package_dir={'estimate_complexity':
                 'estimate_complexity'},
    include_package_data=True,
    install_requires=requirements,
    license="GNU General Public License v3",
    zip_safe=False,
    keywords='estimate_complexity',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
