"""
VecShare: A Framework for Sharing Word Embeddings

This is a framework for efficiently selecting, querying and downloading word
embeddings for NLP tasks. This release follows the indexer-signature model
described in <PROCEEDINGS URL>.

Latest Stable Version: 'pip install vecshare'
File Issues at: 'https://github.com/JaredFern/VecShare'
"""

import re
from os import path
from setuptools import setup, find_packages

def read(*paths):
	filename = path.join(path.abspath(path.dirname(__file__)),*paths)
	with open(filename) as f:
		return f.read()

def find_version(*paths):
	contents = read(*paths)
	match = re.search(r'^__version__ = [\'"]([^\'"]+)[\'"]', contents, re.M)
	if not match:
		raise RuntimeError('Unable to find version string.')
	return match.group(1)

setup(
	name = 'vecshare',
	version = find_version('vecshare','__init__.py'),
	description = 'Python library for sharing word embeddings',
	long_description = read('README.md'),
	url = 'https://github.com/JaredFern/VecShare',
	author = 'JaredFern',
	author_email = 'jared.fern@u.northwestern.edu',
	license = 'Apache 2.0',
	packages = find_packages(),
	keywords = [
		'word embeddings',
		'word vectors',
		'vecshare',
		'natural language processing',
		'NLP'
	],
	classifiers = [
		'Development Status :: 5 - Production/Stable',
		'Intended Audience :: Developers',
		'Intended Audience :: Science/Research',
		'Operating System :: OS Independent',
		'License :: OSI Approved :: Apache Software License',
		'Programming Language :: Python :: 2.7',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.4',
		'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 3.6',
		'Topic :: Software Development :: Libraries :: Python Modules',
		'Topic :: Scientific/Engineering :: Information Analysis',
	],

	install_requires=[
		'pandas',
		'numpy',
		'nltk',
		'scipy',
		'datadotworld',
		'bs4',
		'selenium',
		'progressbar2',
		'tabulate'
	],

	setup_requires=[
		'pytest-runner',
	],
)
