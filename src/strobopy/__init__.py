# -*- coding: utf-8 -*-

"""Top-level package for strobopy."""

__author__ = """Karl Schliep"""
__email__ = 'kbschliep@gmail.com'
__version__ = '0.1.0'

from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

from src.strobopy.get_dm3 import get_dm3
from src.strobopy.load_dm3 import load_dm3
from src.strobopy.denoise import cleaner
from src.strobopy.strobopy import linescan
