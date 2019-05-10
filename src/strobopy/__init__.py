# -*- coding: utf-8 -*-

"""Top-level package for strobopy."""


from __future__ import absolute_import, division, print_function


__author__ = """Karl Schliep"""
__email__ = 'kbschliep@gmail.com'
__version__ = '0.1.0'
__title__ = "strobopy"
__description__ = "Analysis package for ultrafast stroboscopic TEM data"
# __url__ = "https://www.attrs.org/"
# __uri__ = __url__
# __doc__ = __description__ + " <" + __uri__ + ">"

__license__ = "mit"
__copyright__ = "Copyright (c) 2019 Karl Schliep"


from src.strobopy.get_dm3 import get_dm3
from src.strobopy.load_dm3 import load_dm3
from src.strobopy.denoise import cleaner
from src.strobopy.strobopy import linescan
