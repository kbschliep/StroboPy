#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `strobopy` package."""

import pytest

from click.testing import CliRunner

import strobopy as st
from strobopy import cli
import numpy as np
# import os
# print(os.path.dirname(os.path.realpath('__file__')))
def test_load_dm3():
    import os
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, 'tests\\test_data\\test_data.dm3')
    assert st.load_dm3(filename)[0][0]==-1

def test_outliers_norm():
    from strobopy.denoise import outliers
    num_stdevs=2
    m=np.arange(25).reshape(5,5)
    m[1,3]=np.mean(m)-num_stdevs*np.std(m)-1
    m[3,1]=+np.mean(m)+num_stdevs*np.std(m)+1
    assert     (outliers(m)==np.array([[False, False, False, False, False],
       [False, False, False,  True, False],
       [False, False, False, False, False],
       [False,  True, False, False, False],
       [False, False, False, False, False]])).all

def test_outliers_pct():
    from strobopy.denoise import outliers
    quart=5
    uquart=100-quart
    m=np.arange(25).reshape(5,5)
    q25, q75 = np.percentile(m, quart), np.percentile(m, uquart)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    m[1,3]=q25-cut_off-1
    m[3,1]=q75+cut_off+1
    assert     (outliers(m)==np.array([[False, False, False, False, False],
       [False, False, False,  True, False],
       [False, False, False, False, False],
       [False,  True, False, False, False],
       [False, False, False, False, False]])).all

# @pytest.fixture
# def load():
#     """Sample pytest fixture.
#
#     See more at: http://doc.pytest.org/en/latest/fixture.html
#     """
#
#     # import requests
#     # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')
#
#
# def test_content(response):
#     """Sample pytest test function with the pytest fixture as an argument."""
#     # from bs4 import BeautifulSoup
#     # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'strobopy.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output
