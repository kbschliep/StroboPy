#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `strobopy` package."""

import pytest

from click.testing import CliRunner
import os, sys
print('PATH IS',os.path.dirname(os.path.realpath('__file__')))
# filename = os.path.join(fileDir, 'tests\\test_data\\test_data.dm3')

from src import strobopy as st
from src.strobopy import cli
import numpy as np


def test_load_dm3():
    import os

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, 'tests\\test_data\\test_data.dm3')
    assert st.load_dm3(filename)[0][0]==-1

def test_outliers_norm_():
    from src.strobopy.denoise import outliers
    num_stdevs=2
    m=np.arange(25).reshape(5,5)
    mmean=np.mean(m)
    mstd=np.std(m)
    m[1,3]=mmean-num_stdevs*mstd-4
    m[3,1]=+mmean+num_stdevs*mstd+4
    outliersworks,indexworks,get_cutoffworks=outliers(m,index=True,get_cutoff=True)
    assert     (outliersworks==np.array([[False, False, False, False, False],
        [False, False, False,  True, False],
        [False, False, False, False, False],
        [False,  True, False, False, False],
        [False, False, False, False, False]])).all
    assert (indexworks==np.array([[1, 3],[3, 1]], dtype=np.int64)).all
    assert get_cutoffworks==(-5.50885490259143, 29.50885490259143)

def test_outliers_pct():
    from src.strobopy.denoise import outliers
    m=np.arange(25).reshape(5,5)
    quart=5
    uquart=100-quart
    q25, q75 = np.percentile(m, quart), np.percentile(m, uquart)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    m[1,3]=upper+5
    m[3,1]=lower-5
    outliersworks,indexworks,get_cutoffworks=outliers(m,dist='Non-Normal',index=True,get_cutoff=True)
    assert     (outliersworks==np.array([[False, False, False, False, False],
        [False, False, False,  True, False],
        [False, False, False, False, False],
        [False,  True, False, False, False],
        [False, False, False, False, False]])).all
    assert (indexworks==np.array([[1, 3],[3, 1]], dtype=np.int64)).all
    assert get_cutoffworks==(-35.199999999999996, 59.199999999999996)

def test_local_mean():
    from src.strobopy.denoise import local_mean
    m=np.arange(25).reshape(5,5)
    m[1,3]=100
    m[3,1]=-100
    assert local_mean(m,1,3,NN=2)==7.333333333333333

def test_diff_outliers():
    from src.strobopy.denoise import diff_outliers
    m=np.arange(25).reshape(5,5)
    m[1,3]=100
    m[3,1]=-100
    diffoutlerworks=diff_outliers(m)
    indexworks=diff_outliers(m, index=True)
    assert (diffoutlerworks==np.array([[False, False, False, False, False],
       [False, False, False,  True, False],
       [False, False, False, False, False],
       [False,  True, False, False, False],
       [False, False, False, False, False]])).all
    assert (indexworks==np.array([[1, 3],[3, 1]], dtype=np.int64)).all

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


# def test_command_line_interface():
#     """Test the CLI."""
#     runner = CliRunner()
#     result = runner.invoke(cli.main)
#     assert result.exit_code == 0
#     assert 'strobopy.cli.main' in result.output
#     help_result = runner.invoke(cli.main, ['--help'])
#     assert help_result.exit_code == 0
#     assert '--help  Show this message and exit.' in help_result.output
