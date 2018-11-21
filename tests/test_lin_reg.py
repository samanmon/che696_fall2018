#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `lin_reg` package."""

import unittest
from click.testing import CliRunner
import numpy as np
from numpy.core.multiarray import ndarray

from lin_reg import lin_reg
from lin_reg.lin_reg import regr
from lin_reg import cli


class TestLin_reg(unittest.TestCase):
    """Tests for `lin_reg` package."""

    # def setUp(self):
    #      print("setup")
    #
    # def tearDown(self):
    #      print("cleanup")

    def test_lin_reg(self):
        x = np.array([2, 5])
        y = np.array([2, 5])  # type: ndarray
        ypred = regr.predict(x)
        ydiff = ypred - y
        self.assertEqual(ypred, np.array([2, 5]))
        self.assertEqual(ydiff, 0)

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'lin_reg.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
