import unittest
from manim import config
# Prevent Manim from trying to write a movie file or open a window
config.dry_run = True
config.verbosity = "CRITICAL"

from manimai.core.tensors import SmartTensor
import numpy as np

class TestSmartTensor(unittest.TestCase):
    def test_1d_tensor(self):
        """Test 1D Vector structure"""
        t1 = SmartTensor((5,))
        self.assertEqual(len(t1), 5, "1D Tensor should have 5 children (Squares)")
        
    def test_2d_tensor(self):
        """Test 2D Grid structure"""
        rows, cols = 3, 4
        t2 = SmartTensor((rows, cols))
        self.assertEqual(len(t2), rows, f"2D Tensor should have {rows} rows")
        for i, row in enumerate(t2):
            self.assertEqual(len(row), cols, f"Row {i} should have {cols} columns")

    def test_3d_tensor(self):
        """Test 3D Stack structure"""
        depth, rows, cols = 2, 3, 4
        t3 = SmartTensor((depth, rows, cols))
        self.assertEqual(len(t3), depth, f"3D Tensor should have {depth} grids")
        for k, grid in enumerate(t3):
            self.assertEqual(len(grid), rows, f"Grid {k} should have {rows} rows")
            for i, row in enumerate(grid):
                self.assertEqual(len(row), cols, f"Grid {k} Row {i} should have {cols} columns")

if __name__ == "__main__":
    unittest.main()
