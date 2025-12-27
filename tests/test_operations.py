import unittest
from manim import config, VGroup, Animation
from manimai.core.tensors import SmartTensor
# Prevent Manim from trying to write a movie file or open a window
config.dry_run = True
config.verbosity = "CRITICAL"

from manimai.core.operations import MatrixMultiplication, Softmax

class TestOperations(unittest.TestCase):
    def test_matmul(self):
        """Test MatrixMultiplication initialization"""
        shape_a = (2, 3)
        shape_b = (3, 4)
        matmul = MatrixMultiplication(shape_a, shape_b)
        
        # Check result shape
        self.assertEqual(matmul.C.shape, (2, 4), "Result matrix C should be (rows_A, cols_B)")
        
        # Check Annotatable inheritance
        label = matmul.add_label("Test MatMul")
        self.assertIn(label, matmul.submobjects)

    def test_softmax(self):
        """Test Softmax initialization and animation"""
        vec = SmartTensor((5,))
        softmax = Softmax(vec)
        
        # Check structure
        self.assertEqual(softmax.probs.shape, (5,))
        
        # Test animation
        anim = softmax.animate()
        self.assertIsInstance(anim, Animation)

if __name__ == "__main__":
    unittest.main()
