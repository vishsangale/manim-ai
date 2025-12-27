"""
Animation operations (MatMul, Convolve, Softmax).
"""
from manim import *
from .tensors import SmartTensor
from .annotations import Annotatable

class MatrixMultiplication(VGroup, Annotatable):
    """
    Visualizes Matrix Multiplication A x B = C
    """
    def __init__(self, shape_a=(3, 4), shape_b=(4, 2), **kwargs):
        super().__init__(**kwargs)
        self.A = SmartTensor(shape_a).add_label("A")
        self.B = SmartTensor(shape_b).add_label("B")
        self.C = SmartTensor((shape_a[0], shape_b[1])).add_label("C")
        
        # Layout: A slightly left, B next to it, = C
        self.A.move_to(LEFT * 3)
        self.B.next_to(self.A, RIGHT, buff=0.5)
        
        eq = Text("=")
        eq.next_to(self.B, RIGHT, buff=0.5)
        
        self.C.next_to(eq, RIGHT, buff=0.5)
        
        self.add(self.A, self.B, eq, self.C)

class Softmax(VGroup, Annotatable):
    """
    Visualizes Softmax operation on a vector.
    Shows raw scores turning into probabilities (bars summing to 1).
    """
    def __init__(self, input_vector, **kwargs):
        super().__init__(**kwargs)
        self.input_vector = input_vector
        # Placeholder visual: Just an arrow pointing to a 'probs' vector
        self.arrow = Arrow(LEFT, RIGHT)
        self.probs = SmartTensor(input_vector.shape)
        self.probs.add_label("Softmax")
        
        self.arrow.next_to(input_vector, RIGHT)
        self.probs.next_to(self.arrow, RIGHT)
        
        self.add(self.arrow, self.probs)

    def animate(self):
        return Succession(
             Create(self.arrow),
             Transform(self.input_vector.copy(), self.probs) # Simple transform idea
        )
