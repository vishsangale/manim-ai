from manim import *
from ..core.operations import MatrixMultiplication
from ..core.tensors import SmartTensor
from ..core.annotations import Annotatable

class FeedForward(VGroup, Annotatable):
    """
    Visualizes a Dense (Feed Forward) Layer:
    Input (x) -> Weights (W) -> Linear (z = xW) -> Bias (b) -> Activation (a)
    """
    def __init__(self, input_dim=3, output_dim=2, **kwargs):
        super().__init__(**kwargs)
        
        # 1. Matrix Multiplication Part (xW)
        # x: (1, input_dim)
        # W: (input_dim, output_dim)
        self.matmul = MatrixMultiplication(
            shape_a=(1, input_dim), 
            shape_b=(input_dim, output_dim)
        )
        self.matmul.A.add_label("x")
        self.matmul.B.add_label("W")
        self.matmul.C.add_label("z")
        
        # 2. Bias Addition
        self.bias = SmartTensor((1, output_dim))
        self.bias.add_label("b")
        self.plus_sign = Text("+")
        
        # 3. Activation
        self.activation_label = Text("ReLU", font_size=20, color=YELLOW)
        self.arrow = Arrow(LEFT, RIGHT, buff=0.1)
        self.output = SmartTensor((1, output_dim))
        self.output.add_label("y")
        
        # Layout
        # matmulGroup is [A, B, =, C]
        # We want [MatMul] + [Bias] -> [Output]
        
        # Align: MatMul ... + Bias -> Output
        self.matmul.move_to(LEFT * 2)
        
        self.plus_sign.next_to(self.matmul, RIGHT, buff=0.2)
        self.bias.next_to(self.plus_sign, RIGHT, buff=0.2)
        
        self.arrow.next_to(self.bias, RIGHT, buff=0.3)
        self.activation_label.next_to(self.arrow, UP, buff=0.1)
        
        self.output.next_to(self.arrow, RIGHT, buff=0.3)
        
        self.add(self.matmul, self.plus_sign, self.bias, self.arrow, self.activation_label, self.output)

    def animate_forward_pass(self, run_time=6.0):
        """
        Sequentially animate:
        1. MatMul (xW = z)
        2. Bias addition (z + b) visualize as adding values? Or just appearing.
        3. Activation (z+b -> y)
        """
        
        # 1. MatMul
        # animate_matmul returns a Succession
        t_matmul = run_time * 0.5
        anim_1 = self.matmul.animate_matmul(run_time=t_matmul)
        
        # 2. Bias
        t_bias = run_time * 0.25
        # Show bias appearing and then "merging" into result?
        # Simpler: Just Indicate bias and Transform C -> Output (intermediate)
        # Let's say z (matmul.C) transforms into y (output)
        
        # We can simulate the addition by transforming C to match Output's position?
        # Or just show everything.
        
        anim_2 = AnimationGroup(
            FadeIn(self.plus_sign),
            Create(self.bias),
            run_time=t_bias
        )
        
        # 3. Activation
        t_act = run_time * 0.25
        # Transform (C + Bias) visual composite into Output?
        # Let's just create the arrow and output "popping" out.
        
        anim_3 = AnimationGroup(
            Create(self.arrow),
            Write(self.activation_label),
            TransformFromCopy(self.matmul.C, self.output), # Data flows from z to y
            run_time=t_act
        )
        
        return Succession(
            anim_1,
            anim_2,
            anim_3
        )
