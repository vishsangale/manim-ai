from manim import *
from manimai.core.tensors import SmartTensor
from manimai.blocks.attention import AttentionHead
import numpy as np

class SmartTensorDemo(Scene):
    def construct(self):
        # 1. 1D Tensor
        t1 = SmartTensor((5,))
        t1.add_label("1D Tensor (Vector)")
        t1.add_dims("n=5")
        
        self.play(Create(t1))
        self.wait(0.5)
        self.play(FadeOut(t1))
        
        # 2. 2D Tensor
        t2 = SmartTensor((3, 4))
        t2.add_label("2D Tensor (Grid)")
        t2.add_dims("3x4")
        
        self.play(Create(t2))
        self.wait(0.5)
        self.play(FadeOut(t2))
        
        # 3. 3D Tensor
        t3 = SmartTensor((2, 3, 4))
        t3.add_label("3D Tensor (Stack)")
        # Dimensions for 3D might look tricky with standard Brace, but let's try
        # t3.add_dims("2x3x4") 
        # Move it to center
        t3.move_to(ORIGIN)
        
        self.play(Create(t3))
        # Rotate to show depth
        self.play(Rotate(t3, angle=PI/4, axis=UP), run_time=2)
        self.wait(0.5)
        self.play(FadeOut(t3))

class AttentionDemo(Scene):
    def construct(self):
        attention = AttentionHead(sequence_len=5, embed_dim=4)
        attention.add_label("Attention Mechanism", font_size=36)
        attention.move_to(ORIGIN)
        self.add(attention)
        
        # Animate the creation and the specific attention mechanism
        self.play(Create(attention.query), Create(attention.keys))
        self.wait(0.5)
        
        self.play(attention.animate_attention_score())
        self.wait(1)

