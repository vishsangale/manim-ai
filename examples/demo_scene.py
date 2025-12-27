from manim import *
from manimai.core.tensors import SmartTensor
from manimai.blocks.attention import AttentionHead
import numpy as np

class SmartTensorDemo(Scene):
    def construct(self):
        # 1. 1D Tensor
        t1_label = Text("1D Tensor (Vector)").to_edge(UP)
        t1 = SmartTensor((5,))
        
        self.add(t1_label)
        self.play(Create(t1))
        self.wait(0.5)
        self.play(FadeOut(t1), FadeOut(t1_label))
        
        # 2. 2D Tensor
        t2_label = Text("2D Tensor (Grid)").to_edge(UP)
        t2 = SmartTensor((3, 4))
        
        self.add(t2_label)
        self.play(Create(t2))
        self.wait(0.5)
        self.play(FadeOut(t2), FadeOut(t2_label))
        
        # 3. 3D Tensor
        t3_label = Text("3D Tensor (Stack)").to_edge(UP)
        t3 = SmartTensor((2, 3, 4))
        # Move it to center
        t3.move_to(ORIGIN)
        
        self.add(t3_label)
        self.play(Create(t3))
        # Rotate to show depth
        self.play(Rotate(t3, angle=PI/4, axis=UP), run_time=2)
        self.wait(0.5)
        self.play(FadeOut(t3), FadeOut(t3_label))

class AttentionDemo(Scene):
    def construct(self):
        title = Text("Attention Mechanism").to_edge(UP)
        self.add(title)
        
        attention = AttentionHead(sequence_len=5, embed_dim=4)
        attention.move_to(ORIGIN)
        self.add(attention)
        
        # Animate the creation and the specific attention mechanism
        self.play(Create(attention.query), Create(attention.keys))
        self.wait(0.5)
        
        self.play(attention.animate_attention_score())
        self.wait(1)
