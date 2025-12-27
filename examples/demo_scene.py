from manim import *
from manimai.core.tensors import SmartTensor
from manimai.blocks.attention import AttentionHead
from manimai.utils.code_window import CodeTracker
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


class CodeTrackerDemo(Scene):
    def construct(self):
        code = """def forward(self, x):
    # 1. Linear Projection
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    
    # 2. Attention Scores
    scores = q @ k.transpose(-2, -1)
    return scores"""
        
        # Create Code Window
        tracker = CodeTracker(code, language="python")
        tracker.to_edge(RIGHT)
        self.add(tracker)
        
        # Simulate walking through execution
        self.play(tracker.highlight_line(2))
        self.wait(0.5)
        self.play(tracker.highlight_line(3))
        self.wait(0.5)
        self.play(tracker.highlight_line(4))
        self.wait(0.5)
        self.play(tracker.highlight_line(5))
        self.wait(0.5)
        self.play(tracker.highlight_line(8))
        self.wait(0.5)
        self.play(tracker.highlight_line(9))
        self.wait(1)

from manimai.blocks.convolution import Conv2DBlock
from manimai.blocks.transformer import TransformerBlock
from manimai.core.operations import MatrixMultiplication, Softmax

class ConvDemo(Scene):
    def construct(self):
        block = Conv2DBlock()
        block.add_label("Convolution Layer").to_edge(UP)
        self.add(block)
        self.play(block.animate_convolution())
        self.wait(1)

class TransformerDemo(Scene):
    def construct(self):
        block = TransformerBlock()
        # Scale down to fit screen if needed
        block.scale(0.8)
        block.move_to(ORIGIN)
        self.add(block)
        self.play(FadeIn(block))
        self.wait(2)

class OperationsDemo(Scene):
    def construct(self):
        # 1. MatMul
        matmul = MatrixMultiplication()
        matmul.add_label("MatMul Operation").to_edge(UP)
        self.add(matmul)
        self.play(Create(matmul))
        self.wait(1)
        self.play(FadeOut(matmul))
        
        # 2. Softmax
        # Reuse parts of MatMul logic for demo or create new vector
        v = SmartTensor((5,))
        v.move_to(LEFT * 3)
        self.add(v)
        softmax = Softmax(v)
        self.add(softmax)
        self.play(softmax.animate())
        self.wait(1)

