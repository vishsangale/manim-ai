"""
Manimai Demo Scenes
===================

This module enables users to visualize the core components of the Manimai library.
Run any scene using the command:
    manim -ql examples/demo_scene.py <SceneName>

Available Scenes:
- SmartTensorDemo: Visualization of 1D, 2D, and 3D tensors.
- AttentionDemo: Visualization of the Multi-Head Attention mechanism.
- CodeTrackerDemo: Visualization of code execution and highlighting.
- ConvDemo: Visualization of a 2D Convolution layer.
- TransformerDemo: Visualization of a Transformer Encoder block.
- OperationsDemo: Visualization of Matrix Multiplication and Softmax operations.
"""

from manim import *
import numpy as np

from manimai.core.tensors import SmartTensor
from manimai.utils.code_window import CodeTracker
from manimai.blocks.attention import AttentionHead
from manimai.blocks.convolution import Conv2DBlock
from manimai.blocks.transformer import TransformerBlock
from manimai.core.operations import MatrixMultiplication, Softmax

class SmartTensorDemo(ThreeDScene):
    """
    Demonstrates the SmartTensor class for 1D, 2D, and 3D data.
    """
    def construct(self):
        title = Text("SmartTensor Visualization", font_size=40).to_edge(UP)
        self.add(title)

        # 1. 1D Tensor (Vector)
        t1 = SmartTensor((5,))
        t1.add_label("1D Tensor (Vector)", position=UP)
        t1.add_dims("n=5")
        t1.move_to(ORIGIN)
        
        self.play(Create(t1))
        self.wait(1)
        self.play(FadeOut(t1))
        
        # 2. 2D Tensor (Grid)
        t2 = SmartTensor((3, 4))
        t2.add_label("2D Tensor (Grid)", position=UP)
        t2.add_dims("3x4")
        t2.move_to(ORIGIN)
        
        self.play(Create(t2))
        self.wait(1)
        self.play(FadeOut(t2))
        
        # 3. 3D Tensor (Cube/Stack)
        t3 = SmartTensor((3, 3, 3)) # Slightly smaller for demo
        t3.add_label("3D Tensor (Volume)", position=UP, buff=0.5)
        t3.move_to(ORIGIN)
        
        # Initialize viewing angle
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        self.play(Create(t3))
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(t3))
        
        # Reset camera
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)
        self.play(FadeOut(title))

class AttentionDemo(Scene):
    """
    Demonstrates the AttentionHead block and its scoring animation.
    """
    def construct(self):
        title = Text("Multi-Head Attention", font_size=40).to_edge(UP)
        self.add(title)
        
        attention = AttentionHead(sequence_len=5, embed_dim=4)
        attention.add_label("Self-Attention Layer", font_size=24, position=DOWN)
        attention.move_to(ORIGIN)
        
        self.play(Create(attention.query), Create(attention.keys))
        self.add(attention) # Add the parent group just in case
        self.wait(0.5)
        
        self.play(attention.animate_attention_score())
        self.wait(1)

class CodeTrackerDemo(Scene):
    """
    Demonstrates the CodeTracker utility for synchronized code highlighting.
    """
    def construct(self):
        code_source = """def forward(self, x):
    # 1. Linear Projection
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    
    # 2. Attention Scores
    scores = q @ k.transpose(-2, -1)
    return scores"""
        
        tracker = CodeTracker(code_source, language="python")
        tracker.to_edge(RIGHT)
        
        # Scale for visibility
        tracker.scale(0.8)
        
        self.add(tracker)
        
        # Step-by-step execution simulation
        steps = [2, 3, 4, 5, 8, 9]
        for line in steps:
            self.play(tracker.highlight_line(line), run_time=0.8)
            self.wait(0.3)

class ConvDemo(Scene):
    """
    Demonstrates the Conv2DBlock visualization.
    """
    def construct(self):
        title = Text("2D Convolution", font_size=40).to_edge(UP)
        self.add(title)
        
        block = Conv2DBlock(input_shape=(5, 5), kernel_shape=(3, 3))
        block.add_label("Convolution Calculation", position=DOWN)
        block.move_to(ORIGIN)
        
        self.play(block.animate_convolution())
        self.wait(2)

class TransformerDemo(Scene):
    """
    Demonstrates the composition of a Transformer Block.
    """
    def construct(self):
        title = Text("Transformer Encoder Block", font_size=40).to_edge(UP)
        self.add(title)
        
        block = TransformerBlock()
        block.scale(0.7) # Fit to screen
        block.move_to(ORIGIN)
        
        # Animate components appearing from bottom to top
        # Hierarchy: attn -> norm1 -> ffn -> norm2
        self.play(FadeIn(block.attn, shift=UP))
        self.wait(0.3)
        self.play(FadeIn(block.norm1, shift=UP))
        self.wait(0.3)
        self.play(FadeIn(block.ffn, shift=UP))
        self.wait(0.3)
        self.play(FadeIn(block.norm2, shift=UP))
        self.wait(1)

class OperationsDemo(Scene):
    """
    Demonstrates Matrix Multiplication and Softmax operations.
    """
    def construct(self):
        # 1. Matrix Multiplication
        title = Text("Matrix Multiplication", font_size=40).to_edge(UP)
        self.add(title)
        
        matmul = MatrixMultiplication()
        matmul.move_to(ORIGIN)
        
        self.play(Create(matmul))
        self.wait(2)
        self.play(FadeOut(matmul), FadeOut(title))
        
        # 2. Softmax
        title2 = Text("Softmax Activation", font_size=40).to_edge(UP)
        self.add(title2)
        
        input_vec = SmartTensor((5,))
        input_vec.add_label("Logits", position=DOWN)
        
        softmax = Softmax(input_vec)
        softmax.move_to(ORIGIN)
        
        self.add(input_vec) 
        self.play(softmax.animate())
        self.wait(2)
