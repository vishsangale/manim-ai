from manim import *

# Import our blocks
from manimai.core.tensors import SmartTensor
from manimai.blocks.convolution import Conv2DBlock
from manimai.core.operations import MatrixMultiplication
from manimai.blocks.feed_forward import FeedForward

class SmartTensorDemo(Scene):
    def construct(self):
        title = Text("SmartTensor: Dynamic Updates", font_size=36).to_edge(UP)
        self.add(title)
        
        # 1. Create a 3x3 Tensor
        tensor = SmartTensor((3, 3))
        self.play(Create(tensor))
        self.wait(1)
        
        # 2. Update with new random data
        new_data = np.random.rand(3, 3)
        self.play(tensor.update_data(new_data))
        self.wait(1)
        
        # 3. Another update
        new_data_2 = np.random.rand(3, 3)
        self.play(tensor.update_data(new_data_2))
        self.wait(2)

class ConvolutionDemo(Scene):
    def construct(self):
        title = Text("2D Convolution: Sliding Window", font_size=36).to_edge(UP)
        self.add(title)
        
        conv = Conv2DBlock(input_shape=(5, 5), kernel_shape=(3, 3))
        # Add to scene
        self.add(conv)
        self.wait(1)
        
        # Animate
        self.play(conv.animate_convolution(run_time=6))
        self.wait(2)

class MatMulDemo(Scene):
    def construct(self):
        title = Text("Matrix Multiplication: Row x Col", font_size=36).to_edge(UP)
        self.add(title)
        
        matmul = MatrixMultiplication(shape_a=(3, 3), shape_b=(3, 2))
        self.add(matmul)
        self.wait(1)
        
        # Animate
        self.play(matmul.animate_matmul(run_time=5))
        self.wait(2)

class FeedForwardDemo(Scene):
    def construct(self):
        title = Text("FeedForward Layer: xW + b", font_size=36).to_edge(UP)
        self.add(title)
        
        ff = FeedForward(input_dim=4, output_dim=3)
        self.add(ff)
        self.wait(1)
        
        self.play(ff.animate_forward_pass(run_time=6))
        self.wait(2)
