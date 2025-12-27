from manim import *
from manimai.core.tensors import SmartTensor
from manimai.blocks.convolution import Conv2DBlock
from manimai.core.operations import MatrixMultiplication

class VerificationScene(Scene):
    def construct(self):
        # 1. Test SmartTensor Update
        tensor = SmartTensor((3, 3), cell_size=0.5)
        tensor.to_edge(UP)
        self.add(tensor)
        
        # Initial wait
        self.wait(1)
        
        # Create new random data
        new_data = np.random.rand(3, 3)
        self.play(tensor.update_data(new_data))
        
        self.wait(1)
        self.remove(tensor)
        
        # 2. Test Convolution Animation
        conv_block = Conv2DBlock(input_shape=(4, 4), kernel_shape=(2, 2))
        conv_block.move_to(ORIGIN)
        self.add(conv_block)
        
        # Run animation
        self.play(conv_block.animate_convolution(run_time=2))
        
        self.wait(1)
        # Clear the scene completely before next step
        self.play(FadeOut(conv_block))
        self.remove(conv_block)
        self.wait(0.5)
        
        # 3. Test Matrix Multiplication Animation
        matmul = MatrixMultiplication(shape_a=(3, 3), shape_b=(3, 2))
        matmul.move_to(ORIGIN)
        self.add(matmul)
        
        self.play(matmul.animate_matmul(run_time=3))
        
        self.wait(2)
