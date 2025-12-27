from manim import *
from ..core.tensors import SmartTensor
from ..core.annotations import Annotatable

class Conv2DBlock(VGroup, Annotatable):
    """
    Visualizes a 2D Convolution operation.
    Displays Input -> Kernel -> Output.
    """
    def __init__(self, input_shape=(5, 5), kernel_shape=(3, 3), **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        
        # Calculate output shape (valid padding)
        out_rows = input_shape[0] - kernel_shape[0] + 1
        out_cols = input_shape[1] - kernel_shape[1] + 1
        self.output_shape = (out_rows, out_cols)
        
        self._construct_block()

    def _construct_block(self):
        # 1. Input Feature Map
        self.input_tensor = SmartTensor(self.input_shape)
        self.input_tensor.add_label("Input", font_size=20)
        
        # 2. Kernel (displayed separately or floating)
        self.kernel_tensor = SmartTensor(self.kernel_shape)
        self.kernel_tensor.set_color(ORANGE) # Distinct color for kernel
        self.kernel_tensor.add_label("Kernel", font_size=20)
        
        # 3. Output Feature Map
        self.output_tensor = SmartTensor(self.output_shape)
        self.output_tensor.add_label("Output", font_size=20)
        
        # Layout: Input --(conv)--> Output
        # We place Kernel in between
        self.input_tensor.move_to(LEFT * 4)
        self.kernel_tensor.move_to(ORIGIN)
        self.output_tensor.move_to(RIGHT * 4)
        
        # Add visual arrow
        arrow1 = Arrow(self.input_tensor.get_right(), self.kernel_tensor.get_left(), buff=0.5)
        arrow2 = Arrow(self.kernel_tensor.get_right(), self.output_tensor.get_left(), buff=0.5)
        
        self.add(self.input_tensor, self.kernel_tensor, self.output_tensor, arrow1, arrow2)

    def animate_convolution(self):
        """
        Animating the sliding window effect is complex. 
        For now, we return a simple creation animation.
        """
        # A placeholder for a more complex "sliding" animation later
        return Succession(
            Create(self.input_tensor),
            Create(self.kernel_tensor),
            Create(self.output_tensor)
        )
