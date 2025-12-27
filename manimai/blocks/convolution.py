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

    def animate_convolution(self, run_time=5.0):
        """
        Animates the sliding window convolution process.
        Returns a Succession of animations.
        """
        # 1. Reveal everything initially? Or build it?
        # Let's assume the user has added the block to scene or we return a Creation first.
        # But usually 'animate_convolution' describes the action.
        
        anims = []
        
        # Dimensions
        k_rows, k_cols = self.kernel_shape
        out_rows, out_cols = self.output_shape
        
        # Create a sliding window (Rectangle)
        # We need the size of the kernel in terms of local coordinates.
        # We can just get the VGroup of the first k_rows*k_cols cells?
        # Simpler: Get the top-left cell at (0,0) and bottom-right at (k_rows-1, k_cols-1)
        # and surround them.
        
        # Helper to get sub-grid bounding box logic would be nice, but we can do it manually.
        # input_tensor.cells is now available thanks to our refactor!
        
        # Access cells: self.input_tensor.cells[row][col]
        tl_cell = self.input_tensor.cells[0][0]
        br_cell = self.input_tensor.cells[k_rows-1][k_cols-1]
        
        # Create window rectangle
        window = SurroundingRectangle(
            VGroup(tl_cell, br_cell),
            color=YELLOW,
            buff=0.1
        )
        
        # Create output highlighter (single cell)
        out_highlight = SurroundingRectangle(
            self.output_tensor.cells[0][0],
            color=YELLOW,
            buff=0.1
        )
        
        # Hide output cells initially to reveal them?
        # Or just flash/indicate them.
        # Let's set output tensor opacity to near zero initially? 
        # Or we can just Flash/FadeIn each cell.
        # Let's assume they are visible but we "activate" them.
        
        anims.append(Create(window))
        anims.append(FadeIn(out_highlight))
        
        step_time = run_time / (out_rows * out_cols)
        
        sliding_anims = []
        
        for i in range(out_rows):
            for j in range(out_cols):
                # Calculate window position for Input
                # Top-Left of window is at (i, j)
                # Bottom-Right is at (i + k_rows - 1, j + k_cols - 1)
                
                curr_tl = self.input_tensor.cells[i][j]
                curr_br = self.input_tensor.cells[i + k_rows - 1][j + k_cols - 1]
                target_window_group = VGroup(curr_tl, curr_br)
                
                # Output cell position
                curr_out_cell = self.output_tensor.cells[i][j]
                
                # Move Window and Highlighter
                move_window = window.animate(run_time=step_time).move_to(target_window_group)
                move_highlight = out_highlight.animate(run_time=step_time).move_to(curr_out_cell)
                
                # "Compute" effect - flash the output cell
                # We can also flash the kernel or something to show interaction
                compute_effect = Flash(curr_out_cell, color=YELLOW, run_time=step_time, flash_radius=0.2)
                
                # Add a "Write" or "FadeIn" effect specifically for the output cell content
                # reusing our update_data mechanism? 
                # Or just emphasis.
                activate_cell = curr_out_cell.animate.set_fill(opacity=1.0) # Assuming it started dim
                
                # Combine for this step
                # If it's the very first one, we just create, don't move.
                if i==0 and j==0:
                    sliding_anims.append(
                        AnimationGroup(
                            compute_effect,
                            activate_cell
                        )
                    )
                else:
                    sliding_anims.append(
                        AnimationGroup(
                            move_window,
                            move_highlight,
                            compute_effect,
                            activate_cell
                        )
                    )
                    
        anims.append(Succession(*sliding_anims))
        anims.append(FadeOut(window))
        anims.append(FadeOut(out_highlight))
        
        return Succession(*anims)

