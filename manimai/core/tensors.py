from manim import *
import numpy as np

class SmartTensor(VGroup):
    """
    A visualization of a tensor that knows its shape and data.
    Can render as a Vector (1D), Grid (2D), or Stack (3D).
    """
    def __init__(self, shape, data=None, cell_size=0.5, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.cell_size = cell_size
        
        # If no data provided, fill with zeros or random noise visually
        if data is None:
            self.data = np.random.rand(*shape)
        else:
            self.data = data
            
        self.construct_visuals()

    def construct_visuals(self):
        # Logic to build the VGroup based on len(self.shape)
        if len(self.shape) == 1:
            self._build_vector()
        elif len(self.shape) == 2:
            self._build_grid()
        elif len(self.shape) == 3:
            self._build_cube()

    def _build_vector(self):
        # 1D vector visualization: a single row of squares
        rows = self.shape[0]
        # Since it's 1D, we can just treat it as a single VGroup of squares
        for i in range(rows):
            sq = Square(side_length=self.cell_size)
            val = self.data[i]
            opacity = np.clip(val, 0, 1)
            sq.set_fill(color=BLUE, opacity=opacity)
            sq.set_stroke(WHITE, width=1)
            self.add(sq)
        
        # Arrange them horizontally
        self.arrange(RIGHT, buff=0)
        self.center()

    def _build_grid(self):
        # 2D grid visualization: Rows of squares
        rows, cols = self.shape
        for i in range(rows):
            row_group = VGroup()  # Create a VGroup for each row
            for j in range(cols):
                sq = Square(side_length=self.cell_size)
                # cell value determines opacity
                val = self.data[i, j]
                opacity = np.clip(val, 0, 1)
                sq.set_fill(color=BLUE, opacity=opacity)
                sq.set_stroke(WHITE, width=1)
                row_group.add(sq)
            
            # Arrange the row horizontally
            row_group.arrange(RIGHT, buff=0)
            self.add(row_group)
            
        # Arrange the rows vertically
        self.arrange(DOWN, buff=0)
        self.center()

    def _build_cube(self):
        # 3D stack visualization: Stack of Grids
        depth, rows, cols = self.shape
        for k in range(depth):
            grid_group = VGroup()  # Create VGroup for the grid (slice)
            for i in range(rows):
                row_group = VGroup()  # Create VGroup for each row
                for j in range(cols):
                    sq = Square(side_length=self.cell_size)
                    val = self.data[k, i, j]
                    opacity = np.clip(val, 0, 1)
                    sq.set_fill(color=BLUE, opacity=opacity)
                    sq.set_stroke(WHITE, width=1)
                    row_group.add(sq)
                
                # Arrange the row horizontally
                row_group.arrange(RIGHT, buff=0)
                grid_group.add(row_group)
            
            # Arrange the grid vertically
            grid_group.arrange(DOWN, buff=0)
            self.add(grid_group)
        
        # Arrange the grids along the Z-axis (OUT)
        self.arrange(OUT, buff=0.5)
        self.center()
