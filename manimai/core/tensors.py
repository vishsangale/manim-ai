from manim import *
import numpy as np
from .annotations import Annotatable

class SmartTensor(VGroup, Annotatable):
    """
    A visualization of a tensor that knows its shape and data.
    Can render as a Vector (1D), Grid (2D), or Stack (3D).
    """
    def __init__(self, shape, data=None, cell_size=0.5, 
                 tensor_color=BLUE, stroke_color=WHITE, opacity_range=(0.0, 1.0), **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.cell_size = cell_size
        self.tensor_color = tensor_color
        self.stroke_color = stroke_color
        self.opacity_range = opacity_range
        
        # Internal storage for the visual Mobjects (cells)
        # Structure depends on dimension:
        # 1D: List of Squares
        # 2D: List of List of Squares
        # 3D: List of List of List of Cubes
        self.cells = []
        
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

    def _get_opacity(self, value):
        """Map value to opacity based on range."""
        min_op, max_op = self.opacity_range
        # Assuming value is roughly 0-1. If not, we might need normalization?
        # For now, clip and map.
        val_clipped = np.clip(value, 0, 1)
        return min_op + (max_op - min_op) * val_clipped

    def _build_vector(self):
        # 1D vector visualization: a single row of squares
        rows = self.shape[0]
        self.cells = []
        for i in range(rows):
            sq = Square(side_length=self.cell_size)
            val = self.data[i]
            opacity = self._get_opacity(val)
            sq.set_fill(color=self.tensor_color, opacity=opacity)
            sq.set_stroke(self.stroke_color, width=1)
            self.add(sq)
            self.cells.append(sq)
        
        # Arrange them horizontally
        self.arrange(RIGHT, buff=0)
        self.center()

    def _build_grid(self):
        # 2D grid visualization: Rows of squares
        rows, cols = self.shape
        self.cells = []
        for i in range(rows):
            row_cells = []
            row_group = VGroup()  # Create a VGroup for each row
            for j in range(cols):
                sq = Square(side_length=self.cell_size)
                # cell value determines opacity
                val = self.data[i, j]
                opacity = self._get_opacity(val)
                sq.set_fill(color=self.tensor_color, opacity=opacity)
                sq.set_stroke(self.stroke_color, width=1)
                row_group.add(sq)
                row_cells.append(sq)
            
            # Arrange the row horizontally
            row_group.arrange(RIGHT, buff=0)
            self.add(row_group)
            self.cells.append(row_cells)
            
        # Arrange the rows vertically
        self.arrange(DOWN, buff=0)
        self.center()

    def _build_cube(self):
        # 3D stack visualization: Voxel Grid using Cubes
        depth, rows, cols = self.shape
        self.cells = []
        for k in range(depth):
            grid_cells = []
            grid_group = VGroup()  # Create VGroup for the grid (slice)
            for i in range(rows):
                row_cells = []
                row_group = VGroup()  # Create VGroup for each row
                for j in range(cols):
                    voxel = Cube(side_length=self.cell_size)
                    
                    val = self.data[k, i, j]
                    # For 3D, ensure minimum opacity so it's not invisible
                    # Override opacity range slightly for 3D visibility?
                    # Or just rely on user passing good range.
                    opacity = self._get_opacity(val)
                    if opacity < 0.1: opacity = 0.1 # Minimum visibility
                    
                    voxel.set_fill(color=self.tensor_color, opacity=opacity)
                    voxel.set_stroke(self.stroke_color, width=0.5, opacity=0.5)
                    row_group.add(voxel)
                    row_cells.append(voxel)
                
                # Arrange the row horizontally
                row_group.arrange(RIGHT, buff=0)
                grid_group.add(row_group)
                grid_cells.append(row_cells)
            
            # Arrange the grid vertically
            grid_group.arrange(DOWN, buff=0)
            self.add(grid_group)
            self.cells.append(grid_cells)
        
        # Arrange the grids along the Z-axis (OUT)
        self.arrange(OUT, buff=0)
        self.center()

    def update_data(self, new_data):
        """
        Update the internal data and return an animation of the cells changing color.
        """
        self.data = np.array(new_data) # Ensure numpy
        
        dim = len(self.shape)
        anims = []
        
        if dim == 1:
            for i, cell in enumerate(self.cells):
                val = self.data[i]
                new_opacity = self._get_opacity(val)
                anims.append(cell.animate.set_fill(opacity=new_opacity))
        elif dim == 2:
            rows, cols = self.shape
            for i in range(rows):
                for j in range(cols):
                    cell = self.cells[i][j]
                    val = self.data[i, j]
                    new_opacity = self._get_opacity(val)
                    anims.append(cell.animate.set_fill(opacity=new_opacity))
        elif dim == 3:
             depth, rows, cols = self.shape
             for k in range(depth):
                for i in range(rows):
                    for j in range(cols):
                        cell = self.cells[k][i][j]
                        val = self.data[k, i, j]
                        new_opacity = self._get_opacity(val)
                        if new_opacity < 0.1: new_opacity = 0.1
                        anims.append(cell.animate.set_fill(opacity=new_opacity))
                        
        return AnimationGroup(*anims)
        
    def __getitem__(self, item):
        """Access the visual cells or groups conveniently."""
        # This overrides VGroup __getitem__, be careful.
        # But VGroup just acts as a list of Mobjects.
        # We might want to expose cells access.
        # Let's trust VGroup behavior for now, which indexes children.
        # But for 2D, self[0] is the first row VGroup.
        return super().__getitem__(item)

