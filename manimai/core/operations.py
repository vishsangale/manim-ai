"""
Animation operations (MatMul, Convolve, Softmax).
"""
from manim import *
from .tensors import SmartTensor
from .annotations import Annotatable

class MatrixMultiplication(VGroup, Annotatable):
    """
    Visualizes Matrix Multiplication A x B = C
    """
    def __init__(self, shape_a=(3, 4), shape_b=(4, 2), **kwargs):
        super().__init__(**kwargs)
        self.A = SmartTensor(shape_a)
        self.A.add_label("A")
        self.B = SmartTensor(shape_b)
        self.B.add_label("B")
        self.C = SmartTensor((shape_a[0], shape_b[1]))
        self.C.add_label("C")
        
        # Layout: A slightly left, B next to it, = C
        self.A.move_to(LEFT * 3)
        self.B.next_to(self.A, RIGHT, buff=0.5)
        
        eq = Text("=")
        eq.next_to(self.B, RIGHT, buff=0.5)
        
        self.C.next_to(eq, RIGHT, buff=0.5)
        
        self.add(self.A, self.B, eq, self.C)

    def animate_matmul(self, run_time=5.0):
        """
        Animates Matrix Multiplication:
        For each cell in C(i, j):
            - Highlight Row i of A
            - Highlight Col j of B
            - "Flow" to C(i, j)
            - Reveal C(i, j)
        """
        anims = []
        rows_A, cols_A = self.A.shape
        rows_B, cols_B = self.B.shape
        # C shape is (rows_A, cols_B)
        
        # We need A * B, so cols_A must equal rows_B
        if cols_A != rows_B:
            # Just a fallback warning, though typically shapes are valid
            print(f"Warning: dimension mismatch {cols_A} != {rows_B}")

        total_cells = rows_A * cols_B
        step_time = run_time / total_cells
        
        # Create Highlighters
        # A's highlighter: A row rectangle
        # B's highlighter: A col rectangle
        # Since cells are squares, we can just VGroup the row's cells and surround them?
        # Yes, self.A.cells[i] is a list of cells for row i.
        
        # Initial positions (dummy)
        row_rect = SurroundingRectangle(VGroup(*self.A.cells[0]), color=YELLOW, buff=0.1)
        # For columns in B, we need to gather cells: B.cells[k][j] for all k
        col_group_0 = VGroup(*[self.B.cells[k][0] for k in range(rows_B)])
        col_rect = SurroundingRectangle(col_group_0, color=YELLOW, buff=0.1)
        
        c_highlight = SurroundingRectangle(self.C.cells[0][0], color=YELLOW, buff=0.1)
        
        anims.append(Create(row_rect))
        anims.append(Create(col_rect))
        anims.append(FadeIn(c_highlight))
        
        matmul_anims = []
        
        for i in range(rows_A):
            for j in range(cols_B):
                # Target Row Group (A)
                row_group = VGroup(*self.A.cells[i])
                
                # Target Col Group (B)
                col_group = VGroup(*[self.B.cells[k][j] for k in range(rows_B)])
                
                # Target Cell (C)
                c_cell = self.C.cells[i][j]
                
                # Moves
                move_row = row_rect.animate(run_time=step_time).move_to(row_group)
                # Reshape row_rect?
                # SurroundingRectangle doesn't automatically resize on move_to.
                # Better to Transform it to a new SurroundingRectangle?
                # Or use replacement transform.
                # For simplicity, if grid is regular, move_to works if size is same.
                # Rows are same size always. Cols are same size always.
                # So move_to is safe logic-wise.
                
                move_col = col_rect.animate(run_time=step_time).move_to(col_group)
                move_res = c_highlight.animate(run_time=step_time).move_to(c_cell)
                
                # Compute effect
                compute = Flash(c_cell, color=YELLOW, run_time=step_time, flash_radius=0.2)
                activation = c_cell.animate.set_fill(opacity=1.0)
                
                # Group for this step
                if i==0 and j==0:
                    matmul_anims.append(
                        AnimationGroup(
                             move_row, # Even if already there, safe to call
                             move_col,
                             move_res,
                             compute,
                             activation
                        )
                    )
                else:
                     matmul_anims.append(
                        AnimationGroup(
                            move_row,
                            move_col,
                            move_res,
                            compute,
                            activation
                        )
                    )
        
        anims.append(Succession(*matmul_anims))
        anims.append(FadeOut(row_rect))
        anims.append(FadeOut(col_rect))
        anims.append(FadeOut(c_highlight))
        
        return Succession(*anims)

class Softmax(VGroup, Annotatable):
    """
    Visualizes Softmax operation on a vector.
    Shows raw scores turning into probabilities (bars summing to 1).
    """
    def __init__(self, input_vector, **kwargs):
        super().__init__(**kwargs)
        self.input_vector = input_vector
        # Placeholder visual: Just an arrow pointing to a 'probs' vector
        self.arrow = Arrow(LEFT, RIGHT)
        self.probs = SmartTensor(input_vector.shape)
        self.probs.add_label("Softmax")
        
        self.arrow.next_to(input_vector, RIGHT)
        self.probs.next_to(self.arrow, RIGHT)
        
        self.add(self.arrow, self.probs)

    def animate(self):
        return Succession(
             Create(self.arrow),
             Transform(self.input_vector.copy(), self.probs) # Simple transform idea
        )
