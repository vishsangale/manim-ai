"""
Multi-head attention visualizations.
"""
from manim import *
from ..core.tensors import SmartTensor
from ..core.annotations import Annotatable

class AttentionHead(VGroup, Annotatable):
    def __init__(self, sequence_len=5, embed_dim=4):
        super().__init__()
        self.query = SmartTensor((1, embed_dim))
        self.keys = SmartTensor((sequence_len, embed_dim))
        
        # Layout: Query on left, Keys on right
        self.query.move_to(LEFT * 3)
        self.keys.move_to(RIGHT * 3)
        self.add(self.query, self.keys)

    def animate_attention_score(self):
        """
        Animation where Query scans every Key to produce scores.
        """
        # Create a "scanner" highlight box
        # We target the first row of keys (which is the first item in the keys VGroup)
        # keys[0] is the first row VGroup
        scanner = SurroundingRectangle(self.keys[0], color=YELLOW)
        
        # Create a Score Vector to store results
        # We'll position it between Query and Keys for visual clarity, or to the right
        scores = SmartTensor((len(self.keys), 1)).move_to(RIGHT * 6)
        # Initially invisible or created as part of the animation? 
        # The user snippet implies it's created here. 
        # Adding it to the scene would be done by the caller or we return it? 
        # The snippet just creates it. We should probably add it to self or return it.
        # But wait, 'scores' is a local variable. The specific snippet:
        # animations.append(Flash(scores[i], color=YELLOW))
        # implies scores[i] exists.
        
        # Let's add scores to the group so it's part of the scene hierarchy if needed
        # Or usually animations return mobjects to add. 
        # For now, I'll follow the snippet closely.
        
        animations = []
        
        # 1. Slide scanner down the keys
        # self.keys is a VGroup of rows (from SmartTensor._build_grid)
        for i, key_row in enumerate(self.keys):
            if i == 0:
                 # Start scanner at first row
                 animations.append(Create(scanner))
            
            animations.append(scanner.animate.move_to(key_row))
            # Flash the corresponding score "cell"
            # scores is a 2D SmartTensor (Nx1), so iterating it gives rows.
            # scores[i] is the i-th row (which has 1 cell).
            animations.append(Flash(scores[i], color=YELLOW))
            
        return Succession(*animations)
