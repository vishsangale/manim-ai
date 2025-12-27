from manim import *
from .attention import AttentionHead
from ..core.tensors import SmartTensor
from ..core.annotations import Annotatable

class TransformerBlock(VGroup, Annotatable):
    """
    Visualizes a simplified Transformer Encoder Layer:
    Input -> Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm
    """
    def __init__(self, embed_dim=4, seq_len=5, **kwargs):
        super().__init__(**kwargs)
        
        # Components
        self.attn = AttentionHead(sequence_len=seq_len, embed_dim=embed_dim)
        self.attn.add_label("Self-Attention", font_size=18)
        
        # Normalization Layer representation (simple rectangle)
        self.norm1 = Rectangle(height=1, width=3, color=GREEN).set_fill(GREEN, opacity=0.2)
        norm1_label = Text("Add & Norm", font_size=16).move_to(self.norm1)
        self.norm1.add(norm1_label)
        
        # Feed Forward Network (represented as a dense block)
        self.ffn = Rectangle(height=1.5, width=3, color=BLUE).set_fill(BLUE, opacity=0.2)
        ffn_label = Text("Feed Forward", font_size=16).move_to(self.ffn)
        self.ffn.add(ffn_label)
        
        # Normalization Layer 2
        self.norm2 = self.norm1.copy()
        
        # Layout: Stacked Vertically (Bottom-Up usually, but standard flow might be Top-Down or Left-Right)
        # Let's do Bottom-Up data flow
        self.attn.move_to(DOWN * 2)
        self.norm1.next_to(self.attn, UP, buff=0.5)
        self.ffn.next_to(self.norm1, UP, buff=0.5)
        self.norm2.next_to(self.ffn, UP, buff=0.5)
        
        self.add(self.attn, self.norm1, self.ffn, self.norm2)
        
        # Arrows
        a1 = Arrow(self.attn.get_top(), self.norm1.get_bottom())
        a2 = Arrow(self.norm1.get_top(), self.ffn.get_bottom())
        a3 = Arrow(self.ffn.get_top(), self.norm2.get_bottom())
        self.add(a1, a2, a3)
