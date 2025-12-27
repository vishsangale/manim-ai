from manim import *

class Annotatable:
    """
    Mixin to add automatic labeling capabilities to any object.
    """
    def add_label(self, text, position=UP, buff=0.2, font_size=24):
        """Adds a static text label (e.g., 'Input Embeddings')"""
        label = Text(text, font_size=font_size)
        label.next_to(self, position, buff=buff)
        self.add(label) # Add to the VGroup so it moves with the object
        return label

    def add_dims(self, shape_text, position=DOWN, color=GRAY):
        """Adds dimension braces (e.g., 'B x T x D')"""
        brace = Brace(self, direction=position)
        # Use Text instead of brace.get_text (which uses Tex/LaTeX) to avoid dependency
        text = Text(str(shape_text), font_size=24).set_color(color)
        text.next_to(brace, position, buff=0.1)
        self.add(brace, text)
        return brace, text
