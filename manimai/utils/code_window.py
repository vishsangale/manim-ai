from manim import *

class CodeTracker(VGroup):
    def __init__(self, code_string, language="python"):
        super().__init__()
        self.code_obj = Code(
            code_string=code_string,
            language=language,
            background="window"
        )
        self.add(self.code_obj)
        self.highlights = VGroup()
        self.add(self.highlights)

    def highlight_line(self, line_number):
        """
        Returns an animation that moves a highlighter to the specific line.
        """
        # Manim's Code object structure: [background, line_numbers, code_lines]
        # So index 2 contains the actual lines
        target_line = self.code_obj[2][line_number - 1] 
        
        rect = SurroundingRectangle(target_line, color=YELLOW, fill_opacity=0.2, stroke_width=0)
        
        if len(self.highlights) > 0:
            return Transform(self.highlights[0], rect)
        else:
            self.highlights.add(rect)
            return FadeIn(rect)
