import unittest
from manim import config, VGroup, Text, Brace, Transform, FadeIn
# Prevent Manim from trying to write a movie file or open a window
config.dry_run = True
config.verbosity = "CRITICAL"

from manimai.core.annotations import Annotatable
from manimai.utils.code_window import CodeTracker

class MockAnnotatableItem(VGroup, Annotatable):
    """Refactored mock class for testing Annotatable mixin."""
    pass

class TestComponents(unittest.TestCase):
    def test_annotatable_label(self):
        """Test adding labels via Annotatable mixin"""
        item = MockAnnotatableItem()
        label = item.add_label("Test Label")
        
        self.assertIn(label, item.submobjects, "Label should be added to the item")
        self.assertIsInstance(label, Text, "Label should be a Text object")
        self.assertEqual(label.original_text, "Test Label", "Label text should match")

    def test_annotatable_dims(self):
        """Test adding dimensions via Annotatable mixin"""
        item = MockAnnotatableItem()
        # Create a dummy object to brace (needs width)
        item.add(Text("Dummy")) 
        
        brace, text = item.add_dims("10x10")
        
        self.assertIn(brace, item.submobjects, "Brace should be added")
        self.assertIn(text, item.submobjects, "Dimension text should be added")
        self.assertIsInstance(brace, Brace, "Should be a Brace object")
        self.assertIsInstance(text, Text, "Dimension text should be a Text object")
        self.assertEqual(text.original_text, "10x10", "Dimension text content should match")

    def test_code_tracker_init(self):
        """Test CodeTracker initialization"""
        code_str = "print('hello')\nprint('world')"
        tracker = CodeTracker(code_str)
        
        # Check structure: Background, LineNumbers, CodeLines
        # We know from debugging that the code lines are at index 2
        self.assertEqual(len(tracker.code_obj), 3, "Code object should have 3 components (Background, LineNums, Lines)")
        
        # Check actual lines
        code_lines = tracker.code_obj[2]
        self.assertEqual(len(code_lines), 2, "Should have 2 lines of code")

    def test_code_tracker_highlight(self):
        """Test highlight_line returns an animation"""
        code_str = "line1\nline2"
        tracker = CodeTracker(code_str)
        
        # Highlight line 1
        anim = tracker.highlight_line(1)
        
        # Should return an animation (Transform or FadeIn)
        self.assertTrue(isinstance(anim, (Transform, FadeIn)), "Should return a valid Manim animation")
        
        # Check that highlight was added to tracker.highlights VGroup
        self.assertEqual(len(tracker.highlights), 1, "Highlighter rectangle should be added")

if __name__ == "__main__":
    unittest.main()
