# Manimai

**Manimai** is a Python library for creating animations of AI blocks using [Manim](https://www.manim.community/). It provides high-level abstractions like `SmartTensor` and blocks for visualizing Neural Network components like Attention Mechanisms.

## Features

- **SmartTensor**: A base class for tensors that knows its own shape and data, capable of visualizing itself as a:
  - 1D Vector (Row of squares)
  - 2D Grid (Matrix of squares)
  - 3D Stack (Stack of Grids, experimentally supported)
- **AttentionHead**: Visualize the "Searchlight" effect of attention mechanisms, scanning Query vectors against Key vectors to produce Attention Scores.

## Installation

To install the package in editable mode (recommended for development):

```bash
git clone https://github.com/vishsangale/manim-ai.git
cd manim-ai
pip install -e .
```

You will also need [ffmpeg](https://ffmpeg.org/) installed on your system as it is a requirement for Manim.

## Usage

### Using SmartTensor

```python
from manim import *
from manimai.core.tensors import SmartTensor

class MyTensorScene(Scene):
    def construct(self):
        # Create a 2D SmartTensor
        tensor = SmartTensor((3, 4))
        self.add(tensor)
        self.play(Create(tensor))
```

### Visualizing Attention

```python
from manim import *
from manimai.blocks.attention import AttentionHead

class AttentionScene(Scene):
    def construct(self):
        attention = AttentionHead(sequence_len=5, embed_dim=4)
        self.add(attention)
        self.play(attention.animate_attention_score())
```

## Testing

Run the tests using `unittest`:

```bash
python -m unittest discover tests
```
