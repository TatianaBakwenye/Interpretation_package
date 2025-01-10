# Import core classes and methods from interpretation
from .interpretation.interpretation import PlotConfig, DataLoader, ModelLoader, ShapVisualizer

# Import core classes and methods from interpretation_time
from .interpretation_time.interpretation_time import ShapleyFeaturePlot

# Package version
__version__ = "1.0.0"

# Expose core components at the package level
__all__ = [
    "PlotConfig",
    "DataLoader",
    "ModelLoader",
    "ShapVisualizer",
    "ShapleyFeaturePlot",
]
