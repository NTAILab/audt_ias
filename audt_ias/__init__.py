"""Analytic unary density trees and forests.

The package exposes scikit-learn-compatible estimators for nonparametric
density estimation with analytically integrated uniform-background loss.
"""
from ._classification import UnaryDensityBayesClassifier
from ._forest import UnaryDensityForest
from ._tree import UnaryDensityTree, UnaryTree

__version__ = "0.1.0"

__all__ = [
    "UnaryDensityBayesClassifier",
    "UnaryDensityForest",
    "UnaryDensityTree",
    "UnaryTree",
    "__version__",
]
