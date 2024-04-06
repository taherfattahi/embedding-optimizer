"""
Embedding Optimizer.

Strategies for Efficient Data Embedding: Creating Embeddings Optimized for Accuracy - Creating Embeddings Optimized for Storage --- python library for OpenAI
"""
from .optimizer import EmbeddingOptimizer

__version__ = "0.0.1"
__author__ = 'Taher Fattahi'
__author_email__ = "taherfattahi11@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/taherfattahi/embedding-optimizer"

PYPI_SIMPLE_ENDPOINT: str = "https://pypi.org/project/embedding-optimizer"

__all__ = [
    "EmbeddingOptimizer",
    "PYPI_SIMPLE_ENDPOINT",
]