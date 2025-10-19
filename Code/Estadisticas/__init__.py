# domain/__init__.py
from .BayesAgent import BayesAgent
from Code.types import F64, I64, VecF, VecI, MatF, ArrayLike, DTYPE

__all__ = [
    "BayesAgent",
    "F64", "I64", "VecF", "VecI", "MatF", "ArrayLike", "DTYPE",
]
__version__ = "1.0.0"
