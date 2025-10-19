import numpy as np
import numpy.typing as npt
from typing import TypeAlias, Final
from pathlib import Path

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""
Tener cuidado con imports circulares.
Si este archivo se importa en otros módulos del proyecto,
asegurarse de no importar esos módulos aquí.
"""

# Dtypes abreviados (no pises "float" nativo)
F64: TypeAlias = np.float64
I64: TypeAlias = np.int64
F32: TypeAlias = np.float32
I32: TypeAlias = np.int32
I8:  TypeAlias = np.int8

# Arrays tipados
VecF: TypeAlias = npt.NDArray[F32]     # (N,) o (…)
VecI: TypeAlias = npt.NDArray[I64]
MatF: TypeAlias = npt.NDArray[F32]     # 2D en general
ArrayLike: TypeAlias = npt.ArrayLike
ColorImageU8: TypeAlias  = npt.NDArray[np.uint8]   # (H,W,3) BGR/RGB
GrayImageF32: TypeAlias  = npt.NDArray[F32]        # (H,W)
MaskU8: TypeAlias        = npt.NDArray[np.uint8]   # (H,W) {0,255} o {0,1}

# Dtype numérico del proyecto (punto único de verdad)
DTYPE: Final = np.float32
