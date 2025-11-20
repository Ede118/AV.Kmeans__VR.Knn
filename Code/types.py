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

# Alias genéricos
ScalarF: TypeAlias = np.float32
VecF: TypeAlias = npt.NDArray[np.float32]    # (N,)
MatF: TypeAlias = npt.NDArray[np.float32]    # (M, N)

# Alias para Vision Artificial
ImgGray:   TypeAlias = npt.NDArray[np.uint8]     # (H, W), 0–255
ImgColor:  TypeAlias = npt.NDArray[np.uint8]     # (H, W, 3), BGR/RGB
Mask:      TypeAlias = npt.NDArray[np.uint8]     # 0/255 o 0/1
ImgGrayF:  TypeAlias = npt.NDArray[np.float32]   # (H, W), 0–1
ImgColorF: TypeAlias = npt.NDArray[np.float32]   # (H, W, 3), 0–1

# Alias para Reconocimiento de Voz
AudioSignal:   TypeAlias = npt.NDArray[np.float32]  # señal en tiempo
Spectrogram:   TypeAlias = npt.NDArray[np.float32]  # (freq, time)
FeatVec:       TypeAlias = VecF                     # (D,)
FeatMat:       TypeAlias = MatF                     # (N, D)
LabelArray:    TypeAlias = npt.NDArray[np.int64]    # (N,)

# Alias para Agente Bayesiano
ProbVec:     TypeAlias = VecF      # p(c_i)
ProbMat:     TypeAlias = MatF      # p(x | c_i) o similar
LogProbVec:  TypeAlias = VecF
LogProbMat:  TypeAlias = MatF
ClassIdx:    TypeAlias = int       # índice de clase (0..C-1)


# Dtype numérico del proyecto (punto único de verdad)
DTYPE: Final = np.float32
