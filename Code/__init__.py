# Re-exportar tipos para uso cómodo en todo el proyecto
from .types import (
    PROJECT_ROOT,
    DTYPE,
    ScalarF, VecF, MatF,
    ImgGray, ImgColor, Mask, ImgGrayF, ImgColorF,
    AudioSignal, Spectrogram,
    FeatVec, FeatMat, LabelArray,
    ProbVec, ProbMat, LogProbVec, LogProbMat,
    ClassIdx,
)

__all__ = [
    "PROJECT_ROOT",
    "DTYPE",
    "ScalarF", "VecF", "MatF",
    "ImgGray", "ImgColor", "Mask", "ImgGrayF", "ImgColorF",
    "AudioSignal", "Spectrogram",
    "FeatVec", "FeatMat", "LabelArray",
    "ProbVec", "ProbMat", "LogProbVec", "LogProbMat",
    "ClassIdx",
]

