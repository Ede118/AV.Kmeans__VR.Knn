# domain/BayesAgent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Sequence
import numpy as np

from Code.types import VecF, VecI, MatF, ArrayLike, DTYPE

# Fallback si SciPy no está disponible
try:
    from scipy.special import logsumexp as _logsumexp  # estable
except Exception:
    def _logsumexp(s: VecF) -> float:
        smax = float(np.max(s))
        return smax + float(np.log(np.sum(np.exp(s - smax))))

    # --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class BayesAgent:
    """Bayesian inference helper that works safely in probability or log space."""

    K: int      # n° de hipótesis
    C: int      # n° de categorias


    # --------------------------------------------------------------------------------------------- #

    def __post_init__(self) -> None:
            if not (isinstance(self.K, int) and self.K > 0):
                raise ValueError(f"K debe ser int>0, vino {self.K!r}")
            if not (isinstance(self.C, int) and self.C > 0):
                raise ValueError(f"C debe ser int>0, vino {self.C!r}")
            

    # --------------------------------------------------------------------------------------------- #

    def _softmax_from_logs(
            self,
            s: VecF
        ) -> VecF:
        log_norm = _logsumexp(s)
        return np.exp(s - log_norm).astype(np.float64, copy=False)

    # --------------------------------------------------------------------------------------------- #

    def posterior(
            self,
            vecPi: VecF,           # (K,)
            Hipotesis_M: MatF,     # (K, C)
            vecN: VecI,            # (C,)
            *,
            use_logs: bool = True,
            strict_zeros: bool = True
        ) -> VecF:
        
        """Compute the posterior distribution given priors, likelihood matrix and counts."""
        # Estandarizar tipos y shapes
        pi = np.asarray(vecPi, DTYPE).reshape(-1)             # (K,)
        P = np.asarray(Hipotesis_M, DTYPE)                    # (K,C)
        n = np.asarray(vecN, np.int64).reshape(-1)            # (C,)

        # Chequeos de contrato
        if P.shape != (self.K, self.C):
            raise ValueError(f"P debe ser ({self.K},{self.C}), vino {P.shape}")
        if pi.shape != (self.K,):
            raise ValueError(f"pi debe ser ({self.K},), vino {pi.shape}")
        if n.shape  != (self.C,):
            raise ValueError(f"n debe ser ({self.C},), vino {n.shape}")
        if not np.all(n >= 0):
            raise ValueError("n_i deben ser enteros no negativos")
        if not np.isclose(pi.sum(), 1.0, atol=1e-8):
            raise ValueError("pi debe sumar 1")
        if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("cada fila de P debe sumar 1")

        # Cálculo
        if use_logs:
            # Manejo de ceros: imposibles exactos quedan en -inf
            if strict_zeros:
                with np.errstate(divide="ignore"):
                    logP = np.where(P > 0, np.log(P.astype(np.float64, copy=False)), -np.inf)
                    logpi = np.where(pi > 0, np.log(pi.astype(np.float64, copy=False)), -np.inf)
            else:
                eps = np.finfo(np.float64).tiny
                logP = np.log(np.clip(P.astype(np.float64, copy=False), eps, 1.0))
                logpi = np.log(np.clip(pi.astype(np.float64, copy=False), eps, 1.0))

            # s_k = log pi_k + sum_i n_i * log P[k,i]
            s = logpi + (logP @ n.astype(np.float64))           # (K,)
            vecPost = self._softmax_from_logs(s)
            vecPost = vecPost / vecPost.sum(dtype=np.float64)
            return vecPost.astype(DTYPE, copy=False)

        else:
            # Forma “producto directo” (menos estable para m grande)
            # P ** n hace broadcasting: eleva cada columna i a n_i
            weights = pi.astype(np.float64) * np.prod(
                np.power(P.astype(np.float64), n, dtype=np.float64),
                axis=1,
                dtype=np.float64,
            )
            Z = weights.sum()
            if Z == 0.0:
                # todos cero por ceros imposibles; degradamos con clip suave
                eps = 1e-300
                weights = pi.astype(np.float64) * np.prod(
                    np.power(np.clip(P.astype(np.float64), eps, 1.0), n, dtype=np.float64),
                    axis=1,
                    dtype=np.float64,
                )
                Z = weights.sum()
                if Z == 0.0:
                    raise FloatingPointError("Ponderaciones nulas; revisar P y n")
            posterior = weights / Z
            return posterior.astype(DTYPE, copy=False)

    # --------------------------------------------------------------------------------------------- #

    def decide(
        self,
        vecPost: VecF,
        labels: Sequence[str] | None = None,
        *,
        tie: Literal["first", "random", "all"] = "first",
        tol: float = 1e-12,
        rng: np.random.Generator | None = None,
    ) -> str | list[str]:
        """
        Elige la hipótesis más probable según el vector posterior.
        - vecPost: (K,) con posteriori (debe sumar ≈ 1 y ser finito).
        - labels: etiquetas para cada hipótesis; por defecto ['A','B',...].
        - tie: política de empate:
            * 'first'  -> devuelve la primera con máximo (determinista)
            * 'random' -> rompe el empate al azar
            * 'all'    -> devuelve TODAS las empatadas
        - tol: tolerancia absoluta para considerar empate.
        - rng: generador para empates aleatorios (si no, default_rng()).
        """
        p = np.asarray(vecPost, DTYPE).reshape(-1)  # (K,)

        # Chequeos básicos
        if p.shape != (self.K,):
            raise ValueError(f"vecPost debe ser ({self.K},), vino {p.shape}")
        if not np.all(np.isfinite(p)):
            raise ValueError("vecPost contiene NaN/Inf")
        if not np.all(p >= -tol):
            raise ValueError("vecPost tiene probabilidades negativas")
        if not np.isclose(p.sum(), 1.0, atol=1e-8):
            # No rompo: normalizo suave, pero te aviso en tu conciencia
            s = p.sum()
            if s <= 0:
                raise ValueError("vecPost no suma a un valor positivo")
            p = p / s

        # Etiquetas por defecto: A, B, C, ... o H1..HK si K > 26
        if labels is None:
            if self.K <= 26:
                labels = [chr(ord('A') + i) for i in range(self.K)]
            else:
                labels = [f"H{i+1}" for i in range(self.K)]
        else:
            if len(labels) != self.K:
                raise ValueError(f"len(labels)={len(labels)} != K={self.K}")

        # Detectar máximo y empates (con tolerancia)
        m = float(np.max(p))
        ties = np.flatnonzero(np.isclose(p, m, atol=tol, rtol=0.0))

        if tie == "all":
            return [labels[i] for i in ties]
        elif tie == "random":
            if rng is None:
                rng = np.random.default_rng()
            idx = int(rng.choice(ties))
            return labels[idx]
        else:  # "first"
            idx = int(ties[0])  # np.argmax(p) también sirve
            return labels[idx]
