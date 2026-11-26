"""
stats.py — P7: comparación pareada C0 contra B1 y B2 (brief §2.5).

Se juntan los cinco pliegues de prueba, de modo que cada paciente aporta un
error por método. Para cada métrica y cada comparación:
  * diferencia por paciente d = C0 − B (en distancias, negativa = C0 mejor;
    en Dice/IoU, positiva = C0 mejor);
  * prueba de rangos con signo de Wilcoxon, bilateral;
  * tamaño de efecto r = Z / √n, con Z de la aproximación normal;
  * intervalo de confianza al 95 % de la mediana de d por bootstrap;
  * corrección de Holm sobre todas las pruebas de la tabla, α = 0.05.
Solo entran pacientes con valor finito en ambos métodos; el número de
pacientes excluidos (predicción vacía) se reporta aparte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PRIMARY_METRICS = [
    "dice", "iou",
    "chamfer_total", "chamfer_frente", "chamfer_base",
    "hd95_total", "hd95_frente", "hd95_base",
]
HIGHER_IS_BETTER = {"dice", "iou"}


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(np.where(ok, p, np.inf))
    m = ok.sum()
    running = 0.0
    for rank, i in enumerate(idx[:m]):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def bootstrap_median_ci(d: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(d, size=(n_boot, len(d)), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def compare(df: pd.DataFrame, model: str, baselines=("B1", "B2"), metrics=PRIMARY_METRICS,
            alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    wide = {m: df[df.method == m].set_index("pid") for m in [model, *baselines]}
    for b in baselines:
        for met in metrics:
            if met not in df.columns:
                continue
            a = pd.to_numeric(wide[model][met], errors="coerce")
            c = pd.to_numeric(wide[b][met], errors="coerce")
            j = pd.concat([a, c], axis=1, keys=["m", "b"]).dropna()
            n_excl = len(set(a.index) | set(c.index)) - len(j)
            d = (j["m"] - j["b"]).to_numpy()
            rec = {"comparacion": f"{model} vs {b}", "metrica": met, "n": len(d), "n_excluidos": n_excl,
                   "mediana_modelo": float(j["m"].median()) if len(j) else np.nan,
                   "mediana_referencia": float(j["b"].median()) if len(j) else np.nan,
                   "mediana_diferencia": float(np.median(d)) if len(d) else np.nan}
            if len(d) >= 6 and np.any(d != 0):
                w = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox", method="approx")
                z = stats.norm.isf(w.pvalue / 2) * np.sign(np.median(d) if np.median(d) != 0 else d.mean())
                lo, hi = bootstrap_median_ci(d)
                rec.update({"W": float(w.statistic), "p": float(w.pvalue), "Z": float(z),
                            "r": float(z / np.sqrt(len(d))), "IC95_mediana_inf": lo, "IC95_mediana_sup": hi})
            else:
                rec.update({"W": np.nan, "p": np.nan, "Z": np.nan, "r": np.nan,
                            "IC95_mediana_inf": np.nan, "IC95_mediana_sup": np.nan})
            better = (rec["mediana_diferencia"] > 0) if met in HIGHER_IS_BETTER else (rec["mediana_diferencia"] < 0)
            rec["favorece"] = model if better else b
            rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].tolist())
        out["significativo_holm"] = out["p_holm"] < alpha
    return out


def per_fold_summary(df: pd.DataFrame, metrics=PRIMARY_METRICS) -> pd.DataFrame:
    """Media ± DE entre pliegues de la media por pliegue, y mediana [RIC] agrupada."""
    rows = []
    for m, g in df.groupby("method"):
        for met in metrics:
            if met not in g.columns:
                continue
            v = pd.to_numeric(g[met], errors="coerce")
            fold_means = v.groupby(g["fold"]).mean()
            rows.append({
                "metodo": m, "metrica": met, "n": int(v.notna().sum()),
                "media_pliegues": float(fold_means.mean()), "de_pliegues": float(fold_means.std(ddof=1)) if len(fold_means) > 1 else np.nan,
                "mediana": float(v.median()), "q1": float(v.quantile(0.25)), "q3": float(v.quantile(0.75)),
            })
    return pd.DataFrame(rows)
