"""
checks.py — P2: verificación de los supuestos A1 y A3 (brief §2.7).

Versión del 24 nov 2026, ajustada a lo que mostró el paciente 01543: el
archivo `*_mask.stl` es un molde positivo sólido con base plana, y viene en
otro sistema de coordenadas que el rostro.

A1, marco común rostro–molde. Por paciente se reporta:
  * qué fracción del frente del molde queda a menos de `seal_mm` del rostro
    ANTES de registrar (si es ≥ 30 %, ya compartían marco);
  * el resultado del registro rígido (register.py): residuo mediano,
    fracción del frente a menos de 1 y 2 mm del rostro, e incertidumbre
    (cuánto se mueve el molde al cambiar el recorte del ICP).
  Criterio (heurístico): pasa si, ya registrado, al menos 30 % del frente
  está a menos de 2 mm del rostro y la incertidumbre es ≤ 3 mm. Un paciente
  que no pasa no se descarta solo: se anota, porque su error en mm incluye
  el error del registro.

A3, la voxelización representa bien el sólido. Con la cuadrícula de la
configuración se compara el volumen voxelizado con el volumen del STL
(razón ≈ 1), se cuenta en cuántas piezas queda el sólido y se dibujan cortes.
Criterio: pasa si 0.8 ≤ razón ≤ 1.2. La comprobación de la cavidad de
respiración ya no aplica, porque el molde es macizo.

Las figuras se guardan en Drive (verificaciones/) y NO deben subirse a
GitHub: muestran geometría de pacientes.
"""

from __future__ import annotations

import csv
import logging

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from . import geometry as geo
from .config import Config
from .preprocess import load_pair, read_manifest

log = logging.getLogger("tesina")


def _sample_rows(cfg: Config, n: int) -> list[dict]:
    rows = read_manifest(cfg)
    if n and n < len(rows):
        # muestra determinista repartida en todos los pliegues
        rows = sorted(rows, key=lambda r: (r["pid"][::-1]))[:n]
    return rows


def check_a1(cfg: Config, n_patients: int = 20, n_figures: int = 4, verbose: bool = True) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = cfg.work / "verificaciones"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    for i, row in enumerate(_sample_rows(cfg, n_patients)):
        pid = row["pid"]
        try:
            face, mould, info = load_pair(cfg, row)
        except Exception as exc:
            log.exception(f"[A1] {pid}: no se pudo cargar o registrar")
            res.append({"pid": pid, "error": repr(exc)})
            continue
        rec = {"pid": pid, **{k: info.get(k) for k in (
            "astillas_quitadas", "mask_watertight", "frac_frente_a_3mm_sin_registro", "registrado",
            "residuo_mediana_mm", "frac_frente_a_1mm", "frac_frente_a_2mm", "incertidumbre_mm",
            "rotacion_grados", "traslacion_mm", "base_area_mm2", "base_planitud_mm")}}
        if rec["registrado"]:
            rec["pasa_A1"] = int((rec["frac_frente_a_2mm"] or 0) >= 0.3 and (rec["incertidumbre_mm"] or 99) <= 3.0)
        else:
            rec["pasa_A1"] = 1      # ya compartían marco
        res.append(rec)
        if not rec["pasa_A1"]:
            log.warning(f"[A1] {pid}: registro incierto (frente a 2 mm: {rec['frac_frente_a_2mm']}, "
                        f"incertidumbre {rec['incertidumbre_mm']} mm)")
        if i < n_figures:
            try:
                _figure_a1(plt, face, mould, out_dir / f"A1_{pid}.png", pid)
            except Exception:
                log.exception(f"[A1] {pid}: no se pudo dibujar la figura")
        if verbose:
            log.info(f"[A1] {pid}: {rec}")
    _write(out_dir / "A1_registro.csv", res)
    ok = [r for r in res if "pasa_A1" in r]
    if verbose and ok:
        n_reg = sum(1 for r in ok if r.get("registrado"))
        log.info(f"[A1] {n_reg} de {len(ok)} moldes necesitaron registro; "
                 f"pasan {sum(r['pasa_A1'] for r in ok)} de {len(ok)}")
    return res


def _figure_a1(plt, face, mould, path, pid):
    """Perfil de la línea media y vista frontal del molde registrado sobre el rostro."""
    tf = geo.fit_transform(face, 64)
    fp, _ = geo.sample_surface(face, 150000, 0)
    mp, fi = geo.sample_surface(mould, 150000, 1)
    R = np.asarray(tf.R)
    a = (fp - np.asarray(tf.pca_mean)) @ R.T
    b = (mp - np.asarray(tf.pca_mean)) @ R.T
    d, _ = cKDTree(fp).query(mp)
    fig, axs = plt.subplots(1, 2, figsize=(11, 6))
    s, sm = np.abs(a[:, 1]) < 3, np.abs(b[:, 1]) < 3
    axs[0].scatter(a[s, 2], a[s, 0], s=1, c="0.4", label="rostro")
    axs[0].scatter(b[sm, 2], b[sm, 0], s=1, c="tab:green", label="molde registrado")
    axs[0].set_xlabel("profundidad (mm)")
    axs[0].set_ylabel("altura (mm)")
    axs[0].set_title("perfil en la línea media")
    axs[0].legend(loc="lower left")
    axs[1].scatter(a[::8, 1], a[::8, 0], s=0.3, c="0.8")
    # solo las caras del molde que miran al observador (el frente); sin esto, la base
    # plana, que está atrás y lejos del rostro, se cuela en amarillo entre los puntos
    mira = (np.asarray(mould.face_normals)[fi] @ R.T)[:, 2] > 0.3
    sc = axs[1].scatter(b[mira, 1], b[mira, 0], s=0.4, c=np.minimum(d[mira], 10), cmap="viridis", vmin=0, vmax=10)
    axs[1].set_title("frente del molde: distancia al rostro (mm)")
    fig.colorbar(sc, ax=axs[1])
    for ax in axs:
        ax.set_aspect("equal")
    fig.suptitle(f"A1, paciente {pid}")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def check_a3(cfg: Config, n_patients: int = 10, n_figures: int = 4, verbose: bool = True) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = cfg.work / "verificaciones"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    for i, row in enumerate(_sample_rows(cfg, n_patients)):
        pid = row["pid"]
        try:
            face, mould, info = load_pair(cfg, row)
            tf = geo.fit_transform(face, cfg.grid, cfg.align, cfg.scale_mode, cfg.box_margin, cfg.box_mm)
            occ, oinfo = geo.solid_occupancy(geo.mesh_to_norm(mould, tf), cfg.grid)
            face_occ = geo.surface_occupancy(geo.mesh_to_norm(face, tf), cfg.grid)
        except Exception as exc:
            log.exception(f"[A3] {pid}: falló la voxelización")
            res.append({"pid": pid, "error": repr(exc)})
            continue
        vol = abs(mould.volume) if mould.is_watertight else float("nan")
        ratio = occ.sum() * tf.pitch_mm ** 3 / vol if np.isfinite(vol) and vol > 0 else float("nan")
        mp, _ = geo.sample_surface(geo.mesh_to_norm(mould, tf), 20000, 0)
        rec = {
            "pid": pid, "grid": cfg.grid, "pitch_mm": round(tf.pitch_mm, 3), "caja_mm": round(tf.scale, 1),
            "mask_watertight": int(mould.is_watertight), "volumen_ml": round(vol / 1000, 1),
            "razon_volumen": round(float(ratio), 3),
            "frac_molde_fuera_de_caja": round(float((np.abs(mp) > 0.5).any(axis=1).mean()), 4),
            "componentes_solido": int(label(occ, structure=np.ones((3, 3, 3)))[1]),
            "columnas_impares": "|".join(map(str, oinfo["odd_columns_xyz"])),
            "vox_solido": int(occ.sum()),
        }
        rec["pasa_A3"] = int(0.8 <= ratio <= 1.2) if np.isfinite(ratio) else 0
        res.append(rec)
        if rec["frac_molde_fuera_de_caja"] > 0.01:
            log.warning(f"[A3] {pid}: {rec['frac_molde_fuera_de_caja']:.1%} del molde queda fuera de la caja")
        if i < n_figures:
            com = np.round(np.argwhere(occ).mean(0)).astype(int) if occ.any() else [cfg.grid // 2] * 3
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            for ax, a in zip(axs, range(3)):
                sl = [slice(None)] * 3
                sl[a] = com[a]
                img = np.zeros(occ[tuple(sl)].shape + (3,))
                img[..., 0] = occ[tuple(sl)]
                img[..., 2] = face_occ[tuple(sl)]
                ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower")
                ax.set_title(f"corte eje {'xyz'[a]} = {com[a]}")
                ax.axis("off")
            fig.suptitle(f"A3 rojo = molde, azul = rostro, magenta = ambos ({pid}, {cfg.grid}³)")
            fig.savefig(out_dir / f"A3_{pid}_G{cfg.grid}.png", dpi=110)
            plt.close(fig)
        if verbose:
            log.info(f"[A3] {pid}: {rec}")
    _write(out_dir / f"A3_solido_G{cfg.grid}.csv", res)
    ok = [r for r in res if "pasa_A3" in r]
    if verbose and ok:
        n_ok = sum(r["pasa_A3"] for r in ok)
        log.info(f"[A3] pasan {n_ok} de {len(ok)} pacientes revisados a {cfg.grid}³")
        if n_ok < len(ok) / 2:
            log.warning("[A3] el sólido no se representa bien a esta resolución; revisa las figuras")
    return res


def _write(path, rows: list[dict]) -> None:
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
