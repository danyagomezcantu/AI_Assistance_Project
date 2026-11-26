"""
prototipo.py — P9: prototipo integrado, un escaneo facial → molde y plantilla.

Encadena en una sola llamada los componentes que el resto del código usa
por separado, y mide el tiempo de cada uno:

    1 cargar el STL del rostro
    2 orientar el rostro (regla anatómica) y voxelizarlo en G³
    3 elegir el giro de 180° que mejor concuerda con el rostro promedio
      de la cohorte (el mismo refinamiento que P3, solo con rostros)
    4 cargar la U-Net entrenada (C0) e inferir las probabilidades
    5 limpiar el volumen, marching cubes, Taubin y regreso a mm → molde STL
    6 derivar la plantilla (negativo del molde) → plantilla STL

No necesita el molde de AirTec: es el caso de uso real, un paciente nuevo
del que solo hay escaneo. Hay dos formas de usarlo:

  * `generar_para_paciente(cfg, pid)`: un paciente de la cohorte, con el
    modelo del pliegue en el que ese paciente estuvo en prueba (el modelo
    nunca lo vio). Además reporta el Dice contra el molde de AirTec.
  * `generar(cfg, "ruta/rostro.stl")`: cualquier rostro, con el promedio
    de las probabilidades de los modelos de los pliegues terminados.

Los STL resultantes son geometría de paciente: se quedan en Drive
(trabajo/prototipo/). Al paquete de resultados solo van los tiempos.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import numpy as np

from . import geometry as geo
from .config import Config
from .preprocess import FLIPS, flip_transform, paths_for, read_manifest, usable_rows
from .template import derive_template

log = logging.getLogger("tesina")

COMPONENTES = ["cargar_rostro_s", "orientar_y_voxelizar_s", "refinar_giro_s", "cargar_modelo_s",
               "inferencia_red_s", "molde_stl_s", "plantilla_stl_s"]


def tiempos_path(cfg: Config) -> Path:
    return cfg.work / "prototipo" / f"tiempos_prototipo_G{cfg.grid}.csv"


def rostro_promedio(cfg: Config) -> np.ndarray:
    """Rostro promedio de la cohorte (suavizado), ya con los giros de P3. Se guarda en Drive."""
    from scipy.ndimage import gaussian_filter

    path = cfg.vox_dir / "rostro_promedio.npy"
    if path.exists():
        return np.load(path)
    pids = [r["pid"] for r in usable_rows(cfg)]
    if not pids:
        raise RuntimeError("No hay rostros preprocesados: corre P3 primero.")
    acc = np.zeros((cfg.grid,) * 3, np.float32)
    for p in pids:
        acc += gaussian_filter(np.load(paths_for(cfg, p)["face"]).astype(np.float32), 1.0)
    ref = acc / len(pids)
    np.save(path, ref)
    return ref


def elegir_giro(face_occ: np.ndarray, ref: np.ndarray) -> str:
    from scipy.ndimage import gaussian_filter

    v0 = gaussian_filter(face_occ.astype(np.float32), 1.0)
    r = (ref - ref.mean()).ravel()
    best, best_s = "none", -np.inf
    for name, axes in FLIPS.items():
        v = (np.flip(v0, axis=axes) if axes else v0).ravel()
        s = float(np.dot(v - v.mean(), r))
        if s > best_s:
            best, best_s = name, s
    return best


def _cargar_modelos(cfg: Config, pliegues, dev):
    import torch
    from .model import UNet3D

    if pliegues is None:
        pliegues = [f for f in range(cfg.k_folds) if (cfg.run_dir / f"pliegue_{f}" / "best.pt").exists()]
    if not pliegues:
        raise RuntimeError("No hay modelos entrenados (best.pt): corre P6 primero.")
    modelos = []
    for f in pliegues:
        m = UNet3D(1, 1, cfg.base_channels, cfg.depth, cfg.dropout).to(dev)
        m.load_state_dict(torch.load(cfg.run_dir / f"pliegue_{f}" / "best.pt", map_location=dev, weights_only=True))
        m.eval()
        modelos.append(m)
    return list(pliegues), modelos


def generar(cfg: Config, rostro_stl: str | Path, salida: str | Path | None = None, pliegues=None,
            etiqueta: str | None = None, device: str | None = None, referencia: np.ndarray | None = None) -> dict:
    """Rostro STL → molde STL + plantilla STL, con el tiempo de cada componente."""
    import torch
    from .train import predict

    cfg.validate()
    etiqueta = etiqueta or Path(rostro_stl).stem.replace("_face_scan", "")
    salida = Path(salida) if salida else cfg.work / "prototipo" / etiqueta
    salida.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t = {}

    t0 = time.time()
    face = geo.load_mesh(rostro_stl)
    t["cargar_rostro_s"] = time.time() - t0

    t0 = time.time()
    tf = geo.fit_transform(face, cfg.grid, cfg.align, cfg.scale_mode, cfg.box_margin, cfg.box_mm,
                           seed=cfg.sample_seed)
    face_occ = geo.surface_occupancy(geo.mesh_to_norm(face, tf), cfg.grid,
                                     cfg.face_samples_per_voxel_area, cfg.sample_seed)
    t["orientar_y_voxelizar_s"] = time.time() - t0

    t0 = time.time()
    giro = elegir_giro(face_occ, rostro_promedio(cfg))
    if FLIPS[giro]:
        face_occ = np.ascontiguousarray(np.flip(face_occ, axis=FLIPS[giro]))
        tf = flip_transform(tf, giro)
    t["refinar_giro_s"] = time.time() - t0

    t0 = time.time()
    usados, modelos = _cargar_modelos(cfg, pliegues, dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t["cargar_modelo_s"] = time.time() - t0

    t0 = time.time()
    X = face_occ[None].astype(np.uint8)
    prob = np.mean([predict(m, X, dev, cfg.amp)[0] for m in modelos], axis=0).astype(np.float32)
    t["inferencia_red_s"] = time.time() - t0

    t0 = time.time()
    prob = geo.clean_volume(prob, cfg.threshold, cfg.largest_component)
    molde = geo.volume_to_mesh_mm(prob, tf, cfg.threshold, cfg.smoothing, cfg.smooth_iterations)
    if molde is not None:
        molde.export(salida / "molde_predicho.stl")
    t["molde_stl_s"] = time.time() - t0

    t0 = time.time()
    binv = prob >= cfg.threshold
    plantilla, pinfo = derive_template(binv, face_occ, tf, cfg.template_margin_mm)
    if plantilla is not None:
        plantilla.export(salida / "plantilla.stl")
    t["plantilla_stl_s"] = time.time() - t0

    rec = {
        "etiqueta": etiqueta, "grid": cfg.grid, "dispositivo": str(dev),
        "modelo": f"pliegue {usados[0]}" if len(usados) == 1 else f"promedio de {len(usados)} pliegues",
        "giro": giro, "pitch_mm": round(tf.pitch_mm, 3),
        **{k: round(v, 3) for k, v in t.items()},
        "total_s": round(sum(t.values()), 3),
        "total_sin_cargar_modelo_s": round(sum(v for k, v in t.items() if k != "cargar_modelo_s"), 3),
        "molde_vacio": int(molde is None),
        "molde_watertight": int(molde.is_watertight) if molde is not None else 0,
        "volumen_molde_ml": pinfo.get("volumen_molde_ml", 0),
        "frac_columnas_contrasalida": pinfo.get("frac_columnas_contrasalida", ""),
        "plantilla_watertight": pinfo.get("plantilla_watertight", 0),
    }
    if referencia is not None:
        from .metrics import dice_iou
        rec["dice_vs_airtec"] = round(dice_iou(binv, referencia)[0], 4)
    (salida / "resumen.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    _anotar(tiempos_path(cfg), rec)
    if molde is None:
        log.warning(f"[P9] {etiqueta}: la red predijo un volumen vacío; no hay molde ni plantilla")
    log.info(f"[P9] {etiqueta}: {rec['modelo']}, {rec['total_s']:.1f} s en total "
             f"({', '.join(f'{k[:-2]} {v:.2f}' for k, v in t.items())}); archivos en {salida}")
    return {**rec, "carpeta": str(salida)}


def generar_para_paciente(cfg: Config, pid: str | None = None, device: str | None = None) -> dict:
    """Paciente de la cohorte con el modelo del pliegue que NO lo vio en entrenamiento."""
    rows = {r["pid"]: r for r in read_manifest(cfg)}
    terminados = {f for f in range(cfg.k_folds) if (cfg.run_dir / f"pliegue_{f}" / "best.pt").exists()}
    if pid is None:
        cands = [p for p, r in sorted(rows.items()) if r["fold"] in terminados and paths_for(cfg, p)["mask"].exists()]
        if not cands:
            raise RuntimeError("No hay pacientes con modelo de su pliegue terminado: corre P6 primero.")
        pid = cands[0]
    row = rows[pid]
    if row["fold"] not in terminados:
        raise RuntimeError(f"El pliegue {row['fold']} del paciente {pid} no ha terminado de entrenar.")
    ref_path = paths_for(cfg, pid)["mask"]
    ref = np.load(ref_path) if ref_path.exists() else None
    return generar(cfg, Path(cfg.data_root) / row["face_path"], pliegues=[row["fold"]], etiqueta=pid,
                   device=device, referencia=ref)


def _anotar(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    rows.append({k: rec[k] for k in rec})
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
