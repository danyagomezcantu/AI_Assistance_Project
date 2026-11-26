"""
preprocess.py — P3: STL → cuadrícula fija compartida G³ (+ registro de transformación).

Por cada paciente del manifiesto escribe en `cfg.vox_dir`:
    {pid}_face.npy   uint8 (G,G,G)  cascarón de la superficie del rostro
    {pid}_mask.npy   uint8 (G,G,G)  material de la pared de la mascarilla
    {pid}_tf.json    registro de transformación (geometry.Transform)
y al final `preprocess_report.csv` con una fila por paciente (arista del
vóxel, fracción de la mascarilla fuera de la caja, razón de volumen, etc.).

El paso es reanudable: si Colab se desconecta, al volver a correrlo salta
los pacientes que ya tienen sus tres archivos.

Antes de voxelizar, el molde (`*_mask.stl`) se limpia (cuerpo principal) y
se lleva al sistema de coordenadas del rostro con register.py; el registro
se guarda en `trabajo/registros/{pid}_reg.json` y lo reutilizan las
verificaciones, la evaluación y las demás resoluciones.

Refinamiento de signos (sign_refine)
------------------------------------
La regla de orientación de geometry.face_frame puede equivocarse en algún
rostro raro. Después de procesar a todos, se compara el rostro de cada
paciente con el rostro promedio de la cohorte bajo las cuatro rotaciones de
180° que conservan los ejes (identidad y giros sobre x, y, z) y se elige la
que más se parece. Solo usa rostros (entradas), nunca mascarillas, así que
no filtra información de las etiquetas. El giro elegido se guarda en el
registro de transformación.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import logging

import numpy as np

from . import geometry as geo
from . import register as reg
from .config import Config

log = logging.getLogger("tesina")

FLIPS = {"none": (), "rot_x": (1, 2), "rot_y": (0, 2), "rot_z": (0, 1)}


def read_manifest(cfg: Config) -> list[dict]:
    path = cfg.index_dir / "manifest.csv"
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["fold"] = int(r["fold"])
    return rows


def paths_for(cfg: Config, pid: str) -> dict:
    d = cfg.vox_dir
    return {"face": d / f"{pid}_face.npy", "mask": d / f"{pid}_mask.npy", "tf": d / f"{pid}_tf.json"}


def load_pair(cfg: Config, row: dict, register: bool = True):
    """Carga rostro y molde; el molde queda limpio y en el marco del rostro.

    Devuelve (rostro, molde, info). El registro se calcula una vez por
    paciente y se reutiliza desde `cfg.reg_dir`.
    """
    root = Path(cfg.data_root)
    pid = row["pid"]
    face = geo.load_mesh(root / row["face_path"])
    mask = geo.load_mesh(root / row["mask_path"])
    mask, n_debris = reg.largest_component(mask)
    info = {"astillas_quitadas": n_debris, "mask_watertight": int(mask.is_watertight)}
    if n_debris:
        log.debug(f"{pid}: se quitaron {n_debris} componentes sueltos del molde")
    if not mask.is_watertight:
        log.warning(f"{pid}: el molde no es una malla cerrada; la ocupación usa voto entre 3 ejes")
    if not face.faces.size:
        raise ValueError(f"{pid}: rostro vacío")
    if register and cfg.register_mask != "none":
        cfg.reg_dir.mkdir(parents=True, exist_ok=True)
        rpath = cfg.reg_dir / f"{pid}_reg.json"
        if rpath.exists():
            rec = json.loads(rpath.read_text(encoding="utf-8"))
        else:
            mode = "always" if cfg.register_mask == "always" else "auto"
            T, rinfo = reg.register_mould_to_face(face, mask, mode=mode, seed=cfg.sample_seed)
            rec = {"T": T.tolist(), **rinfo}
            rpath.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            if rinfo.get("registrado"):
                msg = (f"{pid}: registro molde→rostro: residuo mediano {rinfo['residuo_mediana_mm']} mm, "
                       f"incertidumbre {rinfo['incertidumbre_mm']} mm")
                (log.warning if rinfo["incertidumbre_mm"] > 3 else log.info)(msg)
        mask.apply_transform(np.asarray(rec["T"]))
        info.update({k: v for k, v in rec.items() if k != "T"})
    return face, mask, info


def preprocess_one(cfg: Config, row: dict) -> dict:
    t0 = time.time()
    face, mask, pinfo = load_pair(cfg, row)
    tf = geo.fit_transform(face, cfg.grid, cfg.align, cfg.scale_mode, cfg.box_margin,
                           cfg.box_mm, seed=cfg.sample_seed)
    face_n = geo.mesh_to_norm(face, tf)
    mask_n = geo.mesh_to_norm(mask, tf)
    G = cfg.grid
    face_occ = geo.surface_occupancy(face_n, G, cfg.face_samples_per_voxel_area, cfg.sample_seed)
    if cfg.mask_occupancy == "solid":
        mask_occ, info = geo.solid_occupancy(mask_n, G)
    else:
        mask_occ = geo.surface_occupancy(mask_n, G, cfg.face_samples_per_voxel_area, cfg.sample_seed)
        info = {"odd_columns_xyz": [0, 0, 0], "axis_disagreement": 0.0}

    # fracción de la superficie de la mascarilla que queda fuera de la caja
    pts, _ = geo.sample_surface(mask_n, 20000, cfg.sample_seed)
    clip = float((np.abs(pts) > 0.5).any(axis=1).mean())
    vol_ratio = (float(mask_occ.sum() * tf.pitch_mm ** 3 / abs(mask.volume))
                 if mask.is_watertight and abs(mask.volume) > 0 else float("nan"))

    p = paths_for(cfg, row["pid"])
    p["face"].parent.mkdir(parents=True, exist_ok=True)
    np.save(p["face"], face_occ)
    np.save(p["mask"], mask_occ)
    tf.to_json(p["tf"])
    ext = (tf.to_norm(face.vertices).max(0) - tf.to_norm(face.vertices).min(0)) * tf.scale
    return {
        "pid": row["pid"], "fold": row["fold"], "pitch_mm": round(tf.pitch_mm, 4),
        "box_mm": round(tf.scale, 2),
        "face_extent_mm": "|".join(f"{e:.1f}" for e in ext),
        "face_vox": int(face_occ.sum()), "mask_vox": int(mask_occ.sum()),
        "mask_watertight": int(mask.is_watertight),
        "mask_volume_ratio": round(vol_ratio, 4),
        "mask_outside_box_frac": round(clip, 4),
        "odd_columns": "|".join(map(str, info["odd_columns_xyz"])),
        "axis_disagreement": round(info["axis_disagreement"], 4),
        "regla_profundidad": tf.frame_info.get("regla_profundidad", ""),
        "margen_altura_mm": tf.frame_info.get("margen_altura_mm", ""),
        **{k: pinfo.get(k, "") for k in ("astillas_quitadas", "registrado", "residuo_mediana_mm",
                                         "frac_frente_a_2mm", "incertidumbre_mm")},
        "n_face_tris": len(face.faces), "n_mask_tris": len(mask.faces),
        "seconds": round(time.time() - t0, 2),
        "sign_flip": "none",
    }


def refine_signs(cfg: Config, pids: list[str], iterations: int = 2) -> dict[str, str]:
    """Elige por paciente el giro de 180° que mejor concuerda con el rostro promedio."""
    from scipy.ndimage import gaussian_filter

    faces = {pid: gaussian_filter(np.load(paths_for(cfg, pid)["face"]).astype(np.float32), 1.0) for pid in pids}
    chosen = {pid: "none" for pid in pids}

    def apply(v, flip):
        axes = FLIPS[flip]
        return np.flip(v, axis=axes) if axes else v

    for _ in range(iterations):
        ref = np.mean([apply(faces[p], chosen[p]) for p in pids], axis=0)
        ref = (ref - ref.mean()).ravel()
        for p in pids:
            best, best_s = chosen[p], -np.inf
            for name in FLIPS:
                v = apply(faces[p], name).ravel()
                s = float(np.dot(v - v.mean(), ref))
                if s > best_s:
                    best, best_s = name, s
            chosen[p] = best
    return chosen


def flip_transform(tf: geo.Transform, flip: str) -> geo.Transform:
    """Transformación equivalente a girar 180° la cuadrícula (niega fila de R y centro)."""
    R = np.asarray(tf.R, dtype=float).copy()
    c = np.asarray(tf.center, dtype=float).copy()
    for a in FLIPS[flip]:
        R[a] = -R[a]
        c[a] = -c[a]
    tf.R, tf.center = R.tolist(), c.tolist()
    return tf


def apply_flip(cfg: Config, pid: str, flip: str) -> None:
    """Gira en 180° los arreglos y actualiza R y center para que sigan siendo exactos."""
    axes = FLIPS[flip]
    if not axes:
        return
    p = paths_for(cfg, pid)
    for k in ("face", "mask"):
        np.save(p[k], np.ascontiguousarray(np.flip(np.load(p[k]), axis=axes)))
    tf = flip_transform(geo.Transform.from_json(p["tf"]), flip)
    tf.to_json(p["tf"])


def run_preprocess(cfg: Config, sign_refine: bool = True, limit: int | None = None, verbose: bool = True,
                   procesos: int = 0, limite_gb: float = 0.0, timeout_s: int = 900) -> list[dict]:
    """P3 para todo el manifiesto. Cada paciente corre en un proceso hijo aparte.

    Así, un paciente que pide demasiada memoria (por ejemplo, un molde abierto
    muy pesado) se detiene solo él y queda con error «sin memoria», en vez de
    tumbar la sesión de Colab. Con `procesos` > 1 se procesan varios pacientes
    a la vez (0 = automático según núcleos y RAM). `limite_gb` es el tope de
    memoria por paciente (0 = automático).
    """
    import warnings
    cfg.validate()
    rows = read_manifest(cfg)
    if limit:
        rows = rows[:limit]
    cfg.vox_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.vox_dir / "preprocess_report.csv"
    done = {}
    if report_path.exists():
        with report_path.open(encoding="utf-8") as fh:
            done = {r["pid"]: r for r in csv.DictReader(fh)}

    resultados: dict[str, dict] = {}
    pendientes = []
    for row in rows:
        pid = row["pid"]
        previo = done.get(pid)
        if previo and all(v.exists() for v in paths_for(cfg, pid).values()):
            resultados[pid] = previo
        elif previo and "sin memoria" in (previo.get("error") or ""):
            resultados[pid] = previo          # no se reintenta: volvería a reventar
        else:
            pendientes.append(row)

    total_gb = _ram_total_gb()
    n_proc, tope = _plan_procesos(procesos, limite_gb, total_gb, len(pendientes))
    t_all = time.time()
    if pendientes:
        log.info(f"[P3] {len(resultados)} ya hechos, {len(pendientes)} por hacer; {n_proc} proceso(s) a la vez, "
                 f"tope {tope:.1f} GB por paciente (RAM total {total_gb:.0f} GB)")
    hechos = [0]

    def al_terminar(row, rec):
        resultados[row["pid"]] = rec
        _write_report(report_path, [resultados[r["pid"]] for r in rows if r["pid"] in resultados])
        hechos[0] += 1
        if verbose:
            msg = rec.get("error") or f"h={rec['pitch_mm']} mm, vóxeles mascarilla={rec['mask_vox']}, {rec['seconds']} s"
            k = hechos[0]
            eta = (time.time() - t_all) / k * (len(pendientes) - k)
            (log.warning if rec.get("error") else log.info)(
                f"[P3] {len(resultados)}/{len(rows)} {row['pid']}: {msg}"
                + (f" | faltan ~{eta / 60:.0f} min" if k % 10 == 0 else ""))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*fork.*", category=DeprecationWarning)
        _correr_en_hijos(cfg, pendientes, n_proc, tope, timeout_s, al_terminar)

    out = [resultados[r["pid"]] for r in rows if r["pid"] in resultados]
    ok = [r for r in out if not r.get("error")]
    if sign_refine and cfg.align == "pca" and len(ok) > 2:
        flag = cfg.vox_dir / "signs_refined.json"
        if not flag.exists():
            chosen = refine_signs(cfg, [r["pid"] for r in ok])
            for r in ok:
                apply_flip(cfg, r["pid"], chosen[r["pid"]])
                r["sign_flip"] = chosen[r["pid"]]
            flag.write_text(json.dumps(chosen, indent=1), encoding="utf-8")
            _write_report(report_path, out)
            n_flip = sum(v != "none" for v in chosen.values())
            log.info(f"[P3] refinamiento de signos: {n_flip} de {len(chosen)} pacientes girados 180°")
    n_err = len(out) - len(ok)
    (log.warning if n_err else log.info)(
        f"[P3] listo: {len(ok)} correctos, {n_err} con error, {time.time() - t_all:.0f} s"
        + (" (las trazas están en la bitácora)" if n_err else ""))
    return out


def _ram_total_gb() -> float:
    try:
        for linea in open("/proc/meminfo"):
            if linea.startswith("MemTotal"):
                return int(linea.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return 8.0


def _rss_gb(pid: int) -> float | None:
    """Memoria residente de un proceso y sus hijos directos, en GB (None si ya no existe)."""
    try:
        for linea in open(f"/proc/{pid}/status"):
            if linea.startswith("VmRSS"):
                return int(linea.split()[1]) / 1024 ** 2
    except OSError:
        return None
    return 0.0


def _plan_procesos(procesos: int, limite_gb: float, total_gb: float, n: int) -> tuple[int, float]:
    """Cuántos pacientes a la vez y con qué tope; deja ~20 % de la RAM para el cuaderno."""
    import os
    usable = 0.8 * total_gb
    if procesos <= 0:
        # cada paciente usa ~1 GB y casi nunca más de 3; se reservan 6 GB por proceso
        procesos = max(1, min(os.cpu_count() or 1, int(usable // 6), 8))
    procesos = max(1, min(procesos, max(n, 1)))
    tope = limite_gb if limite_gb > 0 else usable / procesos
    return procesos, tope


def _hijo(cfg: Config, row: dict, conn) -> None:
    try:
        try:   # si alguien tiene que morir por falta de memoria, que sea este proceso
            Path("/proc/self/oom_score_adj").write_text("1000")
        except OSError:
            pass
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(1)
        except Exception:
            pass
        conn.send(("ok", preprocess_one(cfg, row), ""))
    except BaseException as exc:
        import traceback
        try:
            conn.send(("error", repr(exc), traceback.format_exc()))
        except Exception:
            pass
    finally:
        conn.close()


def _correr_en_hijos(cfg: Config, pendientes: list[dict], n_proc: int, tope_gb: float,
                     timeout_s: int, al_terminar) -> None:
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    cola = list(pendientes)
    activos = {}
    try:
        while cola or activos:
            while cola and len(activos) < n_proc:
                row = cola.pop(0)
                lee, escribe = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=_hijo, args=(cfg, row, escribe), daemon=True)
                proc.start()
                escribe.close()
                activos[row["pid"]] = (proc, lee, row, time.time())
            time.sleep(0.1)
            for pid, (proc, lee, row, t0) in list(activos.items()):
                res = None
                if lee.poll():
                    try:
                        res = lee.recv()
                    except (EOFError, OSError):
                        res = None
                if res is None:
                    rss = _rss_gb(proc.pid)
                    if rss is not None and rss > tope_gb:
                        proc.kill()
                        res = ("error", f"sin memoria: pasó de {tope_gb:.1f} GB y se detuvo solo este paciente", "")
                    elif time.time() - t0 > timeout_s:
                        proc.kill()
                        res = ("error", f"tardó más de {timeout_s} s y se detuvo", "")
                    elif not proc.is_alive():
                        if lee.poll():
                            try:
                                res = lee.recv()
                            except (EOFError, OSError):
                                res = None
                        if res is None:
                            res = ("error", f"sin memoria: el proceso del paciente terminó con código {proc.exitcode}", "")
                if res is None:
                    continue
                proc.join(timeout=10)
                lee.close()
                del activos[pid]
                estado, valor, traza = res
                if estado == "ok":
                    rec = valor
                else:
                    log.error(f"[P3] {pid}: falló el preprocesamiento: {valor}" + (f"\n{traza}" if traza else ""))
                    rec = {"pid": pid, "fold": row["fold"], "error": valor}
                al_terminar(row, rec)
    finally:
        for proc, lee, _, _ in activos.values():
            if proc.is_alive():
                proc.kill()


def _write_report(path: Path, rows: list[dict]) -> None:
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_arrays(cfg: Config, pids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([np.load(paths_for(cfg, p)["face"]) for p in pids])
    Y = np.stack([np.load(paths_for(cfg, p)["mask"]) for p in pids])
    return X, Y


def excluidos_path(cfg: Config) -> Path:
    return cfg.index_dir / "excluidos.csv"


def excluidos(cfg: Config) -> dict[str, str]:
    """pid → motivo de los pacientes que la regla de P3b dejó fuera (vacío si no hay regla)."""
    path = excluidos_path(cfg)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {r["pid"]: r["motivo"] for r in csv.DictReader(fh)}


def usable_rows(cfg: Config, con_excluidos: bool = False) -> list[dict]:
    """Filas del manifiesto con arreglos preprocesados y fuera de la lista de excluidos.

    Los pliegues no cambian: un paciente excluido simplemente no se usa ni para
    entrenar ni para evaluar en el pliegue que ya tenía.
    """
    fuera = {} if con_excluidos else excluidos(cfg)
    return [r for r in read_manifest(cfg) if r["pid"] not in fuera
            and all(v.exists() for v in paths_for(cfg, r["pid"]).values())]
