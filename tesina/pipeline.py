"""
pipeline.py — un paso del cuaderno = una función (numeración P1–P8 del calendario).

    P1 paso_indice            indexador: autoprueba + manifiesto con pliegues
    P2 paso_verificaciones    supuestos A1 (marco común / registro) y A3 (voxelización del sólido)
    P3 paso_preprocesamiento  STL → cuadrícula fija G³ + transformaciones
    P3b paso_exclusiones      regla de exclusión fijada antes de ver resultados
    P4 paso_referencias       B1/B2 y su evaluación por la ruta común
    P5 paso_prueba_humo       C0 a 16³, pocas épocas, extremo a extremo
    P6 paso_entrenamiento     C0 por pliegue (reanudable)
    P7 paso_evaluacion_c0     evaluación de C0 + Wilcoxon/Holm contra B1/B2
    P7b paso_suavizado        mismas predicciones de C0 sin suavizar y con Laplace (sin reentrenar)
    P8 paso_paquete           plantillas, casos ilustrativos y ZIP de resultados
    P9 paso_prototipo         prototipo integrado: un rostro STL → molde y plantilla STL, con tiempos

Todas las funciones son idempotentes: si se vuelven a correr, saltan lo que
ya está hecho y continúan donde se quedaron. Cada una se envuelve en
`bitacora.paso`, que anota inicio, fin y duración, y si algo falla deja la
traza completa en trabajo/bitacora/ junto con una pista de qué revisar.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from pathlib import Path

from . import baselines, checks, evaluate, preprocess, prototipo, report, train
from . import index_dataset as idx
from .bitacora import paso
from .config import Config

log = logging.getLogger("tesina")


# --------------------------------------------------------------------------
# P0.3: datos al disco local de Colab (ZIP o carpeta en Drive)
# --------------------------------------------------------------------------

_NECESARIOS = ("face_scan", "mask")      # lo único que leen P1–P9


def preparar_datos(datos: str, destino: str, solo_necesarios: bool = True,
                   buscar_en: str = "/content/drive/MyDrive") -> Path:
    """Deja los datos en el disco local de la VM y devuelve la raíz para el indexador.

    `datos` puede ser un ZIP o una carpeta de Drive (por ejemplo «1. Training»).
    Si la ruta no existe, se busca en `buscar_en` una carpeta o ZIP cuyo nombre
    contenga «training» o «repository»; si hay exactamente uno, se usa.

    De una carpeta se copian solo los archivos que el código lee
    (`<ID>_face_scan.stl` y `<ID>_mask.stl`, unos 50 MB por paciente), con la
    misma estructura de subcarpetas. Leer directo de Drive es lento y se
    cae a media corrida; copiar una vez por sesión es más seguro. Si la sesión
    no se reinició, una segunda llamada no copia nada.
    """
    destino = Path(destino)
    marca = destino / ".datos_ok"
    if marca.exists():
        raiz = Path(marca.read_text(encoding="utf-8").strip())
        log.info(f"[P0.3] datos ya preparados en esta sesión: {raiz}")
        return raiz
    with paso("P0.3 datos al disco local",
              "revisa que DATOS apunte a tu ZIP o a la carpeta «1. Training» en Drive; "
              "si el ZIP está dañado, vuelve a subirlo o sube la carpeta sin comprimir"):
        src = Path(datos)
        if not src.exists():
            src = _buscar_datos(Path(buscar_en), src)
        destino.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            import zipfile
            if not zipfile.is_zipfile(src):
                raise ValueError(f"{src} no es un ZIP ni una carpeta")
            raiz, danados = _extraer_zip(src, destino, solo_necesarios)
        else:
            raiz, danados = _copiar_carpeta(src, destino / src.name, solo_necesarios), []
        if danados:
            # Sin marca: si vuelves a correr P0.3 reintenta solo los dañados
            # (los buenos ya están en disco y se saltan).
            log.warning(f"[P0.3] se siguió SIN {len(danados)} archivo(s) dañado(s); "
                        f"esos pacientes saldrán como incompletos en P1")
        else:
            marca.write_text(str(raiz), encoding="utf-8")
        log.info(f"[P0.3] raíz de los datos: {raiz}")
        return raiz


# Si más de esta fracción de los archivos necesarios sale dañada, el ZIP entero
# está mal (subida incompleta, ZIP mal creado) y no tiene caso seguir.
_MAX_DANADOS = 0.05


def _extraer_zip(src: Path, destino: Path, solo_necesarios: bool) -> tuple[Path, list[str]]:
    """Extrae del ZIP solo rostro y molde, archivo por archivo, y sigue si alguno está dañado.

    - Salta plantillas, hojas de datos, `_additional` y basura de macOS (__MACOSX, ._*).
    - Cada archivo se escribe primero como `.parcial` y se renombra al terminar,
      así que nunca queda un STL a medias; si ya existe con el tamaño correcto
      (corrida anterior en la misma sesión), se salta.
    - Si un archivo no se puede descomprimir, se reintenta una vez (falla
      pasajera al leer Drive) y luego con `unzip` del sistema; si sigue mal,
      se anota en `archivos_danados.txt` y se continúa con los demás.
    Devuelve (raíz para el indexador, lista de archivos dañados).
    """
    import errno
    import time
    import zipfile
    t0 = time.time()
    with zipfile.ZipFile(src) as zf:
        miembros = []
        for info in zf.infolist():
            nombre = info.filename.replace("\\", "/")
            base = nombre.rsplit("/", 1)[-1]
            if info.is_dir() or nombre.startswith("__MACOSX/") or base.startswith("._"):
                continue
            m = idx.FILE_RE.match(base)
            if solo_necesarios and not (m and m.group("kind").lower() in _NECESARIOS):
                continue
            partes = [x for x in nombre.split("/") if x not in ("", ".", "..")]
            miembros.append((info, Path(*partes)))
        if not miembros:
            raise FileNotFoundError(f"{src} no tiene archivos <ID>_face_scan.stl / <ID>_mask.stl")
        total = sum(i.file_size for i, _ in miembros)
        libre = shutil.disk_usage(destino).free
        log.info(f"[P0.3] {len(miembros)} archivos de rostro y molde ({total / 1e9:.1f} GB) de "
                 f"{len(zf.infolist())} en el ZIP; espacio libre en disco {libre / 1e9:.0f} GB")
        if total > libre * 0.95:
            raise OSError(errno.ENOSPC, f"No caben {total / 1e9:.1f} GB en el disco de Colab "
                                        f"(libres {libre / 1e9:.0f} GB)")
        danados, hecho, ultimo = [], 0, time.time()
        for k, (info, rel) in enumerate(miembros, 1):
            q = destino / rel
            if not (q.exists() and q.stat().st_size == info.file_size):
                q.parent.mkdir(parents=True, exist_ok=True)
                error = _extraer_uno(zf, src, info, q)
                if error:
                    danados.append(info.filename)
                    log.warning(f"[P0.3] DAÑADO, se salta: {info.filename} ({error})")
            hecho += info.file_size
            if k % 50 == 0 or k == len(miembros) or time.time() - ultimo > 60:
                ultimo = time.time()
                dt = max(ultimo - t0, 1e-6)
                faltan = (total - hecho) / max(hecho / dt, 1.0)
                log.info(f"[P0.3] {k}/{len(miembros)} archivos, {hecho / 1e9:.1f} de "
                         f"{total / 1e9:.1f} GB, {hecho / 1e6 / dt:.0f} MB/s, faltan ~{faltan / 60:.0f} min")
            if len(danados) > max(5, _MAX_DANADOS * len(miembros)):
                _anotar_danados(destino, danados, sigue=False)
                raise RuntimeError(
                    f"Ya van {len(danados)} archivos dañados en {k} leídos: el ZIP de Drive está mal "
                    "(subida incompleta o ZIP mal creado). Vuelve a subirlo, o sube la carpeta "
                    "«1. Training» sin comprimir y pon esa carpeta en DATOS.")
    _anotar_danados(destino, danados)
    entradas = [p for p in destino.iterdir() if not p.name.startswith((".", "__MACOSX"))
                and p.name != "archivos_danados.txt"]
    dirs = [p for p in entradas if p.is_dir()]
    raiz = dirs[0] if len(dirs) == 1 and len(entradas) == 1 else destino
    return raiz, danados


def _extraer_uno(zf, src: Path, info, q: Path) -> str:
    """Extrae un miembro a `q`; devuelve "" si salió bien o el error si no."""
    import errno
    import subprocess
    import zlib
    import zipfile
    parcial = q.with_name(q.name + ".parcial")
    error = ""
    for intento in (1, 2):
        try:
            with zf.open(info) as fuente, open(parcial, "wb") as destino:
                shutil.copyfileobj(fuente, destino, 1 << 22)   # lee y comprueba el CRC
            parcial.replace(q)
            return ""
        except OSError as exc:
            parcial.unlink(missing_ok=True)
            if exc.errno == errno.ENOSPC:
                raise
            error = f"{type(exc).__name__}: {exc}"
        except (zlib.error, zipfile.BadZipFile, EOFError, ValueError) as exc:
            parcial.unlink(missing_ok=True)
            error = f"{type(exc).__name__}: {exc}"
        log.debug(f"[P0.3] intento {intento} falló en {info.filename}: {error}")
    if shutil.which("unzip"):
        nombre = "".join("\\" + c if c in "[]*?" else c for c in info.filename)
        try:
            with open(parcial, "wb") as destino:
                r = subprocess.run(["unzip", "-p", str(src), nombre], stdout=destino,
                                   stderr=subprocess.PIPE, timeout=1800)
            if r.returncode == 0 and parcial.stat().st_size == info.file_size:
                parcial.replace(q)
                log.info(f"[P0.3] {info.filename}: recuperado con unzip")
                return ""
            error += f" | unzip: código {r.returncode} {r.stderr.decode(errors='replace')[-200:].strip()}"
        except Exception as exc:
            error += f" | unzip: {exc}"
        parcial.unlink(missing_ok=True)
    return error


def _anotar_danados(destino: Path, danados: list[str], sigue: bool = True) -> None:
    ruta = destino / "archivos_danados.txt"
    if not danados:
        ruta.unlink(missing_ok=True)
        return
    ruta.write_text("\n".join(danados) + "\n", encoding="utf-8")
    ids = sorted({m.group("pid") for d in danados
                  if (m := idx.FILE_RE.match(d.replace("\\", "/").rsplit("/", 1)[-1]))})
    log.warning(f"[P0.3] {len(danados)} archivo(s) dañado(s) en el ZIP, pacientes: {', '.join(ids)}. "
                f"Lista en {ruta}." + (" Son pocos: se sigue y P1 los marca incompletos." if sigue else ""))


def pacientes_danados(destino: str) -> list[str]:
    """IDs de pacientes con rostro o molde dañado en el ZIP (según la última corrida de P0.3)."""
    ruta = Path(destino) / "archivos_danados.txt"
    if not ruta.exists():
        return []
    ids = set()
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        m = idx.FILE_RE.match(linea.replace("\\", "/").rsplit("/", 1)[-1])
        if m:
            ids.add(m.group("pid"))
    return sorted(ids)


def _buscar_datos(base: Path, pedido: Path) -> Path:
    cands = []
    if base.exists():
        for p in list(base.glob("*")) + list(base.glob("*/*")):
            n = p.name.lower()
            if ("training" in n or "repository" in n) and (p.is_dir() or n.endswith(".zip")):
                cands.append(p)
    if len(cands) == 1:
        log.warning(f"[P0.3] no existe {pedido}; uso el único candidato que encontré: {cands[0]}")
        return cands[0]
    lista = "\n".join(f"    {c}" for c in cands) or "    (ninguno)"
    raise FileNotFoundError(f"No existe {pedido}. Candidatos en {base}:\n{lista}\n"
                            "Pon en DATOS la ruta correcta (carpeta o ZIP).")


def _copiar_carpeta(src: Path, dst: Path, solo_necesarios: bool) -> Path:
    import os
    import time
    todos = []
    for d, _, files in os.walk(src):
        for f in files:
            m = idx.FILE_RE.match(f)
            if solo_necesarios and not (m and m.group("kind").lower() in _NECESARIOS):
                continue
            todos.append(Path(d) / f)
    if not todos:
        raise FileNotFoundError(f"{src} no tiene archivos <ID>_face_scan.stl / <ID>_mask.stl")
    total = sum(p.stat().st_size for p in todos)
    log.info(f"[P0.3] copiando {len(todos)} archivos ({total / 1e9:.1f} GB) de {src} a {dst}")
    t0, hecho = time.time(), 0
    for i, p in enumerate(todos, 1):
        q = dst / p.relative_to(src)
        size = p.stat().st_size
        if not (q.exists() and q.stat().st_size == size):
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, q)
        hecho += size
        if i % 100 == 0 or i == len(todos):
            dt = max(time.time() - t0, 1e-6)
            log.info(f"[P0.3] {i}/{len(todos)} archivos, {hecho / 1e9:.1f} de {total / 1e9:.1f} GB, "
                     f"{hecho / 1e6 / dt:.0f} MB/s")
    return dst


def paso_indice(cfg: Config, autoprueba: bool = True) -> dict:
    with paso("P1 indexador", "¿data_root apunta a '1. Training' o a la carpeta que la contiene?"):
        if autoprueba:
            code = idx.self_test()
            if code != 0:
                raise RuntimeError("La autoprueba del indexador falló; no sigas hasta revisarlo.")
        cfg.index_dir.mkdir(parents=True, exist_ok=True)
        rep = idx.run_index(Path(cfg.data_root), cfg.index_dir, k=cfg.k_folds, seed=cfg.fold_seed,
                            expect=cfg.expect_pairs, exclude_nested=cfg.exclude_nested)
        log.info(f"[P1] pares completos: {rep['totals']['complete_pairs_face_and_mask']}")
        return rep


def paso_verificaciones(cfg: Config, n_a1: int = 20, n_a3: int = 10) -> tuple[list, list]:
    with paso("P2 verificaciones A1 y A3", "¿corriste P1? ¿el ZIP está extraído en esta sesión (P0.3)?"):
        a1 = checks.check_a1(cfg, n_a1)
        a3 = checks.check_a3(cfg, n_a3)
        return a1, a3


def paso_preprocesamiento(cfg: Config, limit: int | None = None, procesos: int = 0) -> list[dict]:
    with paso("P3 preprocesamiento", "revisa en la bitácora qué paciente falló; los demás se guardan igual"):
        return preprocess.run_preprocess(cfg, limit=limit, procesos=procesos)


def paso_exclusiones(cfg: Config, moldes_abiertos: bool = True, no_pasan_a1: bool = True,
                     min_frente_2mm: float = 0.30, max_incertidumbre_mm: float = 3.0,
                     forzar: bool = False) -> dict:
    """P3b: decide qué pacientes quedan fuera, con una regla fijada ANTES de ver resultados.

    Lee `preprocess_report.csv` de P3 y escribe `trabajo/indice/excluidos.csv`
    (pid, pliegue, motivo) y `exclusiones_resumen.json` (solo conteos). Motivos:
    - «falló P3»: el preprocesamiento no terminó para ese paciente;
    - «molde abierto»: el STL del molde no es una malla cerrada, así que su
      sólido no está definido (si `moldes_abiertos`);
    - «no pasa A1»: el molde se registró y quedó con menos de `min_frente_2mm`
      del frente a menos de 2 mm o con incertidumbre mayor a `max_incertidumbre_mm`
      (si `no_pasan_a1`).
    Si ya hay métricas de P4 en adelante con otra lista, se detiene (salvo `forzar`).
    """
    import csv
    from collections import Counter
    with paso("P3b exclusiones", "¿terminó P3? ¿existe trabajo/voxeles_G64/preprocess_report.csv?"):
        ruta = cfg.vox_dir / "preprocess_report.csv"
        if not ruta.exists():
            raise FileNotFoundError(f"No existe {ruta}: corre P3 primero.")
        with ruta.open(encoding="utf-8") as fh:
            rep = {r["pid"]: r for r in csv.DictReader(fh)}

        def num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        manifiesto = preprocess.read_manifest(cfg)
        fuera = []
        for m in manifiesto:
            r = rep.get(m["pid"])
            motivos = []
            if r is None or r.get("error"):
                motivos.append("falló P3")
            else:
                if moldes_abiertos and num(r.get("mask_watertight")) == 0:
                    motivos.append("molde abierto")
                if no_pasan_a1 and num(r.get("registrado")) == 1:
                    inc, f2 = num(r.get("incertidumbre_mm")), num(r.get("frac_frente_a_2mm"))
                    if (inc is not None and inc > max_incertidumbre_mm) or (f2 is not None and f2 < min_frente_2mm):
                        motivos.append("no pasa A1")
            if motivos:
                fuera.append({"pid": m["pid"], "fold": m["fold"], "motivo": "; ".join(motivos)})

        nueva = {f["pid"]: f["motivo"] for f in fuera}
        previa = preprocess.excluidos(cfg)
        hay_resultados = evaluate.results_path(cfg).exists() or any(cfg.run_dir.glob("pliegue_*/last.pt"))
        if hay_resultados and nueva != previa and not forzar:
            raise RuntimeError(
                "Ya hay resultados de P4 en adelante calculados con otra lista de exclusiones. "
                "Cambiar la regla ahora mezclaría cohortes: avísame antes de seguir.")

        cfg.index_dir.mkdir(parents=True, exist_ok=True)
        with preprocess.excluidos_path(cfg).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["pid", "fold", "motivo"])
            w.writeheader()
            w.writerows(fuera)
        por_motivo = Counter(x for f in fuera for x in f["motivo"].split("; "))
        quedan = Counter(m["fold"] for m in manifiesto if m["pid"] not in nueva)
        resumen = {
            "regla": {"moldes_abiertos": moldes_abiertos, "no_pasan_a1": no_pasan_a1,
                      "min_frente_2mm": min_frente_2mm, "max_incertidumbre_mm": max_incertidumbre_mm},
            "n_manifiesto": len(manifiesto), "n_excluidos": len(fuera),
            "n_cohorte": len(manifiesto) - len(fuera),
            "excluidos_por_motivo": dict(por_motivo),
            "cohorte_por_pliegue": {str(k): quedan[k] for k in sorted(quedan)},
        }
        (cfg.index_dir / "exclusiones_resumen.json").write_text(
            json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"[P3b] cohorte final: {resumen['n_cohorte']} de {len(manifiesto)}; "
                 f"excluidos {len(fuera)} {dict(por_motivo)}; por pliegue {resumen['cohorte_por_pliegue']}")
        return resumen


def paso_referencias(cfg: Config) -> Path:
    with paso("P4 B1, B2 y evaluación", "¿terminó P3? ¿el ZIP está extraído en esta sesión?"):
        baselines.run_baselines(cfg)
        return evaluate.evaluate_methods(cfg, ["B1", "B2"])


def paso_prueba_humo(cfg: Config, n_pacientes: int = 40) -> Path:
    """C0 a 16³ con 3 épocas sobre `n_pacientes` reales (otra carpeta de vóxeles).

    Usa un subconjunto para que tarde minutos; los pliegues son los mismos del
    manifiesto, así que hay pacientes de los cinco. Reutiliza los registros
    molde→rostro de la carpeta principal.
    """
    with paso("P5 prueba de humo a 16³", "si falla aquí, fallaría igual a 64³: manda la bitácora"):
        humo = dataclasses.replace(cfg, name="C0_humo", grid=16, base_channels=8, depth=3, epochs=3,
                                   patience=0, eval_samples=3000, work_dir=str(cfg.work / "prueba_humo_16"),
                                   expect_pairs=None, registros_dir=str(cfg.reg_dir))
        humo.index_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg.index_dir / "manifest.csv", humo.index_dir / "manifest.csv")
        preprocess.run_preprocess(humo, limit=n_pacientes, verbose=False)
        train.train_all(humo)
        baselines.run_baselines(humo, verbose=False)
        out = evaluate.evaluate_methods(humo, [humo.name, "B1", "B2"])
        import pandas as pd
        df = pd.read_csv(out)
        print(df.groupby("method")[["dice", "chamfer_total", "hd95_total"]].median().round(3))
        log.info("[P5] la prueba de humo terminó sin errores; los números a 16³ NO son resultados.")
        return out


def paso_entrenamiento(cfg: Config, pliegues="pendientes", max_pliegues: int | None = None) -> list[dict]:
    with paso("P6 entrenamiento", "si es 'CUDA out of memory', usa G = 32 (ver README)"):
        folds = train.pending_folds(cfg) if pliegues == "pendientes" else list(pliegues)
        if max_pliegues:
            folds = folds[:max_pliegues]
        if not folds:
            log.info("[P6] no hay pliegues pendientes.")
        return [train.train_fold(cfg, f) for f in folds]


def paso_evaluacion_c0(cfg: Config):
    with paso("P7 evaluación de C0", "¿terminaron los 5 pliegues de P6? ¿el ZIP está extraído en esta sesión?"):
        pend = train.pending_folds(cfg)
        if pend:
            log.warning(f"[P7] faltan los pliegues {pend}; se evalúan solo los terminados.")
        out = evaluate.evaluate_methods(cfg, [cfg.name, "B1", "B2"])
        import pandas as pd
        from .stats import compare
        df = pd.read_csv(out, dtype={"pid": str})
        tests = compare(df, cfg.name)
        tests.to_csv(cfg.results_dir / f"pruebas_estadisticas_G{cfg.grid}.csv", index=False)
        cols = ["comparacion", "metrica", "n", "mediana_diferencia", "p_holm", "favorece"]
        print(tests[cols].round(4).to_string(index=False))
        return tests


SUAVIZADOS = {"none": "_sin_suavizado", "laplacian": "_laplaciano"}


def paso_suavizado(cfg: Config):
    """Segunda alternativa medida sin volver a entrenar: el suavizado del molde de C0.

    Re-evalúa las mismas probabilidades de C0 sin suavizar y con Laplace, y las
    compara (Wilcoxon + Holm, solo métricas de superficie) contra C0 con Taubin,
    que es el de la tabla principal. Dice e IoU no cambian con el suavizado.
    """
    with paso("P7b comparación de suavizado", "¿terminó P7?"):
        import pandas as pd
        from .stats import PRIMARY_METRICS, compare
        out = evaluate.smoothing_path(cfg)
        for sm, suf in SUAVIZADOS.items():
            evaluate.evaluate_methods(cfg, [cfg.name], smoothing=sm, sufijo=suf, out_path=out)
        main = pd.read_csv(evaluate.results_path(cfg), dtype={"pid": str})
        df = pd.concat([main[main.method == cfg.name], pd.read_csv(out, dtype={"pid": str})])
        variantes = tuple(cfg.name + s for s in SUAVIZADOS.values())
        mets = [m for m in PRIMARY_METRICS if m not in ("dice", "iou")]
        tests = compare(df, cfg.name, baselines=variantes, metrics=mets)
        tests.to_csv(cfg.results_dir / f"pruebas_suavizado_G{cfg.grid}.csv", index=False)
        print(df.groupby("method")[["chamfer_frente", "hd95_frente", "chamfer_total", "postproc_s"]]
              .median().round(3))
        cols = ["comparacion", "metrica", "n", "mediana_diferencia", "p_holm", "favorece"]
        print(tests[cols].round(4).to_string(index=False))
        return tests


def paso_prototipo(cfg: Config, pid: str | None = None, rostro_stl: str | None = None) -> dict:
    """P9: prototipo integrado. Sin argumentos usa un paciente de la cohorte con
    el modelo de su pliegue; con `rostro_stl` procesa cualquier escaneo facial."""
    with paso("P9 prototipo integrado", "¿terminó P6? ¿están los datos en esta sesión (P0.3)?"):
        if rostro_stl:
            return prototipo.generar(cfg, rostro_stl)
        return prototipo.generar_para_paciente(cfg, pid)


def paso_paquete(cfg: Config) -> Path:
    with paso("P8 plantillas y paquete de resultados", "¿terminó P7?"):
        report.derive_templates(cfg)
        return report.build_results_pack(cfg)


def estado(cfg: Config) -> dict:
    """Qué pasos ya están hechos, leyendo lo que hay en Drive."""
    vox = cfg.vox_dir
    st = {
        "P1_indice": (cfg.index_dir / "manifest.csv").exists(),
        "P2_A1": (cfg.work / "verificaciones" / "A1_registro.csv").exists(),
        "P2_A3": (cfg.work / "verificaciones" / f"A3_solido_G{cfg.grid}.csv").exists(),
        "P3_preprocesamiento": (vox / "signs_refined.json").exists(),
        "P4_referencias": (train.pred_dir(cfg, "B2").parent / "baselines_tiempos.json").exists(),
        "P6_pliegues_pendientes": train.pending_folds(cfg),
        "P7_metricas": evaluate.results_path(cfg).exists(),
        "P7b_suavizado": (cfg.results_dir / f"pruebas_suavizado_G{cfg.grid}.csv").exists(),
        "P9_prototipo": prototipo.tiempos_path(cfg).exists(),
    }
    print(json.dumps(st, indent=1, ensure_ascii=False))
    return st
