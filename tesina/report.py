"""
report.py — P7/P8: estadística, plantillas, casos ilustrativos y paquete de resultados.

`build_results_pack` junta todo lo que hay que traer de vuelta al proyecto
en una sola carpeta y un ZIP:

    resultados_tesina_G{G}_{fecha}/
        resumen_resultados.md           tablas listas para leer (Markdown)
        resumen_por_metodo.csv          mediana [RIC] agrupada y media ± DE entre pliegues
        pruebas_estadisticas.csv        Wilcoxon, r, IC bootstrap, Holm
        costo_computo.csv               tiempo y memoria por pliegue (C0) y tiempos de B1/B2
        viabilidad_tecnica.json         horas de GPU, memoria pico, tiempos por paciente y del prototipo
        prototipo_tiempos.csv           tiempo de cada componente del prototipo integrado (P9)
        pruebas_suavizado.csv           C0 con Taubin contra sin suavizar y Laplace (P7b)
        metricas_por_paciente_anon.csv  una fila por (método, paciente) con ID anónimo
        plantillas_resumen.csv          derivación de plantilla por paciente (ID anónimo)
        verificaciones_resumen.csv      A1 y A3 agregados
        indice_conteos.json             conteos del indexador (sin rutas)
        exclusiones_resumen.json        regla de P3b y cuántos quedaron fuera por motivo (sin IDs)
        preprocesamiento_resumen.json   arista del vóxel, recortes, razón de volumen
        cajas_metricas.png              diagramas de caja agregados (sin geometría)
        config.json, entorno.json

Los identificadores de paciente se sustituyen por P001, P002, … ; la tabla de
correspondencia se queda en Drive (resultados/correspondencia_ids_NO_SUBIR.csv).
Ninguna geometría de paciente entra al paquete.
"""

from __future__ import annotations

import logging

import gc
import json
import platform
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import geometry as geo
from .config import Config
from .evaluate import results_path, smoothing_path
from .metrics import dice_iou
from .preprocess import paths_for
from .stats import PRIMARY_METRICS, compare, per_fold_summary
from .template import derive_template
from .train import pred_dir

log = logging.getLogger("tesina")


def derive_templates(cfg: Config, method: str | None = None, n_examples: int = 3, verbose: bool = True) -> pd.DataFrame:
    """Plantilla de cada paciente a partir de la predicción y de la referencia.

    Guarda en Drive los STL de mascarilla predicha y plantilla para el mejor,
    el mediano y el peor paciente según Chamfer total (casos ilustrativos).
    """
    method = method or cfg.name
    df = pd.read_csv(results_path(cfg), dtype={"pid": str})
    dm = df[df.method == method].copy()
    rows = []
    for pid in dm.pid:
        try:
            p = paths_for(cfg, pid)
            tf = geo.Transform.from_json(p["tf"])
            face = np.load(p["face"])
            pred = geo.clean_volume(np.load(pred_dir(cfg, method) / f"{pid}_prob.npy").astype(np.float32),
                                    cfg.threshold, cfg.largest_component) >= cfg.threshold
            ref = np.load(p["mask"])
            _, info_p = derive_template(pred, face, tf, cfg.template_margin_mm)
            _, info_r = derive_template(ref, face, tf, cfg.template_margin_mm)
        except Exception:
            log.exception(f"[P8] {pid}: falló la derivación de la plantilla; se salta")
            continue
        rows.append({"pid": pid, **{f"pred_{k}": v for k, v in info_p.items()},
                     **{f"ref_{k}": v for k, v in info_r.items()}})
        gc.collect()   # mallas de trimesh con ciclos de referencias (ver evaluate.py)
    out = pd.DataFrame(rows)

    # casos ilustrativos
    ex_dir = cfg.work / "casos_ilustrativos"
    ex_dir.mkdir(parents=True, exist_ok=True)
    for old in list(ex_dir.glob("*.stl")) + list(ex_dir.glob("*.png")):
        old.unlink()          # la selección puede cambiar entre corridas
    dm["chamfer_total"] = pd.to_numeric(dm["chamfer_total"], errors="coerce")
    dm = dm.dropna(subset=["chamfer_total"]).sort_values("chamfer_total")
    picks = {}
    if len(dm):
        picks = {"mejor": dm.pid.iloc[0], "mediano": dm.pid.iloc[len(dm) // 2], "peor": dm.pid.iloc[-1]}
    for tag, pid in list(picks.items())[:n_examples]:
        p = paths_for(cfg, pid)
        tf = geo.Transform.from_json(p["tf"])
        prob = geo.clean_volume(np.load(pred_dir(cfg, method) / f"{pid}_prob.npy").astype(np.float32),
                                cfg.threshold, cfg.largest_component)
        mesh = geo.volume_to_mesh_mm(prob, tf, cfg.threshold, cfg.smoothing, cfg.smooth_iterations)
        tmesh, _ = derive_template(prob >= cfg.threshold, np.load(p["face"]), tf, cfg.template_margin_mm)
        if mesh is not None:
            mesh.export(ex_dir / f"{tag}_{pid}_molde_predicho.stl")
        if tmesh is not None:
            tmesh.export(ex_dir / f"{tag}_{pid}_plantilla_derivada.stl")
    (ex_dir / "seleccion.json").write_text(json.dumps(picks, indent=1))
    out.to_csv(cfg.results_dir / f"plantillas_G{cfg.grid}.csv", index=False)
    if verbose:
        log.info(f"[P8] plantillas derivadas para {len(out)} pacientes; casos ilustrativos en {ex_dir}")
    return out


def _render_examples(cfg: Config, method: str) -> None:
    """Figuras de los casos ilustrativos (se quedan en Drive: son geometría de pacientes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ex_dir = cfg.work / "casos_ilustrativos"
    sel = ex_dir / "seleccion.json"
    if not sel.exists():
        return
    picks = json.loads(sel.read_text())
    for tag, pid in picks.items():
        p = paths_for(cfg, pid)
        ref = np.load(p["mask"])
        pred = geo.clean_volume(np.load(pred_dir(cfg, method) / f"{pid}_prob.npy").astype(np.float32),
                                cfg.threshold, cfg.largest_component) >= cfg.threshold
        face = np.load(p["face"])
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for a, ax in enumerate(axs):
            img = np.zeros(np.delete(ref.shape, a).tolist() + [3])
            img[..., 0] = pred.max(axis=a)
            img[..., 1] = ref.max(axis=a)
            img[..., 2] = face.max(axis=a) * 0.5
            ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower")
            ax.set_title(f"proyección eje {'xyz'[a]}")
            ax.axis("off")
        d, _ = dice_iou(pred, ref)
        fig.suptitle(f"{tag}: rojo = predicción, verde = AirTec, amarillo = coinciden (Dice {d:.3f})")
        fig.savefig(ex_dir / f"{tag}_{pid}_proyecciones.png", dpi=110)
        plt.close(fig)


def build_results_pack(cfg: Config, method: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method = method or cfg.name
    res = cfg.results_dir
    stamp = date.today().isoformat()
    pack = res / f"resultados_tesina_G{cfg.grid}_{stamp}"
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True)

    df = pd.read_csv(results_path(cfg), dtype={"pid": str})
    pids = sorted(df.pid.unique())
    anon = {pid: f"P{i + 1:03d}" for i, pid in enumerate(pids)}
    pd.DataFrame({"pid": list(anon), "id_anonimo": list(anon.values())}).to_csv(
        res / "correspondencia_ids_NO_SUBIR.csv", index=False)

    dfa = df.copy()
    dfa["pid"] = dfa.pid.map(anon)
    dfa = dfa.rename(columns={"pid": "id_anonimo"})
    dfa.to_csv(pack / "metricas_por_paciente_anon.csv", index=False)

    summ = per_fold_summary(df)
    summ.to_csv(pack / "resumen_por_metodo.csv", index=False)
    tests = compare(df, method) if {"B1", "B2"} <= set(df.method) and method in set(df.method) else pd.DataFrame()
    tests.to_csv(pack / "pruebas_estadisticas.csv", index=False)

    # ---- costo de cómputo
    cost = []
    for f in range(cfg.k_folds):
        s = cfg.run_dir / f"pliegue_{f}" / "resumen.json"
        if s.exists():
            cost.append({"metodo": method, **json.loads(s.read_text())})
    bt = pred_dir(cfg, "B1").parent / "baselines_tiempos.json"
    if bt.exists():
        for f, v in json.loads(bt.read_text()).items():
            cost.append({"metodo": "B1", "fold": int(f), "train_time_s": v["B1_s"], "n_test": v["n_test"]})
            cost.append({"metodo": "B2", "fold": int(f), "train_time_s": v["B2_s"], "n_test": v["n_test"]})
    cost_df = pd.DataFrame(cost)
    post = df.groupby("method")["postproc_s"].median().rename("posproceso_mediana_s_por_paciente")
    cost_df.to_csv(pack / "costo_computo.csv", index=False)

    # ---- viabilidad técnica, prototipo y suavizado (solo tiempos y agregados)
    (pack / "viabilidad_tecnica.json").write_text(
        json.dumps(_viabilidad(cfg, method, cost_df, df), indent=2, ensure_ascii=False))
    from .prototipo import tiempos_path
    if tiempos_path(cfg).exists():
        pt = pd.read_csv(tiempos_path(cfg))
        pt.drop(columns=[c for c in ("etiqueta",) if c in pt]).to_csv(pack / "prototipo_tiempos.csv", index=False)
    ps = res / f"pruebas_suavizado_G{cfg.grid}.csv"
    if ps.exists():
        shutil.copy(ps, pack / "pruebas_suavizado.csv")
    if smoothing_path(cfg).exists():
        sv = pd.concat([df[df.method == method], pd.read_csv(smoothing_path(cfg), dtype={"pid": str})])
        sv.groupby("method")[[c for c in PRIMARY_METRICS if c in sv and c not in ("dice", "iou")] + ["postproc_s"]] \
          .median().round(4).to_csv(pack / "suavizado_resumen.csv")

    # ---- plantillas
    tp = res / f"plantillas_G{cfg.grid}.csv"
    if tp.exists():
        t = pd.read_csv(tp, dtype={"pid": str})
        t["pid"] = t.pid.map(anon)
        t.rename(columns={"pid": "id_anonimo"}).to_csv(pack / "plantillas_resumen.csv", index=False)

    # ---- verificaciones e índice (solo agregados)
    ver = []
    vdir = cfg.work / "verificaciones"
    for name in ("A1_registro.csv", f"A3_solido_G{cfg.grid}.csv"):
        if (vdir / name).exists():
            v = pd.read_csv(vdir / name)
            col = "pasa_A1" if name.startswith("A1") else "pasa_A3"
            num = v.select_dtypes("number").drop(columns=[c for c in ("pid",) if c in v], errors="ignore")
            ver.append({"verificacion": name.split("_")[0], "n_revisados": len(v),
                        "n_pasan": int(v[col].sum()) if col in v else None,
                        **{f"mediana_{c}": float(num[c].median()) for c in num.columns if c != col}})
    pd.DataFrame(ver).to_csv(pack / "verificaciones_resumen.csv", index=False)
    ir = cfg.index_dir / "index_report.json"
    if ir.exists():
        rep = json.loads(ir.read_text())
        keep = {k: rep[k] for k in ("totals", "patient_folders_by_category",
                                     "complete_pairs_by_category", "cross_validation") if k in rep}
        keep["anomalias_por_tipo"] = {k: len(v) for k, v in rep.get("anomalies", {}).items()}
        (pack / "indice_conteos.json").write_text(json.dumps(keep, indent=2, ensure_ascii=False))
    exc = cfg.index_dir / "exclusiones_resumen.json"
    if exc.exists():
        shutil.copy2(exc, pack / "exclusiones_resumen.json")
    prep = cfg.vox_dir / "preprocess_report.csv"
    if prep.exists():
        p = pd.read_csv(prep)
        ok = p[p.get("error").isna()] if "error" in p else p
        info = {
            "n_pacientes": int(len(p)), "n_con_error": int(len(p) - len(ok)),
            "arista_voxel_mm_mediana": float(ok.pitch_mm.median()),
            "arista_voxel_mm_min_max": [float(ok.pitch_mm.min()), float(ok.pitch_mm.max())],
            "caja_mm_mediana": float(ok.box_mm.median()),
            "frac_mascarilla_fuera_de_caja_mediana": float(ok.mask_outside_box_frac.median()),
            "frac_mascarilla_fuera_de_caja_max": float(ok.mask_outside_box_frac.max()),
            "razon_volumen_mediana": float(pd.to_numeric(ok.mask_volume_ratio, errors="coerce").median()),
            "mascarillas_cerradas_frac": float(ok.mask_watertight.mean()),
            "giros_de_signo": ok.sign_flip.value_counts().to_dict() if "sign_flip" in ok else {},
            "segundos_por_paciente_mediana": float(ok.seconds.median()),
        }
        if "registrado" in ok:
            r = ok[pd.to_numeric(ok.registrado, errors="coerce") == 1]
            info["registro_molde_rostro"] = {
                "n_registrados": int(len(r)),
                "residuo_mediana_mm_mediana": float(pd.to_numeric(r.residuo_mediana_mm, errors="coerce").median()),
                "incertidumbre_mm_mediana": float(pd.to_numeric(r.incertidumbre_mm, errors="coerce").median()),
                "incertidumbre_mm_p90": float(pd.to_numeric(r.incertidumbre_mm, errors="coerce").quantile(0.9)),
                "frac_frente_a_2mm_mediana": float(pd.to_numeric(r.frac_frente_a_2mm, errors="coerce").median()),
                "n_incertidumbre_mayor_3mm": int((pd.to_numeric(r.incertidumbre_mm, errors="coerce") > 3).sum()),
            }
        (pack / "preprocesamiento_resumen.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

    # ---- figura agregada
    mets = ["dice", "chamfer_frente", "hd95_frente", "chamfer_base", "hd95_total"]
    methods = [m for m in (method, "B1", "B2") if m in set(df.method)]
    fig, axs = plt.subplots(1, len(mets), figsize=(3.2 * len(mets), 3.6))
    for ax, met in zip(axs, mets):
        data = [pd.to_numeric(df[df.method == m][met], errors="coerce").dropna() for m in methods]
        try:
            ax.boxplot(data, tick_labels=methods, showfliers=True)
        except TypeError:  # matplotlib < 3.9
            ax.boxplot(data, labels=methods, showfliers=True)
        ax.set_title(met + ("" if met == "dice" else " (mm)"), fontsize=9)
    fig.tight_layout()
    fig.savefig(pack / "cajas_metricas.png", dpi=150)
    plt.close(fig)
    _render_examples(cfg, method)

    cfg.save(pack / "config.json")
    env = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "sklearn", "skimage", "trimesh", "torch", "pandas"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ninguna"
    except Exception:
        pass
    (pack / "entorno.json").write_text(json.dumps(env, indent=2))

    (pack / "resumen_resultados.md").write_text(
        _markdown(cfg, method, summ, tests, cost_df, post, pack), encoding="utf-8")
    zip_path = shutil.make_archive(str(pack), "zip", root_dir=pack)
    log.info(f"[P8] paquete de resultados: {zip_path}")
    return Path(zip_path)


def _viabilidad(cfg: Config, method: str, cost_df: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Números para la viabilidad técnica y económica: tiempos de pared y memoria, sin tarifas."""
    def num(s):
        return pd.to_numeric(s, errors="coerce")

    v = {"nota": "Tiempos de pared medidos en Colab en esta corrida. El costo económico se obtiene "
                 "multiplicando las horas por una tarifa supuesta; el código no fija tarifas."}
    c = cost_df[cost_df.metodo == method] if len(cost_df) and "metodo" in cost_df else pd.DataFrame()
    if len(c) and "train_time_s" in c:
        v["entrenamiento_C0"] = {
            "pliegues_terminados": int(len(c)),
            "horas_gpu_total": round(float(num(c.train_time_s).sum()) / 3600, 3),
            "horas_por_pliegue_mediana": round(float(num(c.train_time_s).median()) / 3600, 3),
            "epocas_corridas_mediana": float(num(c.get("epochs_run")).median()) if "epochs_run" in c else None,
            "memoria_pico_gpu_mb_max": float(num(c.get("peak_gpu_mem_mb")).max()) if "peak_gpu_mem_mb" in c else None,
            "gpu": sorted(set(c["gpu_name"].astype(str))) if "gpu_name" in c else [],
            "parametros": int(c["n_parameters"].iloc[0]) if "n_parameters" in c else None,
            "inferencia_red_s_por_paciente_mediana": float(num(c.get("infer_s_per_patient")).median())
            if "infer_s_per_patient" in c else None,
        }
    for b in ("B1", "B2"):
        cb = cost_df[cost_df.metodo == b] if len(cost_df) and "metodo" in cost_df else pd.DataFrame()
        if len(cb):
            v[f"ajuste_{b}_s_total"] = round(float(num(cb.train_time_s).sum()), 2)
    prep = cfg.vox_dir / "preprocess_report.csv"
    if prep.exists():
        p = pd.read_csv(prep)
        ok = p[p["error"].isna()] if "error" in p else p
        sec = num(ok.get("seconds"))
        v["preprocesamiento_por_paciente_s"] = {
            "mediana": round(float(sec.median()), 2), "p90": round(float(sec.quantile(0.9)), 2),
            "horas_total": round(float(sec.sum()) / 3600, 3),
            "nota": "incluye el registro molde→rostro, que solo hace falta para entrenar y evaluar",
        }
    dm = df[df.method == method]
    if "postproc_s" in dm and len(dm):
        v["posproceso_y_metricas_s_por_paciente_mediana"] = round(float(num(dm.postproc_s).median()), 3)
    from .prototipo import COMPONENTES, tiempos_path
    if tiempos_path(cfg).exists():
        pt = pd.read_csv(tiempos_path(cfg))
        v["prototipo_por_paciente_s_mediana"] = {
            "n_corridas": int(len(pt)),
            "dispositivo": sorted(set(pt["dispositivo"].astype(str))) if "dispositivo" in pt else [],
            **{k: round(float(num(pt[k]).median()), 3) for k in COMPONENTES + ["total_s", "total_sin_cargar_modelo_s"]
               if k in pt},
        }
    return v


def _fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _linea_cohorte(cfg) -> list[str]:
    exc = cfg.index_dir / "exclusiones_resumen.json"
    if not exc.exists():
        return []
    e = json.loads(exc.read_text())
    motivos = ", ".join(f"{k}: {v}" for k, v in e.get("excluidos_por_motivo", {}).items()) or "ninguno"
    return [f"Cohorte: {e['n_cohorte']} de {e['n_manifiesto']} pares; excluidos {e['n_excluidos']} "
            f"({motivos}) con la regla de P3b, fijada antes de ver resultados.", ""]


def _markdown(cfg, method, summ, tests, cost_df, post, pack) -> str:
    L = [f"# Resultados de la tesina, {method} a {cfg.grid}³", "",
         f"Generado el {date.today().isoformat()}. Cinco pliegues por paciente (semilla {cfg.fold_seed}); "
         "cada paciente aparece una sola vez en prueba. Distancias en mm del paciente contra el STL original.", "",
         *_linea_cohorte(cfg),
         "## Mediana [RIC] agrupada y media ± DE entre pliegues", "",
         "| Métrica | Método | n | Mediana [Q1, Q3] | Media ± DE pliegues |", "|---|---|---|---|---|"]
    for met in PRIMARY_METRICS:
        for _, r in summ[summ.metrica == met].iterrows():
            L.append(f"| {met} | {r.metodo} | {r.n} | {_fmt(r.mediana)} [{_fmt(r.q1)}, {_fmt(r.q3)}] | "
                     f"{_fmt(r.media_pliegues)} ± {_fmt(r.de_pliegues)} |")
    if len(tests):
        L += ["", "## Comparaciones pareadas (Wilcoxon, Holm, α = 0.05)", "",
              "| Comparación | Métrica | n | Mediana dif. | IC95 % | r | p | p Holm | Favorece |",
              "|---|---|---|---|---|---|---|---|---|"]
        for _, r in tests.iterrows():
            L.append(f"| {r.comparacion} | {r.metrica} | {r.n} | {_fmt(r.mediana_diferencia)} | "
                     f"[{_fmt(r.IC95_mediana_inf)}, {_fmt(r.IC95_mediana_sup)}] | {_fmt(r.r, 2)} | "
                     f"{_fmt(r.p, 4)} | {_fmt(r.p_holm, 4)} | {r.favorece}{' *' if r.significativo_holm else ''} |")
    if len(cost_df) and "peak_gpu_mem_mb" in cost_df:
        c = cost_df[cost_df.metodo == method]
        L += ["", "## Costo de cómputo", "",
              f"Entrenamiento por pliegue: mediana {_fmt(c.train_time_s.median() / 60, 1)} min; "
              f"memoria pico de GPU: máximo {_fmt(c.peak_gpu_mem_mb.max(), 0)} MB; "
              f"GPU: {', '.join(sorted(set(c.gpu_name.astype(str))))}; parámetros: {int(c.n_parameters.iloc[0]):,}.",
              "Posprocesamiento (marching cubes, suavizado, mm, métricas), mediana por paciente: "
              + ", ".join(f"{k} {v:.2f} s" for k, v in post.items()) + "."]
    vp = pack / "viabilidad_tecnica.json"
    if vp.exists():
        via = json.loads(vp.read_text())
        e, pr, pt = via.get("entrenamiento_C0", {}), via.get("preprocesamiento_por_paciente_s", {}), \
            via.get("prototipo_por_paciente_s_mediana", {})
        L += ["", "## Viabilidad técnica", ""]
        if e:
            L.append(f"Entrenamiento de C0: {_fmt(e.get('horas_gpu_total'), 2)} h de GPU en {e.get('pliegues_terminados')} "
                     f"pliegues; inferencia de la red {_fmt(e.get('inferencia_red_s_por_paciente_mediana'), 3)} s por paciente.")
        if pr:
            L.append(f"Preprocesamiento: mediana {_fmt(pr.get('mediana'), 1)} s por paciente "
                     f"({_fmt(pr.get('horas_total'), 2)} h en total, incluye el registro).")
        if pt:
            L.append(f"Prototipo integrado (rostro STL → molde y plantilla STL): mediana {_fmt(pt.get('total_s'), 1)} s "
                     f"por paciente, {_fmt(pt.get('total_sin_cargar_modelo_s'), 1)} s sin contar la carga del modelo.")
    L += ["", "Archivos de este paquete: " + ", ".join(sorted(p.name for p in pack.iterdir())) + "."]
    return "\n".join(L) + "\n"
