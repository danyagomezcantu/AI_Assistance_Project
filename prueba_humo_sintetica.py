#!/usr/bin/env python3
"""
prueba_humo_sintetica.py — corre P1–P9 completos sobre un archivo sintético.

No usa ningún dato de paciente. Sirve para comprobar, en Colab o en
cualquier máquina, que el código funciona de extremo a extremo antes de
tocar el archivo real.

    python prueba_humo_sintetica.py --dir /content/humo --grid 16 --pairs 20 --epochs 3
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tesina import pipeline, synthetic  # noqa: E402
from tesina.bitacora import configurar  # noqa: E402
from tesina.config import Config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/content/humo")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--base", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--keep", action="store_true", help="no borrar la carpeta previa")
    a = ap.parse_args(argv)

    base = Path(a.dir)
    if base.exists() and not a.keep:
        shutil.rmtree(base)
    t0 = time.time()
    archive = synthetic.build_archive(base / "archivo", n_pairs=a.pairs)
    cfg = Config(name="C0_sint", data_root=str(archive), work_dir=str(base / "trabajo"),
                 expect_pairs=a.pairs, grid=a.grid, base_channels=a.base, depth=a.depth,
                 epochs=a.epochs, patience=0, eval_samples=3000, batch_size=2)
    configurar(cfg)
    tiempos = {}

    def paso(nombre, fn, *args, **kw):
        t = time.time()
        r = fn(*args, **kw)
        tiempos[nombre] = round(time.time() - t, 1)
        return r

    paso("P1", pipeline.paso_indice, cfg)
    paso("P2", pipeline.paso_verificaciones, cfg, 6, 4)
    paso("P3", pipeline.paso_preprocesamiento, cfg)
    paso("P4", pipeline.paso_referencias, cfg)
    paso("P6", pipeline.paso_entrenamiento, cfg)
    paso("P7", pipeline.paso_evaluacion_c0, cfg)
    paso("P7b", pipeline.paso_suavizado, cfg)
    paso("P9", pipeline.paso_prototipo, cfg)
    z = paso("P8", pipeline.paso_paquete, cfg)
    pipeline.estado(cfg)
    print("tiempos por paso (s):", tiempos, "total", round(time.time() - t0, 1))
    print("PRUEBA DE HUMO SINTÉTICA: OK ->", z)


if __name__ == "__main__":
    main()
