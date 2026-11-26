"""
bitacora.py — registro de mensajes y errores en pantalla y en un archivo de Drive.

Cada mensaje sale en la salida de la celda con la hora y el nivel, y además
se escribe (con más detalle, incluidas las trazas completas de los errores)
en `trabajo/bitacora/bitacora_AAAA-MM-DD.txt`. Si algo falla en Colab, ese
archivo dice en qué paso, en qué paciente y en qué línea del código.

Niveles: INFO = avance normal; AVISO = algo raro pero se sigue (por ejemplo,
un registro molde→rostro incierto o un molde abierto); ERROR = un paciente o
un paso falló (la traza completa queda en el archivo).

Uso desde el cuaderno:
    from tesina.bitacora import configurar
    configurar(cfg)
y cada paso de pipeline.py se envuelve solo con `paso(...)`.
"""

from __future__ import annotations

import contextlib
import logging
import platform
import sys
import time
from datetime import date
from pathlib import Path

LOGGER = "tesina"
_NIVELES = {"WARNING": "AVISO", "ERROR": "ERROR", "INFO": "INFO", "DEBUG": "DETALLE", "CRITICAL": "ERROR"}


class _Formato(logging.Formatter):
    def format(self, record):
        record.nivel = _NIVELES.get(record.levelname, record.levelname)
        return super().format(record)


def configurar(cfg=None, nivel_pantalla: int = logging.INFO, carpeta=None) -> Path | None:
    """Configura la bitácora; se puede llamar varias veces sin duplicar mensajes.

    Sin argumentos: solo pantalla. Con `cfg`: además el archivo en cfg.log_dir.
    Con `carpeta`: el archivo en esa carpeta (sirve en P0.3, antes de tener cfg).
    El archivo es uno por día y se abre para agregar, así que P0.3 y P0.4
    escriben en el mismo.
    """
    log = logging.getLogger(LOGGER)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    pantalla = logging.StreamHandler(sys.stdout)
    pantalla.setLevel(nivel_pantalla)
    pantalla.setFormatter(_Formato("[%(asctime)s] %(nivel)s %(message)s", "%H:%M:%S"))
    log.addHandler(pantalla)
    ruta = None
    log_dir = cfg.log_dir if cfg is not None else (Path(carpeta) if carpeta else None)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        ruta = log_dir / f"bitacora_{date.today().isoformat()}.txt"
        archivo = logging.FileHandler(ruta, encoding="utf-8")
        archivo.setLevel(logging.DEBUG)
        archivo.setFormatter(_Formato("%(asctime)s %(nivel)s [%(module)s:%(lineno)d] %(message)s"))
        log.addHandler(archivo)
        log.info(f"bitácora en {ruta}")
        _entorno(log)
    return ruta


def _entorno(log: logging.Logger) -> None:
    partes = [f"python {platform.python_version()}"]
    for mod in ("numpy", "scipy", "trimesh", "skimage", "torch"):
        try:
            partes.append(f"{mod} {__import__(mod).__version__}")
        except Exception:
            partes.append(f"{mod} NO INSTALADO")
    try:
        import torch
        partes.append("GPU " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ninguna"))
    except Exception:
        pass
    log.debug("entorno: " + ", ".join(partes))


@contextlib.contextmanager
def paso(nombre: str, sugerencia: str = ""):
    """Anota inicio, fin y duración de un paso; si falla, deja la traza y una pista."""
    log = logging.getLogger(LOGGER)
    t0 = time.time()
    log.info(f"===== {nombre}: inicio =====")
    try:
        yield
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            log.warning(f"{nombre}: detenido a mano tras {time.time() - t0:.0f} s; al volver a correrlo continúa")
        elif isinstance(exc, SystemExit):
            log.error(f"{nombre}: el paso se detuvo a propósito (código {exc.code}); lee el diagnóstico de arriba")
        else:
            log.exception(f"{nombre}: FALLÓ tras {time.time() - t0:.0f} s con {type(exc).__name__}: {exc}")
            if sugerencia:
                log.error(f"{nombre}: qué revisar → {sugerencia}")
            archivos = [h.baseFilename for h in log.handlers if isinstance(h, logging.FileHandler)]
            if archivos:
                log.error(f"la traza completa también quedó en {archivos[0]}")
            else:
                log.error("todavía no hay bitácora en Drive: copia el error de arriba tal cual")
        raise
    log.info(f"===== {nombre}: terminado en {time.time() - t0:.0f} s =====")
