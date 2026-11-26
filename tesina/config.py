"""
config.py — un solo objeto de configuración por experimento.

Todo lo que cambia el resultado de una corrida (resolución, caja, red,
pérdida, posprocesamiento, semillas, rutas) vive en `Config`. El objeto se
guarda como JSON junto a cada resultado, de modo que cualquier fila de la
tabla final se puede reproducir a partir de su configuración (REQ-N4).

La configuración C0 del brief (§2.3) es el valor por defecto:
PCA · binario · 64³ · U-Net 3D · Dice+BCE · marching cubes + Taubin.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ------------------------------------------------------------------ nombre
    name: str = "C0"

    # ------------------------------------------------------------------ rutas
    #: Carpeta raíz del archivo ya extraído (la que contiene "1. Training").
    data_root: str = "/content/KI_Assistenz_File_Repository"
    #: Carpeta de trabajo persistente (Drive). Aquí quedan índice, .npy,
    #: checkpoints, predicciones y resultados.
    work_dir: str = "/content/drive/MyDrive/198618_tesina/trabajo"

    # ------------------------------------------------------------------ índice
    k_folds: int = 5
    fold_seed: int = 42
    expect_pairs: int | None = 368   # None = no detener si el conteo difiere
    exclude_nested: bool = False     # ver anomalía 04358/04425

    # --------------------------------------------------------- preprocesamiento
    grid: int = 64                   # G; 16 = prueba de humo, 32 o 64 = reales
    align: str = "pca"               # "pca" | "none"
    #: "face_bbox": la caja de cada paciente se calcula SOLO con el rostro
    #: (normalización por paciente, brief §2.4); así la entrada de inferencia
    #: no depende de la mascarilla, que no existe en un paciente nuevo.
    #: "fixed_mm": caja de arista fija `box_mm` para todos los pacientes.
    scale_mode: str = "face_bbox"
    box_margin: float = 1.15         # factor sobre la mayor extensión del rostro
    box_mm: float = 200.0            # solo si scale_mode == "fixed_mm"
    #: Ocupación de la mascarilla: "solid" = material de la pared por prueba de
    #: paridad de rayos en los tres ejes, SIN relleno (brief §3.2);
    #: "shell" = cascarón cercano a la superficie (respaldo documentado).
    mask_occupancy: str = "solid"
    #: Registro rígido molde → rostro (register.py). "auto" registra solo si el
    #: molde no está ya pegado al rostro; "always" siempre; "none" nunca.
    #: En el paciente 01543 el molde venía en otro sistema de coordenadas.
    register_mask: str = "auto"
    #: Carpeta donde se guardan los registros (vacío = trabajo/registros).
    #: Es independiente de G, así que P3 a 32³ y a 64³ reutilizan el mismo registro.
    registros_dir: str = ""
    face_samples_per_voxel_area: float = 8.0  # densidad de muestreo del rostro
    sample_seed: int = 0

    # --------------------------------------------------------------- modelo
    base_channels: int = 16
    depth: int = 4                   # niveles de la U-Net (G debe ser múltiplo de 2**(depth-1))
    dropout: float = 0.0

    # ------------------------------------------------------------ entrenamiento
    epochs: int = 80
    batch_size: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    bce_pos_weight: float | None = None   # None = calcular con el pliegue de entrenamiento
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    inner_val_fraction: float = 0.1  # pacientes del entrenamiento reservados para elegir la mejor época
    patience: int = 20               # épocas sin mejora antes de parar (0 = desactivar)
    amp: bool = True                 # precisión mixta en GPU
    num_workers: int = 2
    train_seed: int = 1234

    # --------------------------------------------------------- superficie
    threshold: float = 0.5
    #: conservar solo la componente conexa más grande del volumen umbralizado
    #: (quita islas sueltas que inflan HD95); se aplica igual a C0, B1 y B2
    largest_component: bool = True
    smoothing: str = "taubin"        # "none" | "laplacian" | "taubin"
    smooth_iterations: int = 10

    # --------------------------------------------------------- evaluación
    eval_samples: int = 20000        # puntos muestreados por superficie
    seal_mm: float = 3.0             # umbral de cercanía al rostro en la verificación A1
    eval_seed: int = 0

    # --------------------------------------------------------- plantilla
    template_margin_mm: float = 10.0

    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------- utilidades
    @property
    def work(self) -> Path:
        return Path(self.work_dir)

    @property
    def index_dir(self) -> Path:
        return self.work / "indice"

    @property
    def vox_dir(self) -> Path:
        return self.work / f"voxeles_G{self.grid}"

    @property
    def run_dir(self) -> Path:
        return self.work / "corridas" / f"{self.name}_G{self.grid}"

    @property
    def reg_dir(self) -> Path:
        return Path(self.registros_dir) if self.registros_dir else self.work / "registros"

    @property
    def log_dir(self) -> Path:
        return self.work / "bitacora"

    @property
    def results_dir(self) -> Path:
        return self.work / "resultados"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Config":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        step = 2 ** (self.depth - 1)
        if self.grid % step:
            raise ValueError(f"grid={self.grid} debe ser múltiplo de {step} con depth={self.depth}")
        if self.align not in ("pca", "none"):
            raise ValueError("align debe ser 'pca' o 'none'")
        if self.scale_mode not in ("face_bbox", "fixed_mm"):
            raise ValueError("scale_mode debe ser 'face_bbox' o 'fixed_mm'")
        if self.mask_occupancy not in ("solid", "shell"):
            raise ValueError("mask_occupancy debe ser 'solid' o 'shell'")
        if self.register_mask not in ("auto", "always", "none"):
            raise ValueError("register_mask debe ser 'auto', 'always' o 'none'")
        if self.smoothing not in ("none", "laplacian", "taubin"):
            raise ValueError("smoothing debe ser 'none', 'laplacian' o 'taubin'")


def smoke_config(work_dir: str, data_root: str, n_pairs: int | None = None) -> Config:
    """Configuración de prueba de humo: 16³, pocas épocas, red pequeña."""
    return Config(
        name="C0_humo",
        data_root=data_root,
        work_dir=work_dir,
        expect_pairs=n_pairs,
        grid=16,
        base_channels=8,
        depth=3,
        epochs=3,
        batch_size=2,
        patience=0,
        eval_samples=3000,
        num_workers=0,
    )
