"""
geometry.py — alineación, cuadrícula fija compartida, ocupación y superficies.

Convenciones (brief §2.2, §3.2 y §4.3)
--------------------------------------
* Marco del paciente: milímetros, tal como vienen los STL de AirTec.
* Marco alineado: x_al = (x - mean) @ R.T, con R (3×3) cuyas filas son los
  ejes principales del ROSTRO. La mascarilla recibe exactamente la misma
  transformación, lo cual solo es válido si el rostro y la mascarilla
  comparten sistema de coordenadas (supuesto A1, que verifica checks.py).
* Marco normalizado: x_n = (x_al - center) / scale, de modo que la caja del
  paciente es el cubo [-0.5, 0.5]^3. `center` y `scale` se calculan SOLO con
  el rostro, porque en un paciente nuevo no hay mascarilla de la cual tomar
  la caja; usar la caja combinada filtraría la respuesta a la entrada.
* Cuadrícula: G³ celdas; el centro de la celda i es -0.5 + (i + 0.5)/G en
  cada eje. Rostro y mascarilla se consultan en los MISMOS puntos, así que
  sus arreglos tienen la misma forma y el mismo origen (corrige el defecto de
  voxelización de noviembre de 2025, que usaba la caja de cada malla).
* Arista del vóxel en mm: h = scale / G.

Ocupación
---------
* Rostro (superficie abierta): vóxeles atravesados por la superficie,
  marcados con un muestreo denso y con semilla fija de puntos sobre la malla.
* Molde (sólido positivo con base plana): interior del sólido, decidido por
  paridad de cruces de rayos a lo largo de cada columna de la cuadrícula en
  los tres ejes y voto de mayoría. No se usa relleno de huecos (`.fill()`):
  la paridad funciona igual para un sólido macizo y para uno con huecos, y
  el voto de mayoría tolera mallas con pequeños defectos.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import trimesh


# --------------------------------------------------------------------------
# Lectura
# --------------------------------------------------------------------------

def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Carga un STL como una sola malla de triángulos (sin reparar ni rellenar)."""
    mesh = trimesh.load(str(path), force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"No se pudo leer una malla válida de {path}")
    return mesh


def sample_surface(mesh: trimesh.Trimesh, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Muestreo uniforme por área, reproducible. Devuelve (puntos, índice de cara)."""
    rng = np.random.default_rng(seed)
    areas = mesh.area_faces
    prob = areas / areas.sum()
    face_idx = rng.choice(len(areas), size=n, p=prob)
    tri = mesh.triangles[face_idx]
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    pts = tri[:, 0] + u * (tri[:, 1] - tri[:, 0]) + v * (tri[:, 2] - tri[:, 0])
    return pts, face_idx


# --------------------------------------------------------------------------
# Registro de transformación
# --------------------------------------------------------------------------

@dataclass
class Transform:
    """Todo lo necesario para ir del marco del paciente a la cuadrícula y volver."""

    R: list                 # 3×3, filas = ejes (altura, ancho, profundidad) del rostro
    pca_mean: list          # 3, media de los puntos del rostro (mm)
    center: list            # 3, centro de la caja en el marco alineado (mm)
    scale: float            # arista de la caja en mm
    grid: int               # G
    pitch_mm: float         # h = scale / G
    align: str
    frame_info: dict = field(default_factory=dict)  # diagnóstico de la regla de signos

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "Transform":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d.pop("skewness", None)
        d.pop("sign_axes", None)
        return cls(**d)

    # ---- marcos ------------------------------------------------------------
    def to_norm(self, x_mm: np.ndarray) -> np.ndarray:
        R = np.asarray(self.R)
        x_al = (np.asarray(x_mm) - np.asarray(self.pca_mean)) @ R.T
        return (x_al - np.asarray(self.center)) / self.scale

    def to_mm(self, x_n: np.ndarray) -> np.ndarray:
        R = np.asarray(self.R)
        x_al = np.asarray(x_n) * self.scale + np.asarray(self.center)
        return x_al @ R + np.asarray(self.pca_mean)


def boundary_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """Vértices del borde abierto de la malla (aristas que pertenecen a una sola cara)."""
    from trimesh.grouping import group_rows

    e = mesh.edges_sorted
    idx = group_rows(e, require_count=1)
    if len(idx) == 0:
        return np.zeros((0, 3))
    return mesh.vertices[np.unique(e[idx])]


def face_frame(face: trimesh.Trimesh, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Ejes del rostro (altura, ancho, profundidad) con signos deterministas.

    1. PCA de puntos muestreados: el eje de mayor varianza es la altura y el
       de menor varianza la profundidad.
    2. Profundidad: el escaneo es una superficie abierta hacia atrás, así que
       el centroide de la superficie queda delante del centroide de su borde
       abierto; ese vector fija el signo (positivo = hacia afuera de la cara).
       La regla anterior, por asimetría, fallaba en rostros reales porque el
       cuello y el cabello pesan más que la nariz. Si la malla no tiene borde,
       se usa la asimetría como respaldo.
    3. Punta de la nariz: promedio de los puntos a menos de 3 mm del máximo
       en profundidad.
    4. Altura: en la línea media, el dorso de la nariz (arriba) baja despacio
       y debajo de la punta hay un escalón hacia el labio; el lado con mayor
       profundidad media entre 5 y 25 mm de la punta es "arriba".
    5. Ancho = profundidad × altura, para que R sea una rotación propia.
    Devuelve (R con filas = ejes, media, diagnóstico).
    """
    mean = points.mean(axis=0)
    X = points - mean
    evals, evecs = np.linalg.eigh(X.T @ X / len(X))
    axes = evecs[:, np.argsort(evals)[::-1]].T.copy()
    info = {}
    rim = boundary_vertices(face)
    if len(rim) >= 10:
        front = mean - rim.mean(axis=0)
        sgn = np.sign(front @ axes[2]) or 1.0
        info["regla_profundidad"] = "borde"
        info["separacion_borde_mm"] = round(float(abs(front @ axes[2])), 2)
    else:
        z = X @ axes[2]
        sgn = np.sign((z ** 3).mean()) or 1.0
        info["regla_profundidad"] = "asimetria"
    axes[2] *= sgn
    z = X @ axes[2]
    nose = points[z > z.max() - 3.0].mean(axis=0)
    rel = points - nose
    u, w, zz = rel @ axes[0], rel @ axes[1], rel @ axes[2]
    mid = np.abs(w) < 8.0
    up = zz[mid & (u > 5) & (u < 25)]
    dn = zz[mid & (u < -5) & (u > -25)]
    if len(up) > 20 and len(dn) > 20:
        margin = float(up.mean() - dn.mean())
        if margin < 0:
            axes[0] = -axes[0]
        info["margen_altura_mm"] = round(abs(margin), 2)
    else:
        info["margen_altura_mm"] = None
    axes[1] = np.cross(axes[2], axes[0])
    assert np.linalg.det(axes) > 0.999
    info["nariz_mm"] = [round(float(v), 2) for v in nose]
    return axes, mean, info


def fit_transform(face: trimesh.Trimesh, grid: int, align: str = "pca",
                  scale_mode: str = "face_bbox", box_margin: float = 1.15,
                  box_mm: float = 200.0, seed: int = 0, n_pca: int = 50000) -> Transform:
    """Calcula la transformación del paciente usando solo el rostro."""
    pts, _ = sample_surface(face, n_pca, seed)
    if align == "pca":
        R, mean, info = face_frame(face, pts)
    else:
        R, mean, info = np.eye(3), np.zeros(3), {}
    al = (pts - mean) @ R.T
    lo, hi = al.min(axis=0), al.max(axis=0)
    center = (lo + hi) / 2.0
    if scale_mode == "face_bbox":
        scale = float((hi - lo).max() * box_margin)
    else:
        scale = float(box_mm)
    return Transform(
        R=R.tolist(), pca_mean=mean.tolist(), center=center.tolist(), scale=scale,
        grid=grid, pitch_mm=scale / grid, align=align, frame_info=info,
    )


def mesh_to_norm(mesh: trimesh.Trimesh, tf: Transform) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=tf.to_norm(mesh.vertices), faces=mesh.faces.copy(), process=False)


def mesh_to_mm(mesh_n: trimesh.Trimesh, tf: Transform) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=tf.to_mm(mesh_n.vertices), faces=mesh_n.faces.copy(), process=False)


def cell_centers(G: int) -> np.ndarray:
    return -0.5 + (np.arange(G) + 0.5) / G


# --------------------------------------------------------------------------
# Ocupación
# --------------------------------------------------------------------------

def surface_occupancy(mesh_n: trimesh.Trimesh, G: int, density: float = 8.0, seed: int = 0) -> np.ndarray:
    """Vóxeles que la superficie atraviesa (cascarón de ~1 vóxel)."""
    area_vox = mesh_n.area * G * G          # área expresada en caras de vóxel
    n = int(min(max(density * area_vox, 10000), 5_000_000))
    pts, _ = sample_surface(mesh_n, n, seed)
    idx = np.floor((pts + 0.5) * G).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < G), axis=1)
    occ = np.zeros((G, G, G), dtype=np.uint8)
    idx = idx[ok]
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return occ


def _ray_parity_axis(mesh_n: trimesh.Trimesh, G: int, axis: int) -> tuple[np.ndarray, int]:
    """Paridad de cruces a lo largo de `axis` para cada columna de la cuadrícula.

    Devuelve (ocupación G³ booleana, número de columnas con cruces impares).
    Una columna con número impar de cruces indica un defecto de la malla
    (agujero o borde abierto) y su resultado en ese eje no es confiable; el
    voto de mayoría entre los tres ejes lo compensa.
    """
    c = cell_centers(G)
    others = [a for a in range(3) if a != axis]
    uu, vv = np.meshgrid(c, c, indexing="ij")
    origins = np.zeros((G * G, 3))
    origins[:, others[0]] = uu.ravel()
    origins[:, others[1]] = vv.ravel()
    start = min(mesh_n.bounds[0, axis], -0.5) - 1.0
    origins[:, axis] = start
    dirs = np.zeros_like(origins)
    dirs[:, axis] = 1.0

    # Saltar columnas que no tocan la caja de la malla (ahorra mucho tiempo).
    b = mesh_n.bounds
    keep = ((origins[:, others[0]] >= b[0, others[0]]) & (origins[:, others[0]] <= b[1, others[0]]) &
            (origins[:, others[1]] >= b[0, others[1]]) & (origins[:, others[1]] <= b[1, others[1]]))
    occ_cols = np.zeros((G * G, G), dtype=bool)
    odd = 0
    ray_ids = np.nonzero(keep)[0]
    if len(ray_ids):
        locs, idx_ray, _ = mesh_n.ray.intersects_location(origins[ray_ids], dirs[ray_ids], multiple_hits=True)
        t = locs[:, axis]
        order = np.lexsort((t, idx_ray))
        idx_ray, t = idx_ray[order], t[order]
        bounds = np.searchsorted(idx_ray, np.arange(len(ray_ids) + 1))
        tol = 1e-7
        for k in range(len(ray_ids)):
            hits = t[bounds[k]:bounds[k + 1]]
            if len(hits) == 0:
                continue
            # quitar cruces duplicados (rayo que pega en una arista compartida)
            keep_h = np.concatenate([[True], np.diff(hits) > tol])
            hits = hits[keep_h]
            if len(hits) % 2:
                odd += 1
            n_before = np.searchsorted(hits, c, side="left")
            occ_cols[ray_ids[k]] = (n_before % 2) == 1
    occ = occ_cols.reshape(G, G, G)              # ejes: (others[0], others[1], axis)
    # reordenar a (x, y, z)
    perm = others + [axis]
    inv = np.argsort(perm)
    return np.transpose(occ, inv), odd


def solid_occupancy(mesh_n: trimesh.Trimesh, G: int) -> tuple[np.ndarray, dict]:
    """Material de la pared por voto de mayoría de paridad en x, y, z (sin relleno)."""
    votes = np.zeros((G, G, G), dtype=np.uint8)
    odd = []
    for axis in range(3):
        occ, n_odd = _ray_parity_axis(mesh_n, G, axis)
        votes += occ.astype(np.uint8)
        odd.append(int(n_odd))
    occ = (votes >= 2).astype(np.uint8)
    disagreement = float(((votes == 1) | (votes == 2)).sum() / max(occ.sum(), 1))
    return occ, {"odd_columns_xyz": odd, "axis_disagreement": disagreement}


# --------------------------------------------------------------------------
# De vuelta a superficie
# --------------------------------------------------------------------------

def clean_volume(prob: np.ndarray, level: float = 0.5, largest_component: bool = True) -> np.ndarray:
    """Pone en cero todo lo que no pertenece a la componente conexa más grande.

    Conserva las probabilidades dentro de la componente, para que marching
    cubes siga interpolando la superficie con precisión por debajo del vóxel.
    """
    if not largest_component:
        return prob
    from scipy.ndimage import binary_dilation, label

    above = prob >= level
    lab, n = label(above, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return prob
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = lab == int(np.argmax(sizes))
    # se conserva la componente y un vóxel de borde (para interpolar), sin las otras islas
    allowed = binary_dilation(keep) & ~(above & ~keep)
    return np.where(allowed, prob, 0.0).astype(prob.dtype)


def volume_to_mesh_norm(vol: np.ndarray, level: float = 0.5) -> trimesh.Trimesh | None:
    """Marching cubes sobre la cuadrícula; vértices en el marco normalizado."""
    from skimage import measure

    G = vol.shape[0]
    v = np.pad(vol.astype(np.float32), 1, mode="constant", constant_values=0.0)
    if v.max() <= level or v.min() >= level:
        return None
    verts, faces, _, _ = measure.marching_cubes(v, level=level)
    verts = -0.5 + (verts - 1.0 + 0.5) / G
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if len(mesh.faces) == 0:
        return None
    mesh.fix_normals()
    if mesh.volume < 0:          # normales hacia afuera del sólido
        mesh.invert()
    return mesh


def _umbrella(mesh: trimesh.Trimesh):
    """Operador de Laplace uniforme normalizado por fila (cada fila suma 1)."""
    import scipy.sparse as sp

    e = mesh.edges_unique
    n = len(mesh.vertices)
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ A


def smooth(mesh: trimesh.Trimesh, method: str = "taubin", iterations: int = 10,
           lamb: float = 0.5, mu: float = -0.53) -> trimesh.Trimesh:
    """Suavizado de Laplace o de Taubin (Taubin, 1995) con operador uniforme.

    Se implementa aquí, en coordenadas centradas, porque el operador de
    trimesh no siempre suma 1 por fila en mallas de marching cubes con
    aristas no variedad, y en coordenadas grandes (mm del escáner) eso hace
    que el paso de inflado de Taubin diverja.
    """
    m = mesh.copy()
    if method == "none" or len(m.vertices) == 0:
        return m
    W = _umbrella(m)
    c = m.vertices.mean(axis=0)
    V = m.vertices - c
    for _ in range(iterations):
        V = V + lamb * (W @ V - V)
        if method == "taubin":
            V = V + mu * (W @ V - V)
    m.vertices = V + c
    return m


def volume_to_mesh_mm(vol: np.ndarray, tf: Transform, level: float = 0.5,
                      smoothing: str = "taubin", iterations: int = 10) -> trimesh.Trimesh | None:
    """Volumen predicho → marching cubes → suavizado → milímetros del paciente."""
    m = volume_to_mesh_norm(vol, level)
    if m is None:
        return None
    m = mesh_to_mm(m, tf)
    if smoothing != "none":
        m = smooth(m, smoothing, iterations)
    return m
