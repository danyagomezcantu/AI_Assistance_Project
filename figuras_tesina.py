"""
figuras_tesina.py — figuras de la tesina hechas a partir de los resultados reales.

Lee SOLO archivos sin geometría ni identificadores reales:
    - el paquete de resultados de P8 (resultados_tesina_G64_<fecha>/):
      metricas_por_paciente_anon.csv, indice_conteos.json, exclusiones_resumen.json
    - el historial de entrenamiento de cada pliegue:
      trabajo/corridas/C0_G64/pliegue_k/historial.json (y resumen.json si existe)

Escribe PDF vectoriales (y una copia PNG para verlas en Colab) en la carpeta
de salida. Esos PDF son los que se suben a la carpeta figuras/ de Overleaf.

    flujo_cohorte.pdf          figura del capítulo 4 (fig:cohorte)
    curvas_entrenamiento.pdf   figura del capítulo 7 (fig:curvas)
    ecdf_frente.pdf            figura del capítulo 7 (fig:ecdf)
    pares_por_paciente.pdf     figura del capítulo 7 (fig:pares)
    arista_voxel.pdf           figura del capítulo 7 (fig:arista)
    medianas_por_pliegue.pdf   figura del apéndice D (fig:pliegues)

Uso en Colab (una celda, después de P8):

    %run /content/drive/MyDrive/198618_tesina/figuras_tesina.py

o, con otras rutas:

    %run figuras_tesina.py --trabajo /ruta/a/trabajo --paquete /ruta/al/paquete --salida /ruta/de/salida

No inventa ni fija ninguna cifra: todos los números que aparecen en las figuras
salen de los archivos de entrada, y el script imprime un resumen de lo que dibujó
para compararlo con el texto de la tesina.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

# --------------------------------------------------------------------------
# Rutas por defecto (las de la guía del cuaderno)
# --------------------------------------------------------------------------
RAIZ_DRIVE = Path("/content/drive/MyDrive/198618_tesina")
TRABAJO_DEF = RAIZ_DRIVE / "trabajo"
SALIDA_DEF = RAIZ_DRIVE / "figuras_tesina"

# --------------------------------------------------------------------------
# Estilo: una sola paleta, en orden fijo, y la misma identidad en todas las
# figuras. Cada serie lleva además su propio tipo de línea o de marcador, para
# que no dependa solo del color (impresión en gris y daltonismo).
# --------------------------------------------------------------------------
METODOS = ["C0", "B2", "B1"]
NOMBRE_METODO = {
    "C0": "C0 (U-Net 3D)",
    "B2": "B2 (rostro más parecido)",
    "B1": "B1 (molde promedio)",
}
COLOR_METODO = {"C0": "#257de0", "B2": "#f47230", "B1": "#00b894"}
LINEA_METODO = {"C0": "-", "B2": "--", "B1": "-."}
MARCA_METODO = {"C0": "o", "B2": "s", "B1": "^"}

COLOR_PLIEGUE = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"]
LINEA_PLIEGUE = ["-", "--", "-.", ":", (0, (6, 1.5, 1, 1.5, 1, 1.5))]

TINTA = "#222222"      # texto
TINTA_2 = "#5c5c5c"    # texto secundario
EJE = "#8c8c8c"        # ejes
REJILLA = "#e3e3e3"    # rejilla
SUPERFICIE = "#ffffff"

ANCHO = 6.3            # pulgadas: el ancho del texto en la tesina (carta, 2.5 cm)


def estilo() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "text.color": TINTA,
        "axes.labelcolor": TINTA,
        "axes.edgecolor": EJE,
        "axes.linewidth": 0.6,
        "xtick.color": TINTA_2,
        "ytick.color": TINTA_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": True,
        "grid.color": REJILLA,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,     # fuentes incrustadas como TrueType (Overleaf las respeta)
        "ps.fonttype": 42,
    })


def guardar(fig, salida: Path, nombre: str) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida / f"{nombre}.pdf")
    fig.savefig(salida / f"{nombre}.png", dpi=200)
    plt.close(fig)
    print(f"  ✓ {nombre}.pdf")


def ticks_log_mm(ax, eje: str, valores) -> None:
    """Marcas legibles (1, 2, 5, 10, 20, 50 mm) en un eje logarítmico."""
    lo, hi = float(np.nanmin(valores)), float(np.nanmax(valores))
    marcas = [v for v in (0.5, 1, 2, 5, 10, 20, 50, 100) if lo * 0.8 <= v <= hi * 1.25]
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    a = ax.xaxis if eje == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(marcas))
    a.set_major_formatter(fmt)
    a.set_minor_locator(NullLocator())


# --------------------------------------------------------------------------
# Carga de datos
# --------------------------------------------------------------------------
def buscar_paquete(trabajo: Path) -> Path:
    candidatos = sorted((trabajo / "resultados").glob("resultados_tesina_G*"))
    candidatos = [c for c in candidatos if c.is_dir()]
    if not candidatos:
        sys.exit(f"No encontré ningún paquete de resultados en {trabajo / 'resultados'}. "
                 "Corre P8 o pasa la carpeta con --paquete.")
    return candidatos[-1]   # el más reciente (la fecha va en el nombre)


def cargar_metricas(paquete: Path) -> pd.DataFrame:
    df = pd.read_csv(paquete / "metricas_por_paciente_anon.csv")
    df = df[df.method.isin(METODOS)].copy()
    for c in ("chamfer_frente", "hd95_frente", "pitch_mm"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def cargar_historial(trabajo: Path, config: str = "C0_G64") -> list[dict]:
    base = trabajo / "corridas" / config
    pliegues = []
    for carpeta in sorted(base.glob("pliegue_*"), key=lambda p: int(p.name.split("_")[1])):
        h = json.loads((carpeta / "historial.json").read_text())
        resumen = carpeta / "resumen.json"
        mejor = json.loads(resumen.read_text()).get("best_epoch") if resumen.exists() else None
        if mejor is None:   # sin resumen: la época con el mejor Dice de validación
            mejor = max(h, key=lambda r: r["val_dice"])["epoch"]
        pliegues.append({"k": int(carpeta.name.split("_")[1]), "h": h, "mejor": int(mejor)})
    if not pliegues:
        sys.exit(f"No encontré historial.json en {base}/pliegue_*/")
    return pliegues


# --------------------------------------------------------------------------
# Figura: flujo de la cohorte (capítulo 4)
# --------------------------------------------------------------------------
def fig_flujo_cohorte(paquete: Path, salida: Path) -> dict:
    ind = json.loads((paquete / "indice_conteos.json").read_text())
    exc = json.loads((paquete / "exclusiones_resumen.json").read_text())

    n_carpetas = int(ind["totals"]["patient_folders_found"])
    n_pares = int(ind["totals"]["complete_pairs_face_and_mask"])
    n_manifiesto = int(exc["n_manifiesto"])
    n_abiertos = int(exc["excluidos_por_motivo"].get("molde abierto", 0))
    n_cerrados = n_manifiesto - n_abiertos
    n_cohorte = int(exc["n_cohorte"])
    n_no_a1 = n_cerrados - n_cohorte            # cerrados que no pasan A1
    por_pliegue = {int(k): int(v) for k, v in exc["cohorte_por_pliegue"].items()}
    if sum(por_pliegue.values()) != n_cohorte:
        print("  ! La suma por pliegue no coincide con la cohorte; revisa P3b antes de usar la figura.")

    fig, ax = plt.subplots(figsize=(ANCHO, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def caja(x, y, w, h, texto, grueso=False, gris=False, tam=8.5):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.4,rounding_size=1.5",
            fc="#f4f6f9" if not gris else "#fafafa",
            ec=TINTA if grueso else EJE, lw=1.0 if grueso else 0.7))
        ax.text(x, y, texto, ha="center", va="center", fontsize=tam,
                color=TINTA if not gris else TINTA_2, linespacing=1.35)

    def flecha(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=EJE, lw=0.8,
                                    shrinkA=0, shrinkB=0, mutation_scale=8))

    xc, w, h = 33, 50, 9
    niveles = [
        (91, f"$\\bf{{{n_carpetas}}}$ carpetas de paciente en el archivo", "P1"),
        (71, f"$\\bf{{{n_pares}}}$ pares rostro–molde legibles", "P1"),
        (51, f"$\\bf{{{n_cerrados}}}$ moldes cerrados", "P3"),
        (31, f"$\\bf{{{n_cohorte}}}$ pacientes en la cohorte\n(registro molde–rostro que pasa A1)", "P3b"),
    ]
    salidas = [
        f"{n_carpetas - n_pares} sin molde legible\n(archivo dañado)",
        f"{n_abiertos} moldes abiertos\n(malla no cerrada)",
        f"{n_no_a1} moldes cerrados\nque no pasan A1",
    ]
    for i, (y, texto, paso) in enumerate(niveles):
        caja(xc, y, w, h if i < 3 else h + 3, texto, grueso=(i == 3))
        ax.text(xc - w / 2 - 2.5, y, paso, ha="right", va="center", fontsize=7.5,
                color=TINTA_2, style="italic")
        if i < 3:
            y_sig = niveles[i + 1][0]
            flecha(xc, y - h / 2 - 0.6, xc, y_sig + (h if i + 1 < 3 else h + 3) / 2 + 0.6)
            ym = (y + y_sig) / 2
            ax.plot([xc, 64], [ym, ym], color=EJE, lw=0.8)
            flecha(64, ym, 67, ym)
            caja(83, ym, 30, 8, salidas[i], gris=True, tam=7.5)

    # pliegues de prueba
    ax.text(50, 17.5, "Validación cruzada de cinco pliegues: pacientes de prueba en cada pliegue",
            ha="center", va="center", fontsize=8, color=TINTA_2)
    flecha(xc, 31 - (h + 3) / 2 - 0.6, xc, 21)
    xs = np.linspace(12, 88, 5)
    for x, k in zip(xs, sorted(por_pliegue)):
        caja(x, 8, 14.5, 8, f"Pliegue {k}\n$\\bf{{{por_pliegue[k]}}}$", tam=8)

    guardar(fig, salida, "flujo_cohorte")
    return {"carpetas": n_carpetas, "pares": n_pares, "cerrados": n_cerrados,
            "cohorte": n_cohorte, "abiertos": n_abiertos, "no_A1": n_no_a1,
            "por_pliegue": por_pliegue}


# --------------------------------------------------------------------------
# Figura: curvas de entrenamiento (capítulo 7)
# --------------------------------------------------------------------------
def fig_curvas(pliegues: list[dict], salida: Path) -> dict:
    fig, (ax_l, ax_d) = plt.subplots(1, 2, figsize=(ANCHO, 2.7))
    for p in pliegues:
        k = p["k"]
        ep = np.array([r["epoch"] for r in p["h"]])
        loss = np.array([r["loss"] for r in p["h"]])
        vd = np.array([r["val_dice"] for r in p["h"]])
        estilo_k = dict(color=COLOR_PLIEGUE[k], ls=LINEA_PLIEGUE[k], lw=1.3)
        ax_l.plot(ep, loss, label=f"Pliegue {k}", **estilo_k)
        ax_d.plot(ep, vd, **estilo_k)
        i = int(np.where(ep == p["mejor"])[0][0])
        ax_d.plot(ep[i], vd[i], "o", ms=5.5, color=COLOR_PLIEGUE[k],
                  mec=SUPERFICIE, mew=1.2, zorder=5)

    ax_l.set_title("Pérdida de entrenamiento")
    ax_l.set_ylabel("Dice + BCE")
    ax_d.set_title("Dice de validación")
    ax_d.set_ylabel("Dice")
    for ax in (ax_l, ax_d):
        ax.set_xlabel("Época")
        ax.set_xlim(left=0)
    h, l = ax_l.get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="o", ls="none", ms=5.5, color=TINTA_2, mec=SUPERFICIE))
    l.append("Mejor época")
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1),
               handlelength=2.6, columnspacing=1.2)
    fig.tight_layout()
    guardar(fig, salida, "curvas_entrenamiento")
    return {"mejor_epoca": [p["mejor"] for p in pliegues],
            "epocas_corridas": [len(p["h"]) for p in pliegues]}


# --------------------------------------------------------------------------
# Figura: distribución acumulada por paciente (capítulo 7)
# --------------------------------------------------------------------------
def fig_ecdf(df: pd.DataFrame, salida: Path) -> dict:
    metricas = [("chamfer_frente", "Chamfer en el frente"),
                ("hd95_frente", "HD95 en el frente")]
    fig, axs = plt.subplots(1, 2, figsize=(ANCHO, 2.9), sharey=True)
    medianas = {}
    for ax, (col, titulo) in zip(axs, metricas):
        for m in METODOS:
            v = np.sort(df.loc[df.method == m, col].dropna().to_numpy())
            y = np.arange(1, len(v) + 1) / len(v)
            ax.step(v, y, where="post", color=COLOR_METODO[m], ls=LINEA_METODO[m],
                    lw=1.4, label=NOMBRE_METODO[m])
            med = float(np.median(v))
            medianas[(m, col)] = med
            ax.plot(med, 0.5, MARCA_METODO[m], ms=5.5, color=COLOR_METODO[m],
                    mec=SUPERFICIE, mew=1.0, zorder=5)
        ax.axhline(0.5, color=EJE, lw=0.6, ls=":", zorder=1)
        ax.set_xscale("log")
        ticks_log_mm(ax, "x", df[col])
        ax.set_title(titulo)
        ax.set_xlabel("mm por paciente (escala logarítmica)")
        ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("Fracción de pacientes")
    axs[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}".replace("%", " %")))
    # medianas en un bloque de texto en la esquina vacía de cada panel (sin choques)
    for ax, (col, _) in zip(axs, metricas):
        lineas = ["Mediana (mm)"] + [f"{m}   {medianas[(m, col)]:.2f}" for m in METODOS]
        ax.text(0.97, 0.05, "\n".join(lineas), transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7.5, color=TINTA, linespacing=1.4, family="DejaVu Sans Mono")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), handlelength=2.6)
    fig.tight_layout()
    guardar(fig, salida, "ecdf_frente")
    return {f"mediana_{m}_{c}": round(v, 2) for (m, c), v in medianas.items()}


# --------------------------------------------------------------------------
# Figura: comparación pareada paciente por paciente (capítulo 7)
# --------------------------------------------------------------------------
def fig_pares(df: pd.DataFrame, salida: Path, col: str = "chamfer_frente") -> dict:
    w = df.pivot_table(index="id_anonimo", columns="method", values=col).dropna()
    fig, axs = plt.subplots(1, 2, figsize=(ANCHO, 3.15), sharey=True)
    cuentas = {}
    lo = float(w.min().min()) * 0.85
    hi = float(w.max().max()) * 1.15
    for ax, ref in zip(axs, ["B2", "B1"]):
        x, y = w[ref].to_numpy(), w["C0"].to_numpy()
        gana = int((y < x).sum())
        cuentas[ref] = (gana, len(w))
        ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=COLOR_METODO["C0"], alpha=0.06, lw=0)
        ax.plot([lo, hi], [lo, hi], color=EJE, lw=0.8, ls="--", zorder=2)
        ax.scatter(x, y, s=13, color=COLOR_METODO["C0"], alpha=0.8,
                   edgecolors=SUPERFICIE, linewidths=0.4, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ticks_log_mm(ax, "x", [lo, hi])
        ticks_log_mm(ax, "y", [lo, hi])
        ax.set_title(f"C0 frente a {ref}", pad=16)
        ax.text(0, 1.015, f"C0 tiene menor error en {gana} de {len(w)} pacientes",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=TINTA_2)
        ax.set_xlabel(f"{NOMBRE_METODO[ref]}, mm")
        ax.text(0.04, 0.96, f"zona: {ref} mejor", transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, color=TINTA_2)
        ax.text(0.97, 0.03, "zona: C0 mejor", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color=TINTA_2)
    axs[0].set_ylabel(f"{NOMBRE_METODO['C0']}, mm")
    fig.tight_layout()
    guardar(fig, salida, "pares_por_paciente")
    return {f"C0_mejor_que_{r}": f"{g}/{n}" for r, (g, n) in cuentas.items()}


# --------------------------------------------------------------------------
# Figura: arista del vóxel por paciente (capítulo 7)
# --------------------------------------------------------------------------
def fig_arista(df: pd.DataFrame, salida: Path) -> dict:
    a = df[df.method == "C0"].drop_duplicates("id_anonimo").pitch_mm.dropna().to_numpy()
    med = float(np.median(a))
    fig, ax = plt.subplots(figsize=(ANCHO, 2.3))
    paso = 0.1
    bordes = np.arange(np.floor(a.min() / paso) * paso, a.max() + paso, paso)
    ax.hist(a, bins=bordes, color=COLOR_METODO["C0"], alpha=0.85,
            edgecolor=SUPERFICIE, linewidth=1.0)
    ax.axvline(med, color=TINTA, lw=0.9, ls="--")
    ax.text(med, ax.get_ylim()[1] * 0.97, f"  mediana {med:.2f} mm", ha="left", va="top",
            fontsize=7.5, color=TINTA)
    ax.set_xlabel("Arista del vóxel $h$ en la cuadrícula de $64^3$ (mm)")
    ax.set_ylabel("Pacientes")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    guardar(fig, salida, "arista_voxel")
    return {"n": int(len(a)), "arista_mediana_mm": round(med, 2),
            "arista_min_max_mm": [round(float(a.min()), 2), round(float(a.max()), 2)]}


# --------------------------------------------------------------------------
# Figura: medianas por pliegue (apéndice D)
# --------------------------------------------------------------------------
def fig_pliegues(df: pd.DataFrame, salida: Path, col: str = "chamfer_frente") -> dict:
    fig, ax = plt.subplots(figsize=(ANCHO, 2.7))
    pliegues = sorted(df.fold.unique())
    desfase = {"C0": -0.18, "B2": 0.0, "B1": 0.18}
    tabla = {}
    for m in METODOS:
        meds, q1s, q3s = [], [], []
        for k in pliegues:
            v = df.loc[(df.method == m) & (df.fold == k), col].dropna()
            meds.append(v.median())
            q1s.append(v.quantile(0.25))
            q3s.append(v.quantile(0.75))
        x = np.array(pliegues) + desfase[m]
        ax.vlines(x, q1s, q3s, color=COLOR_METODO[m], lw=1.6, alpha=0.9)
        ax.plot(x, meds, MARCA_METODO[m], ms=6, color=COLOR_METODO[m], mec=SUPERFICIE,
                mew=1.0, ls="none", label=NOMBRE_METODO[m], zorder=4)
        tabla[m] = [round(float(v), 2) for v in meds]
    ax.set_xticks(pliegues)
    ax.set_xticklabels([f"Pliegue {k}\n(n = {int((df[(df.method == 'C0') & (df.fold == k)]).shape[0])})"
                        for k in pliegues])
    ax.set_ylabel("Chamfer en el frente (mm)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    fig.tight_layout()
    guardar(fig, salida, "medianas_por_pliegue")
    return {f"medianas_{m}": v for m, v in tabla.items()}


# --------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--trabajo", type=Path, default=TRABAJO_DEF)
    ap.add_argument("--paquete", type=Path, default=None,
                    help="carpeta resultados_tesina_G64_<fecha> (por defecto, la más reciente)")
    ap.add_argument("--salida", type=Path, default=SALIDA_DEF)
    args, _ = ap.parse_known_args(argv)   # parse_known_args: tolera los argumentos de Colab

    estilo()
    paquete = args.paquete or buscar_paquete(args.trabajo)
    print(f"Paquete de resultados: {paquete}")
    print(f"Salida: {args.salida}\n")
    df = cargar_metricas(paquete)
    pliegues = cargar_historial(args.trabajo)

    resumen = {"paquete": str(paquete)}
    resumen["cohorte"] = fig_flujo_cohorte(paquete, args.salida)
    resumen["curvas"] = fig_curvas(pliegues, args.salida)
    resumen["ecdf"] = fig_ecdf(df, args.salida)
    resumen["pares"] = fig_pares(df, args.salida)
    resumen["arista"] = fig_arista(df, args.salida)
    resumen["pliegues"] = fig_pliegues(df, args.salida)

    (args.salida / "resumen_figuras.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False))
    print("\nCifras dibujadas (compáralas con el texto de la tesina):")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
