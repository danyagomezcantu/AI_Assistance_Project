"""tesina — flujo de procesamiento de la tesina: rostro 3D → molde positivo CPAP → plantilla.

Módulos: config, bitacora (registro de mensajes y errores), index_dataset (P1),
checks (P2), geometry, register (molde→rostro) y preprocess (P3),
baselines (P4), model y train (P5–P6), metrics, evaluate y stats (P4, P7),
template y report (P8), synthetic (datos inventados para pruebas), pipeline
(un paso del cuaderno = una función).

1.3.6 (25 nov 2026): P7, P7b y P8 liberan la memoria de cada paciente (las mallas
de trimesh forman ciclos y la RAM crecía ~0.35 GB por paciente); la bitácora de
la evaluación muestra la memoria en uso cada 10 pacientes.
1.3.5 (25 nov 2026): P3 corre cada paciente en un proceso aparte, con tope de memoria
(un paciente que revienta la RAM queda con error «sin memoria» y la sesión sigue) y
varios a la vez si hay RAM.
1.3.4 (25 nov 2026): P3b, exclusión de moldes abiertos y de los que no pasan A1, sin
cambiar pliegues (excluidos.csv); la figura de A1 dibuja solo el frente del molde.
1.3.3 (25 nov 2026): P6 se detiene con un mensaje claro si cambias EPOCAS (u otro
parámetro de entrenamiento) con pliegues ya empezados.
1.3.2 (25 nov 2026): P0.3 extrae del ZIP solo rostro y molde, archivo por archivo;
si uno está dañado lo anota y sigue. La bitácora en Drive empieza en P0.3.
1.3.1 (25 nov 2026): el paquete se llama «tesina» (antes cpap_caso).
1.3.0 (25 nov 2026): tesina: prototipo integrado (P9), suavizado (P7b), viabilidad,
datos como carpeta o ZIP.
1.1.0 (24 nov 2026): molde positivo sólido, registro molde→rostro, regla
anatómica de orientación del rostro, regiones frente/base y bitácora.
"""

__version__ = "1.3.6"
