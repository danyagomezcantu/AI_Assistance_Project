# Custom CPAP mask moulds from 3D face scans

This repository holds the code and anonymized results of my undergraduate thesis (*tesina*, Computer Engineering, ITAM, 2026):

> **Deep-learning design of moulds for custom CPAP masks from 3D facial scans: prototype and viability evaluation**
> (Spanish title: *Diseño de moldes para mascarillas CPAP personalizadas con aprendizaje profundo a partir de escaneos faciales 3D*)

The method works in two stages. A 3D U-Net predicts the positive mould of a custom mask from the patient's face scan. That mould is then turned into a printable template.

The work started as part of the ZIM project **KI-Assistenz** at TH Köln, together with AirTec Beatmungshilfen, which makes custom ventilation masks. The data are anonymized face scans and the moulds that AirTec's technicians made for those patients. **No patient geometry is in this repository.**

_**Note: All previous work from 2024 and 2025 is stored in the "legacy-2024-2025" branch.**_

## Pipeline

The Colab notebook `cuaderno_principal_tesina.ipynb` runs one step per section. Every step saves its output to Google Drive and skips work that is already done, so a step can be re-run or resumed after a disconnection.

|Step|What it does|
|-|-|
|P0|Mounts Drive, loads the code, copies only the face and mould STL files to the VM's local disk, and builds the configuration.|
|P1|Indexes face–mould pairs and assigns each patient to one of 5 folds (seed 42).|
|P2|Runs checks A1 (mould-to-face registration) and A3 (solid voxelization) on a sample of patients.|
|P3|Orients each face (PCA plus an anatomical sign rule), registers the mould to the face (rigid ICP), and voxelizes both on a shared 64³ grid.|
|P3b|Excludes open moulds and pairs that fail A1. The rule was fixed before any results were seen.|
|P4|Builds the two reference methods and the evaluation path. **B1** gives every patient the same average mould of the training fold. **B2** reuses the mould of the training patient whose face is most similar.|
|P5|Smoke test of the full path at 16³.|
|P6|Trains the 3D U-Net (C0) for each fold, with early stopping on an inner validation split.|
|P7|Evaluates C0 against B1 and B2: Dice, IoU, and Chamfer and HD95 distances by region, with paired Wilcoxon tests and Holm correction.|
|P7b|Compares mesh smoothing on the same predictions (Taubin, none, Laplacian).|
|P8|Derives the templates, draws illustrative cases and writes the anonymized results pack.|
|P9|Integrated prototype: one face STL in, one mould STL and one template STL out.|

## Results

Run of 25 Nov 2026 on a Tesla T4, n = 223 patients, 5-fold cross-validation. Each patient is tested once by a model that never saw them. Values are medians.

|Method|Dice|Chamfer, front (mm)|Chamfer, total (mm)|HD95, total (mm)|
|-|-|-|-|-|
|C0 (3D U-Net)|0.865|1.81|3.24|9.51|
|B2 (mould of the most similar face)|0.770|3.69|5.22|13.34|
|B1 (average mould)|0.749|5.21|5.90|14.90|

* On the front of the mould, C0 is closer to AirTec's mould than B2 for 99.1 % of patients, and closer than B1 for all of them.
* All 16 paired comparisons are significant after Holm correction (p ≤ 1.8 × 10⁻¹⁹).
* Training took 0.31 GPU-hours.
* The prototype turns a face scan into a mould and a template in a median of 1.96 s (0.81 s without loading the model).
* All 223 templates are closed meshes.

The full tables, per-patient metrics, statistics, configuration and environment are in [`results/`](results/).

### Cohort

The file repository has 368 patients with a face scan and a mould. One mould file is damaged in the archive, which leaves 367 pairs. Of those, 263 moulds are closed meshes, and 223 of them pass the registration check A1.

### Patient IDs

Patients appear under two IDs:

* AirTec's five-digit code (for example `00617`), which is already a pseudonym.
* A thesis ID, `P001`–`P223`, which numbers the cohort in ascending order of AirTec code.

[`results/anexo/trazabilidad_368_pacientes.csv`](results/anexo/trazabilidad_368_pacientes.csv) lists all 368 patients with both IDs, their fold, whether the mould is closed, the registration check, the reason for exclusion (if any) and the C0 metrics.

## Running it

**In Google Colab, with the data:**

1. Put the dataset (`1. Training`, as a folder or a ZIP) in your Google Drive.
2. Make the code available in one of two ways. You can set `GIT_URL` in the notebook's first cell to this repository's URL, and P0 clones it. Or you can zip the `tesina/` folder, together with `requirements.txt`, into `tesina.zip` and put it at the path in `ZIP_CODIGO`.
3. Open `cuaderno_principal_tesina.ipynb` and edit the first cell (paths, `G`, `EPOCAS`). Then run P0 to P9 in order.
4. P1–P4 run on CPU. P5–P6 need a GPU; a T4 is enough (about 4 s per epoch at 64³).

**Without the data:**

* Set `MODO_SINTETICO = True` in the notebook. It runs P0–P9 on synthetic data.
* Or run the smoke test locally:

```
  python prueba_humo_sintetica.py --dir humo --grid 16 --pairs 20 --epochs 3
  ```

The step-by-step guide in Spanish is [`README_guia_paso_a_paso.md`](README_guia_paso_a_paso.md).

## Reproducibility

* **Seeds:** folds 42, training 1234, sampling and evaluation 0.
* **Saved with every run:** `config.json` and `entorno.json` (library versions and GPU).
* **Rules fixed in advance.** The exclusion rule (P3b) was fixed before the reference methods were evaluated. The code refuses to change training settings once a fold has started, and refuses to change exclusions after P4.
* **Log.** Every step writes a timestamped log (*bitácora*) to Drive: what ran, with which settings, how long it took, and any warning or error. The log contains patient IDs and Drive paths, so it is not published.
* **Versions.** The results come from versions 1.3.3 to 1.3.6 of the `tesina` package. These versions differ only operationally (memory handling, messages, the P3b step), not in geometry, model or metrics. The tag `v1.3.6` marks the code at the end of the run. The changelog is in `tesina/__init__.py`.
* **Determinism.** B1, B2 and all metrics are deterministic for the same data. GPU training can vary slightly between runs even with fixed seeds.

## Layout

```
cuaderno_principal_tesina.ipynb   Colab notebook, P0–P9 (published without outputs)
tesina/                           Python package: one module per stage
prueba_humo_sintetica.py          end-to-end smoke test on synthetic data
figuras_tesina.py                 thesis figures from the results pack
requirements.txt
README_guia_paso_a_paso.md        step-by-step guide (Spanish)
results/
  resultados_tesina_G64_2026-11-25/   anonymized results pack from P8
  anexo/trazabilidad_368_pacientes.csv
```

The code, comments, notebook and logs are in Spanish, like the thesis.

## Branches

* `main`: the thesis pipeline (this page).
* `legacy-2024-2025`: earlier scripts, from July 2024 to October 2025. They cover file organization, PCA alignment, first voxelization and a first 3D CNN. That branch's README explains each script and why it was replaced.

## Citation

Gómez Cantú, D. C. (2026). *Diseño de moldes para mascarillas CPAP personalizadas con aprendizaje profundo a partir de escaneos faciales 3D: prototipo y evaluación de viabilidad* [Tesina de licenciatura]. Instituto Tecnológico Autónomo de México.

## Acknowledgements

* Dr. Edgar Francisco Román Rangel (ITAM), thesis advisor.
* The KI-Assistenz team at TH Köln, in particular Dr. María Elena Algorri and Dr. Beate Rhein.
* AirTec Beatmungshilfen, for the anonymized data and permission to use it.

