# Meituan ETA pipeline (Etapa 3)

This repository turns the Meituan INFORMS TSL Data-Driven Research Challenge delivery
data into an executable, versioned pipeline for estimating total delivery time. It was
built for Etapa 3 of the assignment, whose product is a reproducible pipeline rather
than a new model. Every transformation is a standalone DVC stage, every dataset version
is content addressed, and every model run is recorded in MLflow with its parameters,
metrics, model artefact and input data.

If you publish research based on this dataset, include the acknowledgement required by
the licence:

> This research was supported by data provided by Meituan.

## Prerequisites

Python 3.12 is the supported interpreter. Create an isolated environment and install the
pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins eleven packages: DVC, MLflow, scikit-learn, pandas, numpy,
pyarrow, scipy, matplotlib, PyYAML, LightGBM and pytest. Activate the environment before
running any command below, or call `.venv/bin/python` directly.

## Raw data prerequisite

The challenge files are **not redistributed with this repository**. They are released
under CC BY-NC 4.0 and the licence (`License.txt`) forbids redistribution, so they must
be obtained from the challenge distribution. `dvc repro` requires exactly four files
under `data/`, because DVC validates every dependency declared in `dvc.yaml` before it
runs any stage and aborts up front if one is missing. The expected layout is:

| File | Notes |
| --- | --- |
| `data/all_waybill_info_meituan_0322.csv.zip` | Required by `dvc repro` as the zipped archive; the uncompressed CSV is only a fallback for a manual bootstrap |
| `data/courier_wave_info_meituan.csv` | Required |
| `data/dispatch_rider_meituan.csv` | Required |
| `data/dispatch_waybill_meituan.csv` | Required |

Verify the layout before running the pipeline from the repository root:

```bash
ls data/all_waybill_info_meituan_0322.csv.zip data/courier_wave_info_meituan.csv data/dispatch_rider_meituan.csv data/dispatch_waybill_meituan.csv
```

The bootstrap stage copies these into `data/raw/`, which is the canonical location the
later stages read. It never moves or rewrites the originals and it fails with an
explicit message when a file is missing.

A manual bootstrap is more permissive than `dvc repro`:
`python -m meituan_eta.data.bootstrap --out-dir data/raw` also accepts the uncompressed
`all_waybill_info_meituan_0322.csv` or files at the repository root, because the module
searches `data/raw/`, then `data/`, then the repository root. Those locations do not
satisfy DVC's declared dependencies, so `dvc repro` still refuses to run. If you only
have the uncompressed CSV, create the archive first:

```bash
zip -j data/all_waybill_info_meituan_0322.csv.zip data/all_waybill_info_meituan_0322.csv
```

The raw files and every derived dataset are gitignored. Do not commit them.

## Reproduction

Run the whole pipeline from the repository root:

```bash
dvc repro
```

`dvc repro` executes the chain `bootstrap → validate → prepare → features → split →
train → evaluate` and caches each stage. `dvc dag` also shows two shortcut edges —
`bootstrap → features` (the features stage reads the materialised wave file) and
`split → evaluate` (evaluate reads the test split) — so the graph is a DAG, not a bare
line. The DVC remote is a local directory outside the repository, configured as
`../../meituan-dvc-remote` relative to `.dvc/` (that is, `../meituan-dvc-remote` from the
repository root), because the licence forbids uploading the data anywhere. `dvc repro`
works without any remote push; the remote only exists for local cache sharing.

The same stages can be run individually, in order:

```bash
python -m meituan_eta.data.bootstrap   --out-dir data/raw
python -m meituan_eta.data.validate    --waybill data/raw/all_waybill_info_meituan_0322.csv \
                                       --waves data/raw/courier_wave_info_meituan.csv \
                                       --out data/interim/validated.parquet \
                                       --report reports/validation.json
python -m meituan_eta.data.prepare     --input data/interim/validated.parquet \
                                       --out data/interim/clean.parquet \
                                       --report reports/prepare.json
python -m meituan_eta.features.build   --input data/interim/clean.parquet \
                                       --waves data/raw/courier_wave_info_meituan.csv \
                                       --out data/processed/features.parquet \
                                       --report reports/features.json
python -m meituan_eta.data.split       --input data/processed/features.parquet \
                                       --out-dir data/processed \
                                       --report reports/split.json
python -m meituan_eta.train            --train data/processed/train.parquet \
                                       --val data/processed/val.parquet \
                                       --models-dir models --runs reports/runs.json
python -m meituan_eta.evaluate         --test data/processed/test.parquet \
                                       --models-dir models --runs reports/runs.json \
                                       --metrics reports/metrics.json \
                                       --registry reports/experiment_registry.md \
                                       --registry-csv reports/experiment_registry.csv
```

Every stage also accepts `--smoke`, which applies the `smoke:` block from `params.yaml`:
`split.max_samples` drops to 5,000 and the estimator counts (`n_estimators`, `max_iter`)
drop to small values. It is the fast path for checking that the graph runs end to end
without a full training pass:

```bash
python -m meituan_eta.data.validate --smoke --out data/interim/validated.parquet
```

Inspect the graph and the metrics with:

```bash
dvc dag
dvc metrics show
dvc status
```

A clean, fully cached workspace reports:

```
Data and pipelines are up to date.
```

The pipeline definition lives in `dvc.yaml`; `dvc.lock` is committed and records the
content hashes (MD5) of every dependency and output, so a given checkout can be traced
back to the exact data that produced a metric.

## Inspecting runs

Training and evaluation write to a SQLite MLflow store, `mlflow.db`. Start the UI from
the repository root:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open `http://localhost:5000`. The `meituan-eta` experiment holds seven runs: six
trained model/stage combinations (random forest, histogram gradient boosting and
LightGBM, each in the `reduced` and `complete` feature sets) plus one `baseline-meituan`
reference run. The baseline run records the four metrics computed from the official
Meituan `estimate_arrived_time` column and has no model artefact, because it is not a
trained model.

## Experiment registry

Two artefacts in `reports/` summarise the same seven runs without starting the UI:

- `reports/experiment_registry.md` is the human-readable table.
- `reports/experiment_registry.csv` is the machine-readable version.

The CSV columns are `run_id, data_version, model, stage, parameters, mae, mse, rmse, r2,
execution_date`. `data_version` is the content hash of the dataset the run consumed.
Both files are committed; the MLflow store itself is not.

## Model card

The Etapa 4 model card (in Portuguese) for the six trained configurations and the official
baseline is [`docs/model_cards/MODEL_CARD.md`](docs/model_cards/MODEL_CARD.md). It
documents the objective, data and population, metrics, limitations, recommended and
non-recommended uses, risks, ethical aspects, model owner and a monitoring proposal. The
model metrics are tied to MLflow run IDs, and the card states the known target-encoding
leakage explicitly.

## Stage mapping

The assignment names six stages. This repository splits them across seven DVC stages,
because training needs a separate deterministic split step. The mapping is:

| Etapa 3 stage | DVC stage | Module |
| --- | --- | --- |
| Dados brutos | `bootstrap` | `meituan_eta.data.bootstrap` |
| Validação | `validate` | `meituan_eta.data.validate` |
| Preparação | `prepare` | `meituan_eta.data.prepare` |
| Features | `features` | `meituan_eta.features.build` |
| Treinamento | `split` + `train` | `meituan_eta.data.split`, `meituan_eta.train` |
| Avaliação | `evaluate` | `meituan_eta.evaluate` |

`dvc dag` shows the corresponding linear chain, and `dvc.lock` records the content
hashes that make each link traceable.

## Why these metrics

MAE in minutes is the primary operational metric. It maps one-to-one to the delivery
time error a customer actually experiences, and it is robust to outliers, so a single
bad prediction does not dominate the score. RMSE is reported alongside it because large
misses are disproportionately costly operationally: a late delivery by twenty minutes
hurts far more than two late deliveries by ten. R² shows how much of the variance in
delivery time the model explains against the variance baseline. All four metrics are
computed on the same test split and compared against the official Meituan
`estimate_arrived_time` baseline (`meituan_baseline_min`), which is logged as its own
reference run. The concrete numbers live in `reports/experiment_registry.md`, never in
this document, because they change when the pipeline is rerun.

## Known limitation

`poi_target_enc` is computed out-of-fold (5 folds, smoothing 10) on the **full cleaned
dataset before the train/test split**. As a consequence the test metrics are optimistic:
the encoding has already seen target information from rows that later land in the test
set. This behaviour is inherited verbatim from the source notebook so the numbers stay
comparable with the earlier etapas. It is documented here deliberately and is not
"fixed" in this pipeline.

## Tooling note

The assignment asks for Git plus "uma solução simples" for the pipeline and calls MLflow
"seria muito interessante". It never names DVC. Both DVC and MLflow are therefore
implementation choices made for this repository, not requirements handed down by the
assignment: DVC supplies content-addressed data versions, and MLflow supplies the
per-run record with parameters, metrics and artefacts.

## Privacy note

The derived datasets keep only the operational identifiers needed for modelling:
pseudonymised ids, coordinates and timestamps. No personal data is redistributed, raw
data stays out of git, and the MLflow store records dataset content hashes rather than
the dataset itself. The original challenge data remains subject to CC BY-NC 4.0 and is
never uploaded by this pipeline.

## Repository layout

```text
meituan_eta/
  config.py           shared constants, feature lists, params loader
  tracking.py         MLflow init, dataset logging, content hashes
  models.py           the six model/stage factories
  train.py            training stage (six MLflow runs)
  evaluate.py         test metrics, baseline run, registry rendering
  data/
    bootstrap.py      materialise data/raw/ from the challenge files
    validate.py       schema and integrity checks
    prepare.py        sentinel and coherence filters, target and baseline
    split.py          deterministic 70/20/10 split
  features/
    build.py          Manhattan, temporal, operational and target-encoding features
data/                 raw challenge files (gitignored) and derived datasets
reports/              stage JSON reports, metrics.json, experiment registry (md and csv)
tests/                pytest suite
notebooks/            the source notebook (never edited)
quality/              Etapa 1/2 verification harness (out of scope here)
```

Gitignored artefacts include `mlruns/`, `mlartifacts/`, `mlflow.db*`, `models/`,
`data/raw/`, `data/interim/`, `data/processed/`, `data/*.csv` and `data/*.zip`. Committed
artefacts include `dvc.lock`, `reports/experiment_registry.md` and
`reports/experiment_registry.csv`.
