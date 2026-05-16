# blackheart-train

Phase 2 / M5 training worker for the Blackheart ML/sentiment integration. Reads `feature_values` + labels from Postgres, trains LightGBM sub-models, and writes content-addressed artifacts to local disk.

**Scope of M5a (this milestone):** the skeleton — load data, train one model end-to-end, write an artifact. No walk-forward (M5c), no gauntlet (M5d), no `model_registry` write (M5e).

## Quick start

```powershell
cd C:\Project\blackheart-train
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env   # then edit DB creds if not local-dev defaults

.\.venv\Scripts\python.exe -m blackheart_train.cli --model regime_btc_v1
```

The CLI loads features over the locked window, trains one LightGBM model, and prints a JSON summary including the artifact `sha256`.

## Sub-model specs (per blueprint § 6.1)

| Spec name | Purpose | Label | Objective |
|---|---|---|---|
| `regime_btc_v1` | regime | `label_regime_risk_on_48h` | binary |
| `positioning_btc_v1` | positioning | `label_meanrev_24h` | regression |
| `flow_btc_v1` | flow | `label_return_7d` | regression |

## DB role

Connects as `blackheart_research` (V14 — read-only on operational tables; V66 — write-allowed on `model_registry` / `training_run` once M5e lands). M5a only reads `feature_values` + `feature_registry`.

## Layout

```
src/blackheart_train/
├── settings.py    pydantic-settings, TRAIN_* env prefix
├── db.py          psycopg connection helper (UTC session)
├── specs.py       ModelSpec + the three locked sub-model specs
├── loader.py      feature_values → wide matrix; PIT-safe label join
├── train.py       LightGBM training wrapper, train/val split, metrics
├── artifacts.py   content-addressed local-FS storage (sha256 keyed)
└── cli.py         CLI entry point
```
