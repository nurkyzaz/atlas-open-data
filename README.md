# ATLAS Open Data Anomaly Detection

This project is a small benchmark study in anomaly detection on ATLAS Open Data.

The main question is simple:

Can a model learn what ordinary top-like particle-collision events look like, and then give higher scores to events that look different?

In this repository:

- one row means one proton-proton collision,
- each row contains a small set of physics features,
- the models are trained only on ordinary events,
- the models are then tested on known beyond-the-Standard-Model benchmark samples.

This is not a discovery claim.

It is a clean first step toward a more serious anomaly-detection study.

## Why this project exists

Most new-physics searches start with a specific hypothesis.

For example:

- look for a heavy new particle with a known decay pattern,
- choose cuts that are good for that one signal,
- test whether the data contains more events than expected in that region.

This project asks a different question:

Instead of telling the model exactly what signal to look for, can we teach it what ordinary events look like and then ask it to flag events that do not look ordinary?

That is why this is called anomaly detection.

## What the pipeline does

The repository does four things:

1. It reads ATLAS PHYSLITE ROOT files.
2. It converts selected collisions into a flat table.
3. It trains simple unsupervised models.
4. It checks whether those models give high scores to known benchmark signals.

The three baseline models are:

- Isolation Forest
- k-nearest-neighbor distance
- autoencoder

## What one row contains

Each row is one selected collision.

The current row contains features such as:

- `n_jets`: number of selected jets
- `n_bjets`: number of selected b-tagged jets
- `lep_pt`: momentum of the selected lepton in the plane transverse to the beam
- `lead_jet_pt`: transverse momentum of the highest-momentum jet
- `ht`: total visible transverse activity in the event
- `m3`: mass of a 3-jet top candidate
- `m_bb`: mass of the two jets with the largest b-tag scores
- `deltaR_lep_nearest_b`: angular distance between the lepton and the nearest b-jet

These are not abstract machine-learning variables.

They are plain physics quantities with direct meaning.

## Current sample choices

Ordinary training sample:

- `601497` `ttbar single lep`

Optional ordinary sample:

- `410470` `ttbar`

Robustness sample:

- `411233` alternate-generator `ttbar single lep`

Proxy anomaly samples:

- `302733` `Wprime -> tb`
- `301333` `Zprime -> ttbar`
- `523692` `stop pair -> top neutralino`

The ordinary sample teaches the model what common one-lepton top-like events look like.

The anomaly samples are not used for training.

They are only used for testing whether the model reacts to physically different event shapes.

## Current status

The code is working end to end:

- flat tables can be built,
- models can be trained,
- anomaly scores can be evaluated,
- the highest-score events can be exported to CSV.

The current results are still preliminary because the ordinary training sample is small.

That means:

- the ranking results are already useful,
- the exact false-positive rate is not yet stable.

## Repository layout

- `configs/samples.json`
  Sample definitions.
- `atlas_anomaly/io.py`
  File reading and table loading.
- `atlas_anomaly/physics.py`
  Event selection and feature building.
- `atlas_anomaly/models.py`
  Training and scoring functions.
- `scripts/build_event_table.py`
  Build flat tables from ROOT files.
- `scripts/train_baselines.py`
  Train the baseline anomaly models.
- `scripts/evaluate_baselines.py`
  Compare ordinary and anomaly samples.
- `scripts/export_top_events.py`
  Export the highest-score rows.

## How to run

Install:

```bash
cd /Users/nurkyz/Documents/Playground/atlas-open-data-anomaly
python3 -m pip install --user -r requirements.txt
```

Build a small ordinary table:

```bash
python3 scripts/build_event_table.py \
  --sample 601497 \
  --max-files-per-sample 1 \
  --max-events-per-file 300 \
  --retries 5 \
  --retry-sleep 15
```

Build the anomaly tables:

```bash
python3 scripts/build_event_table.py --sample 302733 --max-files-per-sample 1 --max-events-per-file 200 --retries 5 --retry-sleep 15
python3 scripts/build_event_table.py --sample 301333 --max-files-per-sample 1 --max-events-per-file 200 --retries 5 --retry-sleep 15
python3 scripts/build_event_table.py --sample 523692 --max-files-per-sample 1 --max-events-per-file 200 --retries 5 --retry-sleep 15
```

Train:

```bash
python3 scripts/train_baselines.py
```

Evaluate:

```bash
python3 scripts/evaluate_baselines.py
```

Export the highest-score events:

```bash
python3 scripts/export_top_events.py \
  --model artifacts/models/autoencoder.pkl \
  --table data/processed/302733.parquet \
  --top-k 20
```

## How to read the results

The main result file is:

- `artifacts/results/metrics.json`

Important fields:

- `auc`
  Measures how well the model ranks anomaly rows above ordinary rows.
  A value near `1.0` is good.
- `normal_mean_score`
  Average anomaly score on ordinary rows.
- `anomaly_mean_score`
  Average anomaly score on the proxy anomaly rows.
- `normal_flag_rate`
  Fraction of ordinary rows above the anomaly threshold.
  Smaller is better.
- `anomaly_flag_rate`
  Fraction of anomaly rows above the same threshold.
  Larger is better.

## What the current first result means

The current prototype already shows that the anomaly samples tend to get larger scores than the ordinary sample.

That is encouraging.

But this does not mean the model has discovered new physics.

It only means the current feature set and models can separate these benchmark samples from the current ordinary sample.

## Main limitations

- The ordinary training sample is still small.
- The project currently uses one final state only: one lepton plus jets.
- The current feature set does not yet include a full reconstructed missing-energy variable.
- The current models use a flat table, not full variable-length object sets.

## Next steps

The most important next steps are:

1. enlarge the ordinary sample,
2. add the alternate-generator robustness sample,
3. inspect the highest-score events in plain physics terms,
4. test whether the score creates fake mass peaks in ordinary events,
5. broaden the ordinary set with single-top samples.

## Short summary

This repository is a working first benchmark for anomaly detection on ATLAS Open Data.

It is useful because it turns a broad idea into a concrete pipeline:

- event selection,
- feature building,
- model training,
- anomaly scoring,
- physical interpretation.
