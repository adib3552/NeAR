# NeAR

NeAR is a core-based architecture for multivariate time series forecasting that jointly models temporal and cross-variate dependencies while conditioning their interaction on periodic resonance. It combines a Resonance Mixer for capturing global and harmonic periodic patterns with Nested Aggregation for exchanging information between temporal and variate representations.

## Dataset

Due to file size limitations, the datasets are hosted externally and can be downloaded from the link below:

[Download datasets](https://drive.google.com/file/d/1wB5nBVVndopl3ti3_QMUXkQisvE5G3eL/view?usp=sharing)

After downloading, place the files in the dataset directory expected by the experiment scripts.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing Experiments

After installing the dependencies, you can reproduce an experiment by running:

```bash
bash ./scripts/NeAR/ECL.sh
```
