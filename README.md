# MIT-BIH ECG Arrhythmia Classification

An experimental Jupyter notebook project for preparing ECG heartbeat segments and training a dense neural-network classifier on the MIT-BIH Arrhythmia Database.

## Notebooks

- `dataset.ipynb` — loads PhysioNet records with WFDB, extracts ECG windows around annotated beats, and explores label distributions.
- `ANN_version.ipynb` — builds training segments and trains an ANN classifier.

The current preprocessing focuses on the first ECG channel and uses short windows around annotated beats. The notebooks are exploratory and include uneven class distributions; accuracy alone can hide poor performance on rare beat types.

## Run locally

```bash
git clone https://github.com/sparsh0106/ECG-Arrhythmia-Classification.git
cd ECG-Arrhythmia-Classification
python -m pip install numpy pandas wfdb matplotlib seaborn tensorflow jupyter
jupyter notebook
```

The notebooks retrieve MIT-BIH records through PhysioNet/WFDB. Review the dataset's terms and attribution requirements before using or redistributing derived data.

## Important

This is an educational research prototype, not a medical device. It is not validated for diagnosis, treatment, or clinical decision-making.
