from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.cluster import HDBSCAN


# ============================================================
# PATHS
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / "data"
RESULT_DIR = ROOT_DIR / "result"
PYTHON_RESULT_DIR = RESULT_DIR / "python"

PYTHON_RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET CONFIG
# ============================================================

DATASET_CONFIGS = {
    "circles": {
        "kind": "synthetic",
        "path": DATA_DIR / "circles_dataset.csv",
    },
    "moons": {
        "kind": "synthetic",
        "path": DATA_DIR / "moons_dataset.csv",
    },
    "varied": {
        "kind": "synthetic",
        "path": DATA_DIR / "varied_dataset.csv",
    },
    "aniso": {
        "kind": "synthetic",
        "path": DATA_DIR / "aniso_dataset.csv",
    },
    "blobs": {
        "kind": "synthetic",
        "path": DATA_DIR / "blobs_dataset.csv",
    },
    "no_structure": {
        "kind": "synthetic",
        "path": DATA_DIR / "no_structure_dataset.csv",
    },
    "heartfailure": {
    "kind": "real",
    "path": DATA_DIR / "heartfailure.csv",
    "feature_cols": [
        "Age (years)",
        "Male (1=Yes, 0=No)",
        "PHQ-9",
        "Systolic BP (mm Hg)",
        "Estimated glomerular filtration rate",
        "Ejection fraction (%)",
        "Serum sodium (mmol/l)",
        "Blood urea nitrogen (mg/dl)",
        "Etiology HF(1=Yes, 0=No)",
        "Prior diabetes mellitus",
        "Elevated level of BNP/NT-BNP (1=Yes, 0=No)",
    ],
},

"cardiacarrest": {
    "kind": "real",
    "path": DATA_DIR / "cardiacarrest.csv",
    "feature_cols": [
        "sex_woman",
        "Age_years",
        "Endotracheal_intubation",
        "Functional_status",
        "Asystole",
        "Bystander",
        "Time_min",
        "Cardiogenic",
        "Cardiac_arrest_at_home",
    ],
},

"neuroblastoma": {
    "kind": "real",
    "path": DATA_DIR / "neuroblastoma.csv",
    "feature_cols": [
        "age",
        "sex",
        "site",
        "stage",
        "time_months",
        "autologous_stem_cell_transplantation",
        "radiation",
        "degree_of_differentiation",
        "UH_or_FH",
        "MYCN_status ",
        "surgical_methods",
    ],
},

"sepsis": {
    "kind": "real",
    "path": DATA_DIR / "sepsis.csv",
    "feature_cols": [
        "Age",
        "sex_woman",
        "diagnosis_0EC_1M_2_AC",
        "APACHE II",
        "SOFA",
        "CRP",
        "WBCC",
        "NeuC",
        "LymC",
        "EOC",
        "NLCR",
        "PLTC",
        "MPV",
        "LOS-ICU",
    ],
},

"type1diabetes": {
    "kind": "real",
    "path": DATA_DIR / "type1diabetes.csv",
    "feature_cols": [
        "age",
        "duration.of.diabetes",
        "body_mass_index",
        "TDD",
        "basal",
        "bolus",
        "HbA1c",
        "eGFR",
        "perc.body.fat",
        "adiponectin",
        "free.testosterone",
        "SMI",
        "grip.strength",
        "knee.extension.strength",
        "gait.speed",
        "ucOC",
        "OC",
        "weight_kg",
        "sex_0man_1woman",
    ],
},
}


DATASETS = [
    "circles",
    "moons",
    "varied",
    "aniso",
    "blobs",
    "no_structure",
    "heartfailure",
    "cardiacarrest",
    "neuroblastoma",
    "sepsis",
    "type1diabetes",
]


# ============================================================
# HDBSCAN PARAMS
# ============================================================




# ============================================================
# HELPERS
# ============================================================

def vector_to_string(x):
    return " ".join(map(str, x))


def load_dataset(dataset_name):
    """
    Return feature matrix X for the given dataset.
    Synthetic datasets are raw numeric matrices with no header.
    Real datasets are loaded via feature_cols.
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    cfg = DATASET_CONFIGS[dataset_name]
    path = cfg["path"]

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {path}")

    if cfg["kind"] == "synthetic":
        return pd.read_csv(path, header=None).to_numpy(dtype=float)

    elif cfg["kind"] == "real":
        df = pd.read_csv(path)
        return df[cfg["feature_cols"]].to_numpy(dtype=float)

    else:
        raise ValueError(f"Unsupported dataset kind: {cfg['kind']}")


# ============================================================
# RUN ONE
# ============================================================

def run_one(dataset_name):
    print(f"Running Python HDBSCAN on {dataset_name}")

    X = load_dataset(dataset_name)

    model = HDBSCAN(**HDBSCAN_PARAMS)

    t0 = time.perf_counter()
    model.fit(X)
    t1 = time.perf_counter()

    labels = model.labels_.astype(int)
    probabilities = model.probabilities_.astype(float)

    out = pd.DataFrame([{
        "dataset": dataset_name,
        "fit_time_sec": t1 - t0,
        "labels": vector_to_string(labels),
        "probabilities": vector_to_string(probabilities),
    }])

    out_path = PYTHON_RESULT_DIR / f"{dataset_name}_hdbscan_python.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved {out_path}")


# ============================================================
# MAIN
# ============================================================

HDBSCAN_PARAMS = dict(
    min_cluster_size=6,
    min_samples=3,
#    cluster_selection_epsilon=0.0,
#    max_cluster_size=None,
#    metric="euclidean",
#    alpha=1.0,
#    leaf_size="40",
#    n_jobs=None,
#    cluster_selection_method="eom",
    allow_single_cluster=False,
#    store_centers=None,
    copy=True,
)
def main():
    for dataset_name in DATASETS:
        run_one(dataset_name)


if __name__ == "__main__":
    main()