import os
import json
import warnings
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import joblib
import torch
from kan import KAN
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------
# Configuration
# ----------------------------
EXCEL_PATH = "/Users/faysalsaylik/Desktop/KAN/MAGGIC.xlsx"
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TARGET_COL = "CIN"
NUMERIC_FEATURES: List[str] = [
    "MAGGIC_score", "HGB", "PLT", "NE", "MO", "LY", "RDW", "CRP",
    "TROP", "alb", "LDL", "HDL", "tg", "stentdiameter", "time",
    "ContrastVolume", "plasmaOsm",
]
CATEGORICAL_FEATURES: List[str] = [
    "previousCAD", "HT", "multiplelesion", "procedure",
]

# Friendly English key aliases for user-provided inputs
# Maps various english names to the canonical column names expected by the model
KEY_ALIASES: Dict[str, str] = {
    # Categorical
    "hypertension": "HT",
    "high_blood_pressure": "HT",
    "ht": "HT",
    "previous_cad": "previousCAD",
    "prior_cad": "previousCAD",
    "history_of_cad": "previousCAD",
    "multiple_lesion": "multiplelesion",
    "multi_lesion": "multiplelesion",
    "procedure": "procedure",
    # Add space/typo variants for previous CAD
    "previous cad": "previousCAD",
    "pravious cad": "previousCAD",

    # Numeric
    "maggic_score": "MAGGIC_score",  # normalized from "MAGGIC score"
    "hgb": "HGB",
    "hemoglobin": "HGB",
    "plt": "PLT",
    "platelets": "PLT",
    "neutrophils": "NE",
    "mo": "MO",
    "monocyte": "MO",
    "ly": "LY",
    "lymphocyte": "LY",
    "rdw": "RDW",
    "crp": "CRP",
    "troponin": "TROP",
    "troponin_i": "TROP",
    "albumin": "alb",
    "ldl": "LDL",
    "hdl": "HDL",
    "triglycerides": "tg",
    "triglyceride": "tg",
    "stent_diameter": "stentdiameter",
    "procedure_time": "time",
    "pain_to_balloon_time": "time",
    "contrast_volume": "ContrastVolume",
    "contrastvol": "ContrastVolume",
    "plasma_osm": "plasmaOsm",
    "plasma_osmolality": "plasmaOsm",

    # More english-friendly variants with spaces will normalize to underscores above, e.g.:
    # "previous cad" -> "previous_cad", "multiple lesion" -> "multiple_lesion"
}


def _read_excel(path: str) -> pd.DataFrame:
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel not found: {path}")
    return pd.read_excel(path)


def _encode_binary_target(series: pd.Series) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Try numeric 0/1 directly
    if pd.api.types.is_numeric_dtype(series):
        s = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
        unique_vals = sorted(set(s.unique().tolist()))
        if set(unique_vals).issubset({0, 1}):
            return s.values, {"mapping": {0: 0, 1: 1}, "positive_index": 1}
        # Common case: {0,2} where 2 means positive
        if set(unique_vals).issubset({0, 2}):
            s = s.replace(2, 1)
            return s.values, {"mapping": {0: 0, 2: 1}, "positive_index": 1}

    # Fallback: string mapping
    text = series.astype(str).str.strip().str.lower()
    positive_mask = text.str.contains(r"^(1|yes|evet|true|\+|positive|cin\+)$")
    y = np.where(positive_mask, 1, 0).astype(int)
    return y, {"mapping": {"negative": 0, "positive": 1}, "positive_index": 1}


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    missing_cols = [c for c in (NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Eksik sütun(lar): {missing_cols}")

    # Target to 0/1
    y, target_info = _encode_binary_target(df[TARGET_COL])

    # Numeric processing
    X_num = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric_means = X_num.mean().to_dict()
    X_num = X_num.fillna(numeric_means)

    # Categorical encoding (fit a separate encoder per column)
    encoders: Dict[str, LabelEncoder] = {}
    categorical_modes: Dict[str, Any] = {}
    X_cat_cols: List[pd.Series] = []
    synonyms: Dict[str, Any] = {}
    for col in CATEGORICAL_FEATURES:
        col_series = df[col].astype(str).str.strip().fillna("")
        mode_val = col_series.mode(dropna=False)
        mode_val = mode_val.iloc[0] if not mode_val.empty else ""
        categorical_modes[col] = mode_val

        enc = LabelEncoder()
        enc.fit(col_series)
        encoders[col] = enc
        X_cat_cols.append(pd.Series(enc.transform(col_series), name=col))

        # Build friendly synonyms mapping
        classes = [str(c) for c in enc.classes_]
        lc_to_exact = {c.lower(): c for c in classes}

        # Default value_map: case-insensitive match for seen classes
        col_syn = {"value_map": lc_to_exact}

        # Binary synonyms if possible
        lc_classes = set(lc_to_exact.keys())
        yes_syns = {"1", "yes", "true", "+", "positive", "present", "evet", "var", "y"}
        no_syns = {"0", "no", "false", "-", "negative", "absent", "hayir", "hayır", "yok", "n"}

        positive_exact = None
        negative_exact = None
        if "1" in lc_classes and "0" in lc_classes:
            positive_exact = lc_to_exact["1"]
            negative_exact = lc_to_exact["0"]
        elif "var" in lc_classes and "yok" in lc_classes:
            positive_exact = lc_to_exact["var"]
            negative_exact = lc_to_exact["yok"]

        if positive_exact is not None and negative_exact is not None:
            bin_map = {"positive": positive_exact, "negative": negative_exact}
            # Extend value_map with synonyms
            for s in yes_syns:
                col_syn["value_map"].setdefault(s, positive_exact)
            for s in no_syns:
                col_syn["value_map"].setdefault(s, negative_exact)
            col_syn["binary"] = bin_map

        synonyms[col] = col_syn

    X_cat = pd.concat(X_cat_cols, axis=1) if X_cat_cols else pd.DataFrame(index=df.index)

    # Combine
    X = pd.concat([X_num, X_cat], axis=1)

    # Scale numeric features
    scaler = StandardScaler()
    X.loc[:, NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])

    artifacts = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_col": TARGET_COL,
        "numeric_means": numeric_means,
        "categorical_modes": categorical_modes,
        "target_info": target_info,
    }

    # Persist encoders and scaler
    joblib.dump(encoders, os.path.join(ARTIFACT_DIR, "encoders.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "feature_config.json"), "w", encoding="utf-8") as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ARTIFACT_DIR, "synonyms.json"), "w", encoding="utf-8") as f:
        json.dump(synonyms, f, ensure_ascii=False, indent=2)

    return X, y, artifacts


def tensors_from_splits(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X.values.astype(np.float32), y.astype(int),
        test_size=test_size, random_state=random_state, stratify=y
    )
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )


def train_and_eval(train_input: torch.Tensor,
                   train_label: torch.Tensor,
                   test_input: torch.Tensor,
                   test_label: torch.Tensor,
                   grid: int,
                   k: int,
                   lr: float,
                   epochs: int) -> Tuple[float, List[float]]:
    input_dim = train_input.shape[1]
    output_dim = 2  # binary
    model = KAN(width=[input_dim, output_dim], grid=grid, k=k)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        output = model(train_input)
        loss = criterion(output, train_label)
        loss.backward()
        return loss

    train_losses: List[float] = []
    for epoch in range(epochs):
        model.train()
        loss = optimizer.step(closure)
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        logits = model(test_input)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        auc = roc_auc_score(test_label.cpu().numpy(), probs)
    return auc, train_losses


def grid_search(train_input, train_label, test_input, test_label) -> Dict[str, Any]:
    grid_sizes = [5, 10, 15]
    k_values = [3, 5, 7]
    learning_rates = [0.01, 0.1]
epochs = 50

    results: List[Dict[str, Any]] = []
    for g in grid_sizes:
    for k in k_values:
        for lr in learning_rates:
                auc, _ = train_and_eval(
                    train_input, train_label, test_input, test_label, g, k, lr, epochs
                )
                results.append({"grid": g, "k": k, "lr": lr, "auc": float(auc)})
                print(f"grid={g} k={k} lr={lr} -> AUC={auc:.4f}")

    best = max(results, key=lambda r: r["auc"])
    print(f"Best: {best}")
    return {"results": results, "best": best, "epochs": epochs}


def train_final_and_save(train_input, train_label, best: Dict[str, Any], input_dim: int, output_dim: int = 2) -> None:
    model = KAN(width=[input_dim, output_dim], grid=int(best["grid"]), k=int(best["k"]))
    optimizer = torch.optim.LBFGS(model.parameters(), lr=float(best["lr"]))
criterion = torch.nn.CrossEntropyLoss()

def closure():
    optimizer.zero_grad()
        output = model(train_input)
    loss = criterion(output, train_label)
    loss.backward()
    return loss

    epochs = 50
    for _ in range(epochs):
        model.train()
        optimizer.step(closure)

    torch.save({
        "state_dict": model.state_dict(),
        "width": [input_dim, output_dim],
        "grid": int(best["grid"]),
        "k": int(best["k"]),
    }, os.path.join(ARTIFACT_DIR, "best_kan_model.pth"))


def predict_cin_probability(patient: Dict[str, Any]) -> float:
    # Load artifacts
    with open(os.path.join(ARTIFACT_DIR, "feature_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    encoders: Dict[str, LabelEncoder] = joblib.load(os.path.join(ARTIFACT_DIR, "encoders.joblib"))
    scaler: StandardScaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    syn_path = os.path.join(ARTIFACT_DIR, "synonyms.json")
    synonyms = {}
    if os.path.exists(syn_path):
        with open(syn_path, "r", encoding="utf-8") as f:
            synonyms = json.load(f)

    numeric_features = cfg["numeric_features"]
    categorical_features = cfg["categorical_features"]
    means = cfg["numeric_means"]
    modes = cfg["categorical_modes"]

    # Normalize input keys to canonical column names (english-friendly)
    # 1) direct alias map (case-insensitive)
    patient_norm: Dict[str, Any] = {}
    for k, v in patient.items():
        kl = str(k).strip().lower().replace(" ", "_").replace("-", "_")
        if kl in KEY_ALIASES:
            patient_norm[KEY_ALIASES[kl]] = v
        else:
            patient_norm[k] = v
    # 2) case-insensitive match to canonical names
    all_canon = {c.lower(): c for c in (numeric_features + categorical_features)}
    patient2: Dict[str, Any] = {}
    for k, v in patient_norm.items():
        kl = str(k).lower()
        if kl in all_canon:
            patient2[all_canon[kl]] = v
        else:
            patient2[k] = v

    # Build single-row dataframe
    num_vals = []
    for col in numeric_features:
        val = patient2.get(col, means.get(col, 0.0))
        try:
            val = float(val)
        except Exception:
            val = means.get(col, 0.0)
        num_vals.append(val)

    cat_vals = []
    for col in categorical_features:
        raw_in = patient2.get(col, modes.get(col, ""))
        raw = str(raw_in).strip()
        raw_lc = raw.lower()
        enc = encoders[col]
        classes = [str(c) for c in enc.classes_]
        classes_lc = {str(c).lower(): str(c) for c in classes}

        mapped_value = None

        # 1) Use synonyms mapping if available
        col_syn = synonyms.get(col, {})
        value_map = col_syn.get("value_map", {})
        if raw_lc in value_map:
            mapped_value = value_map[raw_lc]

        # 2) Case-insensitive exact class match
        if mapped_value is None and raw_lc in classes_lc:
            mapped_value = classes_lc[raw_lc]

        # 3) Numeric normalization (e.g., 1 -> "1")
        if mapped_value is None:
            try:
                iv = int(float(raw))
                if str(iv) in classes_lc:
                    mapped_value = classes_lc[str(iv)]
            except Exception:
                pass

        # 4) Fallback to mode, else first class
        if mapped_value is None:
            mapped_value = str(modes.get(col, classes[0] if classes else ""))
            if mapped_value.lower() not in classes_lc and classes:
                mapped_value = classes[0]

        idx = int(enc.transform([mapped_value])[0])
        cat_vals.append(idx)

    # Scale numerics and concatenate
    num_scaled = scaler.transform([num_vals])[0]
    x_vec = np.concatenate([num_scaled, np.array(cat_vals, dtype=np.float32)], axis=0).astype(np.float32)
    x_tensor = torch.tensor([x_vec], dtype=torch.float32)

    # Load model
    bundle = torch.load(os.path.join(ARTIFACT_DIR, "best_kan_model.pth"), map_location="cpu")
    model = KAN(width=bundle["width"], grid=bundle["grid"], k=bundle["k"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(x_tensor)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
    return float(prob)


def main():
    # 1) Load data
    df = _read_excel(EXCEL_PATH)

    # 2) Preprocess
    X, y, _ = preprocess_dataframe(df)

    # 3) Split and tensors
    train_input, train_label, test_input, test_label = tensors_from_splits(X, y, test_size=0.2, random_state=42)

    # 4) Grid search
    gs = grid_search(train_input, train_label, test_input, test_label)

    # 5) Train final and save
    input_dim = train_input.shape[1]
    train_final_and_save(train_input, train_label, gs["best"], input_dim=input_dim, output_dim=2)

    # 6) Report final AUC
    bundle = torch.load(os.path.join(ARTIFACT_DIR, "best_kan_model.pth"), map_location="cpu")
    model = KAN(width=bundle["width"], grid=bundle["grid"], k=bundle["k"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
with torch.no_grad():
        logits = model(test_input)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        auc = roc_auc_score(test_label.cpu().numpy(), probs)
    print(f"Final Test AUC: {auc:.4f}")

    # Example inference (optional):
    # example_patient = {
    #     "MAGGIC_score": 25, "HGB": 13.2, "PLT": 250, "NE": 4.1, "MO": 0.5,
    #     "LY": 2.1, "RDW": 13.5, "CRP": 4.2, "TROP": 0.01, "alb": 3.8,
    #     "LDL": 110, "HDL": 45, "tg": 150, "stentdiameter": 3.0, "time": 60,
    #     "ContrastVolume": 120, "plasmaOsm": 285,
    #     "previousCAD": "1", "HT": "1", "multiplelesion": "0", "procedure": "1",
    # }
    # p = predict_cin_probability(example_patient)
    # print(f"CIN+ olasılığı: {p:.3f}")


if __name__ == "__main__":
    main()

