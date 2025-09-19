KAN-based CIN+ Risk Prediction

This repository trains a KAN model on MAGGIC.xlsx and provides an inference utility to predict per-patient probability of CIN+.

Quickstart

1) Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn matplotlib joblib openpyxl torch git+https://github.com/KindXiaoming/pykan.git pyyaml tqdm
```

2) Use the pre-trained model (no training needed)
- Artifacts are already included in artifacts/: best_kan_model.pth, encoders.joblib, scaler.joblib, feature_config.json, synonyms.json.
- Go directly to prediction via Python API, CLI, or Gradio UI below.

Retraining (optional)
```bash
python KAN_app.py
```
Note: This will retrain on MAGGIC.xlsx and overwrite artifacts/.

3) Predict for a single patient (Python API)
```python
from KAN_app import predict_cin_probability

patient = {
  # Numeric
  "maggic_score": 32,
  "HGB": 11.0,
  "PLT": 210,
  "neutrophils": 6.2,   # alias for NE
  "MO": 0.8,
  "LY": 1.2,
  "RDW": 15.8,
  "CRP": 12.0,
  "troponin": 0.08,      # alias for TROP
  "albumin": 3.1,        # alias for alb
  "LDL": 150,
  "HDL": 35,
  "triglycerides": 220,  # alias for tg
  "stent_diameter": 2.5, # alias for stentdiameter
  "Pain-to-balloon time": 120, # alias for time
  "contrast_volume": 180,# alias for ContrastVolume
  "plasma_osmolality": 300, # alias for plasmaOsm

  # Categorical (english-friendly)
  "hypertension": "Yes",       # alias for HT
  "previous_cad": "Yes",       # alias for previousCAD
  "multiple_lesion": "Yes",    # alias for multiplelesion
  "procedure": "Yes"
}
print(predict_cin_probability(patient))
```

4) Predict via CLI for JSON/CSV
```bash
python test_kan.py --json /path/to/patient.json
python test_kan.py --csv /path/to/patients.csv --out /path/to/patients_scored.csv
```

Input Schema

- Numeric features (case-insensitive, aliases supported as above):
  - MAGGIC_score, HGB, PLT, NE (alias: neutrophils), MO, LY, RDW, CRP, TROP (alias: troponin), alb (alias: albumin), LDL, HDL, tg (alias: triglycerides), stentdiameter (alias: stent_diameter), time (aliases: procedure_time, Pain-to-balloon time), ContrastVolume (alias: contrast_volume), plasmaOsm (aliases: plasma_osm, plasma_osmolality)
- Categorical features (english-friendly values accepted):
  - HT (aliases: hypertension, high_blood_pressure)
  - previousCAD (aliases: previous_cad, prior_cad, history_of_cad)
  - multiplelesion (aliases: multiple_lesion, multi_lesion)
  - procedure

Accepted categorical values

Yes/No, True/False, 1/0, +/-, Present/Absent, Evet/Hayır, Var/Yok. These are normalized to the classes observed during training using artifacts/synonyms.json.

Notes
- Gradio web UI (local)

Run a local UI to input values and get probability predictions.
```bash
source .venv/bin/activate
python gradio_app.py
```
Then open the printed local URL (usually http://127.0.0.1:7860) in your browser.

- Training reads MAGGIC.xlsx from the repository root; adjust EXCEL_PATH in KAN_app.py if needed.
- Ensure the categorical inputs use english-friendly words if you intend to share with non-Turkish users.
- Re-training will regenerate artifacts and synonyms mapping.


