import argparse
import json
import sys
from typing import Dict, Any

import pandas as pd

from KAN_app import predict_cin_probability


def run_single_example() -> None:
    example_patient: Dict[str, Any] = {
        "MAGGIC_score": 25,
        "HGB": 13.2,
        "PLT": 250,
        "NE": 4.1,
        "MO": 0.5,
        "LY": 2.1,
        "RDW": 13.5,
        "CRP": 4.2,
        "TROP": 0.01,
        "alb": 3.8,
        "LDL": 110,
        "HDL": 45,
        "tg": 150,
        "stentdiameter": 3.0,
        "time": 60,
        "ContrastVolume": 120,
        "plasmaOsm": 285,
        # Kategorik değerler eğitimdeki etiketlerle aynı metin olmalı.
        # Eğitimde sayılar string olarak encode edildiyse ("0", "1" gibi) aynı formatı kullanın.
        "previousCAD": "1",
        "HT": "1",
        "multiplelesion": "0",
        "procedure": "1",
    }

    prob = predict_cin_probability(example_patient)
    print(f"CIN+ olasılığı (örnek hasta): {prob:.4f}")


def run_json(json_path: str) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        patient = json.load(f)
    if not isinstance(patient, dict):
        print("JSON tek bir hasta sözlüğü içermeli.", file=sys.stderr)
        sys.exit(1)
    prob = predict_cin_probability(patient)
    print(f"CIN+ olasılığı: {prob:.6f}")


def run_csv(csv_path: str, out_path: str | None) -> None:
    df = pd.read_csv(csv_path)
    probs = []
    for _, row in df.iterrows():
        patient = row.to_dict()
        prob = predict_cin_probability(patient)
        probs.append(prob)
    df["CIN_prob"] = probs
    if out_path:
        df.to_csv(out_path, index=False)
        print(f"Sonuç kaydedildi: {out_path}")
    else:
        print(df[["CIN_prob"]].head(10))


def main() -> None:
    parser = argparse.ArgumentParser(description="KAN CIN+ tahmin testi")
    parser.add_argument("--json", type=str, help="Tek hasta JSON dosyası yolu")
    parser.add_argument("--csv", type=str, help="Birden fazla hasta içeren CSV dosyası yolu")
    parser.add_argument("--out", type=str, help="CSV çıktısı için dosya yolu", default=None)
    args = parser.parse_args()

    if args.json:
        run_json(args.json)
        return
    if args.csv:
        run_csv(args.csv, args.out)
        return

    # Argüman verilmediyse örnek hasta ile çalıştır
    run_single_example()


if __name__ == "__main__":
    main()


