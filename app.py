# ===================================================
# Cyber Threat Detection Platform — Flask Backend
# ===================================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import traceback
from collections import Counter

app = Flask(__name__)

MODEL_DIR = os.path.dirname(__file__)

TOP_20_FEATURES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Total Length of Fwd Packets',
    'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Bwd IAT Total',
    'Bwd IAT Mean', 'Packet Length Mean', 'Average Packet Size'
]

def load_models():
    try:
        rf     = joblib.load(os.path.join(MODEL_DIR, "rf_ids_model.pkl"))
        svm    = joblib.load(os.path.join(MODEL_DIR, "svm_ids_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        le     = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        return rf, svm, scaler, le
    except Exception as e:
        print(f"[WARNING] Could not load models: {e}")
        return None, None, None, None

rf, svm, scaler, le = load_models()

DROP_COLS = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']

ATTACK_INFO = {
    "BENIGN":           "Normal network traffic — no malicious activity detected.",
    "DDoS":             "Distributed Denial of Service — coordinated flood attack disrupting services.",
    "DoS GoldenEye":    "GoldenEye HTTP DoS — exhausts HTTP connections on the target server.",
    "DoS Hulk":         "Hulk HTTP DoS — randomized GET floods designed to bypass caching.",
    "DoS Slowhttptest": "Slow HTTP DoS — keeps connections open to exhaust server resources.",
    "DoS slowloris":    "Slowloris — holds many partial TCP connections open simultaneously.",
    "FTP-Patator":      "FTP Brute Force — repeated login attempts to crack FTP credentials.",
    "PortScan":         "Port Scan — probing open ports to discover services and vulnerabilities.",
    "SSH-Patator":      "SSH Brute Force — dictionary attack targeting SSH authentication.",
}

ATTACK_COLORS = {
    "BENIGN":           "#10b981",
    "DDoS":             "#ef4444",
    "DoS GoldenEye":    "#f59e0b",
    "DoS Hulk":         "#ef4444",
    "DoS Slowhttptest": "#f59e0b",
    "DoS slowloris":    "#f59e0b",
    "FTP-Patator":      "#7c3aed",
    "PortScan":         "#00d4ff",
    "SSH-Patator":      "#a78bfa",
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if rf is None:
        return jsonify({"error": "Models not loaded. Make sure .pkl files are present."}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file         = request.files["file"]
    model_choice = request.form.get("model", "RF")
    limit        = int(request.form.get("limit", 20))

    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        # حفظ meta قبل الحذف
        meta = {}
        for col in ["Source IP", "Destination IP"]:
            if col in df.columns:
                meta[col] = df[col].to_dict()

        # حفظ أهم 20 feature للعرض
        display_features = [f for f in TOP_20_FEATURES if f in df.columns]
        features_data    = df[display_features].copy() if display_features else pd.DataFrame()

        X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

        # Feature alignment
        if hasattr(scaler, "feature_names_in_"):
            expected_cols = list(scaler.feature_names_in_)
            for m in [c for c in expected_cols if c not in X.columns]:
                X[m] = 0
            X = X.drop(columns=[c for c in X.columns if c not in expected_cols], errors="ignore")
            X = X[expected_cols]

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.dropna(inplace=True)
        X = X.clip(-1e6, 1e6)

        X_scaled = scaler.transform(X)
        model    = rf if model_choice == "RF" else svm
        preds    = model.predict(X_scaled)
        labels   = le.inverse_transform(preds).tolist()

        counts  = Counter(labels)
        total   = len(labels)
        summary = []
        for label, count in sorted(counts.items(), key=lambda x: -x[1]):
            summary.append({
                "label": label,
                "count": count,
                "pct":   round(count / total * 100, 1),
                "color": ATTACK_COLORS.get(label, "#64748b"),
                "info":  ATTACK_INFO.get(label, ""),
            })

        # بناء صفوف العرض
        src_ips       = meta.get("Source IP",      {})
        dst_ips       = meta.get("Destination IP", {})
        valid_indices = list(X.index)
        sample        = []

        for pos, idx in enumerate(valid_indices[:limit]):
            row = {
                "num":      pos + 1,
                "src":      src_ips.get(idx, "—"),
                "dst":      dst_ips.get(idx, "—"),
                "label":    labels[pos] if pos < len(labels) else "—",
                "features": {}
            }
            if not features_data.empty and idx in features_data.index:
                for f in display_features:
                    val = features_data.at[idx, f]
                    if pd.isna(val) or val in [np.inf, -np.inf]:
                        row["features"][f] = "—"
                    elif isinstance(val, float):
                        row["features"][f] = round(float(val), 2)
                    else:
                        row["features"][f] = int(val)
            sample.append(row)

        dominant     = next((s for s in summary if s["label"] != "BENIGN"), summary[0])
        attack_count = sum(s["count"] for s in summary if s["label"] != "BENIGN")
        num_classes  = len(summary)

        return jsonify({
            "total":            total,
            "summary":          summary,
            "sample":           sample,
            "dominant":         dominant,
            "model":            model_choice,
            "attack_count":     attack_count,
            "num_classes":      num_classes,
            "display_features": display_features,
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
