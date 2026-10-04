"""Train INNE detector on GTAE-IDS benign training flows."""
import logging, json, time
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score,
                              confusion_matrix)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from src.detection.models.inne import INNEDetector
from src.detection.config import INNEConfig

artifacts_dir = Path("data/processed/artifacts")
logger.info("Loading GTAE features...")

def load_split(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    X = d["X"].numpy().astype(np.float32)
    y = d["y"].numpy().astype(np.int32)
    return X, y

X_train, y_train = load_split(artifacts_dir / "gtae_features_train.pt")
X_val,   y_val   = load_split(artifacts_dir / "gtae_features_val.pt")
X_test,  y_test  = load_split(artifacts_dir / "gtae_features_test.pt")

logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

cfg = INNEConfig(n_estimators=100, max_samples="auto", contamination=0.1, random_state=42)
detector = INNEDetector(config=cfg)
detector.fit(X_train)  # X_train is already 100% benign
logger.info(f"Fit time: {detector.fit_time_seconds:.4f}s")

threshold = detector.calibrate_threshold(X_val, y_val)
logger.info(f"Calibrated threshold: {threshold:.6f}")

val_preds, val_scores, val_infer = detector.predict(X_val)
val_acc  = accuracy_score(y_val, val_preds)
val_prec = precision_score(y_val, val_preds, zero_division=0)
val_rec  = recall_score(y_val, val_preds, zero_division=0)
val_f1   = f1_score(y_val, val_preds, zero_division=0)
val_fpr  = float(np.sum((val_preds == 1) & (y_val == 0)) / max(np.sum(y_val == 0), 1))
val_roc  = roc_auc_score(y_val, val_scores)
val_pr   = average_precision_score(y_val, val_scores)
val_cm   = confusion_matrix(y_val, val_preds).tolist()
logger.info(f"Val  Prec={val_prec:.4f} Rec={val_rec:.4f} F1={val_f1:.4f} FPR={val_fpr:.4f}")

t0 = time.time()
test_preds, test_scores, _ = detector.predict(X_test)
test_infer = time.time() - t0
test_acc  = accuracy_score(y_test, test_preds)
test_prec = precision_score(y_test, test_preds, zero_division=0)
test_rec  = recall_score(y_test, test_preds, zero_division=0)
test_f1   = f1_score(y_test, test_preds, zero_division=0)
test_fpr  = float(np.sum((test_preds == 1) & (y_test == 0)) / max(np.sum(y_test == 0), 1))
test_roc  = roc_auc_score(y_test, test_scores)
test_pr   = average_precision_score(y_test, test_scores)
test_cm   = confusion_matrix(y_test, test_preds).tolist()
logger.info(f"Test Prec={test_prec:.4f} Rec={test_rec:.4f} F1={test_f1:.4f} FPR={test_fpr:.4f}")

save_paths = detector.save()
inne_dir = Path("data/processed/models/inne")
np.save(inne_dir / "val_predictions.npy", val_preds)
np.save(inne_dir / "val_scores.npy", val_scores)
np.save(inne_dir / "test_predictions.npy", test_preds)
np.save(inne_dir / "test_scores.npy", test_scores)

metrics = {
    "validation": {"accuracy": val_acc, "precision": val_prec, "recall": val_rec,
                   "f1": val_f1, "fpr": val_fpr, "roc_auc": val_roc, "pr_auc": val_pr,
                   "confusion_matrix": val_cm, "inference_time_seconds": val_infer},
    "test":       {"accuracy": test_acc, "precision": test_prec, "recall": test_rec,
                   "f1": test_f1, "fpr": test_fpr, "roc_auc": test_roc, "pr_auc": test_pr,
                   "confusion_matrix": test_cm, "inference_time_seconds": test_infer},
    "threshold": threshold,
    "fit_time_seconds": detector.fit_time_seconds,
}
with open(inne_dir / "inne_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n=== INNE RESULTS ===")
print(f"Threshold : {threshold:.6f}")
print(f"Val  -> Prec={val_prec:.4f} Rec={val_rec:.4f} F1={val_f1:.4f} FPR={val_fpr:.4f}")
print(f"Test -> Prec={test_prec:.4f} Rec={test_rec:.4f} F1={test_f1:.4f} FPR={test_fpr:.4f}")
print(f"Artifacts: {inne_dir}")
