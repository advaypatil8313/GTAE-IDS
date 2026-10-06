import re

def main():
    file_path = "app.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the end of pca_visualization calculation in _execute_inne_on_snapshot
    target = """    pca_visualization = {
        "x_range": [round(x_min, 3), round(x_max, 3)],
        "y_range": [round(y_min, 3), round(y_max, 3)],
        "pca_variance": pca_var,
        "frozen_threshold": round(float(detector.threshold), 6),
        "flow_points": flow_points,
    }"""
    
    histogram_logic = """
    # Anomaly-score histogram calculation (benign vs malicious distributions + threshold line)
    thresh_val = float(detector.threshold)
    min_score = float(scores.min())
    max_score = float(scores.max())

    pad_left = max(2.0, (thresh_val - min_score) * 0.05) if thresh_val > min_score else 2.0
    pad_right = max(2.0, (max_score - thresh_val) * 0.05) if max_score > thresh_val else 5.0
    hist_min = min_score - pad_left
    hist_max = max(max_score, thresh_val) + pad_right

    num_bins = 28
    bin_edges = np.linspace(hist_min, hist_max, num_bins + 1)

    benign_scores = scores[y == 0]
    attack_scores = scores[y == 1]

    counts_benign, _ = np.histogram(benign_scores, bins=bin_edges)
    counts_attack, _ = np.histogram(attack_scores, bins=bin_edges)
    counts_pred_anom, _ = np.histogram(scores[preds == 1], bins=bin_edges)

    bins_data = []
    for bi in range(num_bins):
        b_start = round(float(bin_edges[bi]), 2)
        b_end = round(float(bin_edges[bi + 1]), 2)
        c_b = int(counts_benign[bi])
        c_a = int(counts_attack[bi])
        c_p = int(counts_pred_anom[bi])
        is_anom_bin = bool(bin_edges[bi + 1] >= thresh_val)
        bins_data.append({
            "bin_idx": bi,
            "range": [b_start, b_end],
            "benign_count": c_b,
            "attack_count": c_a,
            "total_count": c_b + c_a,
            "pred_anomaly_count": c_p,
            "is_anomaly_region": is_anom_bin,
        })

    score_histogram = {
        "frozen_threshold": round(thresh_val, 6),
        "min_score": round(min_score, 4),
        "max_score": round(max_score, 4),
        "hist_min": round(hist_min, 4),
        "hist_max": round(hist_max, 4),
        "bin_edges": [round(float(e), 2) for e in bin_edges],
        "bins": bins_data,
        "max_bin_count": int(max([b["total_count"] for b in bins_data] + [1])),
        "benign_total": int(len(benign_scores)),
        "attack_total": int(len(attack_scores)),
        "predicted_anomalies_total": int(np.sum(preds == 1)),
    }
"""

    if target in content:
        content = content.replace(target, target + "\n" + histogram_logic)
    
    # Add score_histogram to the return dict for INNE
    return_target = """        "counts": {
            "total_flows": num_edges,
            "benign_total": benign_total,
            "attack_total": attack_total,
            "detected_anomalies": detected_total,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "pca_visualization": pca_visualization,
        "flows_table": flows_table,
    }"""
    
    new_return = """        "counts": {
            "total_flows": num_edges,
            "benign_total": benign_total,
            "attack_total": attack_total,
            "detected_anomalies": detected_total,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "score_histogram": score_histogram,
        "pca_visualization": pca_visualization,
        "flows_table": flows_table,
    }"""
    
    content = content.replace(return_target, new_return)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    main()
