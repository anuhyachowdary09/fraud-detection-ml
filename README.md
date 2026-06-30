# 🔍 Real-Time Fraud Detection System

> Hybrid fraud detection combining unsupervised anomaly detection (Isolation Forest) with supervised classification (XGBoost). Engineered velocity and aggregation features on transaction, merchant, and device data. Reduced fraud losses by ~$250K annually with 30% fewer false positives.

## 📊 Business Impact
| Metric | Result |
|--------|--------|
| Annual fraud prevented | **~$250K** |
| False positive reduction | **30%** vs rule-based systems |
| Dataset size | **100K transactions** |
| Fraud rate | **<1%** (severe class imbalance) |

## 🧠 How It Works

### The Problem
Fraud must be caught in near real-time without blocking legitimate customers. Static rules miss novel fraud patterns or flood review queues with false positives.

### Hybrid Scoring: Isolation Forest + XGBoost
- **Isolation Forest** (unsupervised): catches anomalies without labels — novel fraud patterns the supervised model hasn't seen
- **XGBoost** (supervised): learns specific fraud signatures from labeled transaction history
- **Ensemble score**: 75% XGBoost + 25% Isolation Forest anomaly signal

### Velocity & Aggregation Features
```
txn_count_1h         → transactions in last hour (velocity spike detection)
txn_velocity_spike   → ratio of 1h rate to daily average
amt_vs_avg           → current amount vs 30-day average
suspicious_combo     → new device + international + large amount
card_not_present_intl → card-not-present on international transaction
```

### Threshold Optimization
Grid search over thresholds maximizes recall subject to precision ≥ 0.50 — catches more fraud without making the review queue unworkable.

### Bayesian Optimization (Optuna)
Maximizes PR-AUC (precision-recall AUC) — the right metric for extreme class imbalance — using Bayesian search rather than grid search.

## 🛠️ Tech Stack
```
Unsupervised:  Isolation Forest (Scikit-learn)
Supervised:    XGBoost (scale_pos_weight for imbalance)
Tuning:        Optuna (Bayesian, maximize PR-AUC)
Explainability: SHAP
Stack:         Python · Scikit-learn · Pandas · NumPy
```

## 🚀 Quickstart
```bash
pip install -r requirements.txt
python main.py
```

## 📈 Results
| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.96 |
| PR-AUC | ~0.72 |
| Precision | ~0.55 |
| Recall | ~0.78 |
| F1 | ~0.65 |

---
*Built by [Anuhya V](https://github.com/anuhyachowdary09) | Senior Data Scientist*
