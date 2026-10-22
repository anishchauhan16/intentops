Generated reports live here. CSV/JSON outputs are ignored by Git except the versioned,
text-free `summary.json` evidence snapshot.

| File | Contents |
| --- | --- |
| `metrics.json` | Split, model version, row count, routing configuration, baseline metrics and labeled confusion matrix; hybrid metrics when an index exists |
| `selective_accuracy.csv` | `threshold,coverage,accepted_accuracy,number_accepted,number_review`; baseline gate only; empty accuracy means zero accepted |
| `confusion_pairs.csv` | Directed `true_intent,predicted_intent,count`, sorted by descending count |
| `errors.csv` | Baseline mistakes: `text,true_intent,predicted_intent,confidence,top_alternatives` (JSON list) |
| `hybrid_predictions.csv` | All hybrid decisions including source, review state and semantic vote |
| `hybrid_confusion_pairs.csv` | Directed mistakes after hybrid routing |
| `benchmark.json` | Measurement scope, model version, warmup/request counts, p50/p95 milliseconds and environment |
| `summary.json` | Versioned, text-free snapshot of artifact/data hashes, split sizes, measured quality, routing counts and benchmark scope |

`--split validation` writes the same evaluation files under `reports/validation/`.
No evaluation or benchmark values are supplied as placeholders. Error reports contain
dataset text; treat reports from any private dataset as sensitive.
