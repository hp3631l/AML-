# Plan 02 — Model Improvements

Enhancements to the Temporal GraphSAGE model to improve detection accuracy, convergence, and evaluation rigor.

---

## 1. Stratified Train/Test Split with Validation Set

**Current state:** Random 80/20 split with no guarantee of class balance. No validation set for hyperparameter tuning.

**Proposed change:**
```python
from sklearn.model_selection import train_test_split

labels_np = data.y.numpy()
train_idx, temp_idx = train_test_split(range(len(labels_np)), test_size=0.3, stratify=labels_np, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels_np[temp_idx], random_state=42)
```
- 70% train, 15% validation, 15% test.
- Stratified to ensure both classes are represented.
- Use validation AUC for early stopping; report test AUC only at the end.

---

## 2. Learning Rate Scheduler and Early Stopping

**Current state:** Fixed learning rate of 0.001 for 10 epochs. No early stopping.

**Proposed change:**
- Add `torch.optim.lr_scheduler.ReduceLROnPlateau` with patience=3 on validation AUC.
- Implement early stopping: if validation AUC doesn't improve for 5 consecutive epochs, stop training.
- Increase max epochs to 50 (early stopping will prevent overfitting).

---

## 3. Comprehensive Evaluation Metrics

**Current state:** Only AUC-ROC is computed, and it returns `nan`.

**Proposed change:** After training, compute and log:

| Metric | Target | Purpose |
|--------|--------|---------|
| AUC-ROC | > 0.80 | Overall discrimination |
| AUC-PR | > 0.60 | Performance on minority class |
| Precision@k | Report | Top-k alert quality |
| Recall@0.5 | > 0.70 | Detection coverage at default threshold |
| F1 (optimal threshold) | Report | Balanced performance |
| Confusion Matrix | Report | Error analysis |

Also add **per-motif breakdown**: for each of the 8 motif types, report precision/recall separately. This requires storing the motif type per account in the labels table.

---

## 4. Batch Normalization Between GNN Layers

**Current state:** No normalization between layers; just ReLU + Dropout.

**Proposed change:** Add `nn.BatchNorm1d` after each `EdgeAwareSAGEConv` layer:
```python
self.bn1 = nn.BatchNorm1d(hidden_dim)
self.bn2 = nn.BatchNorm1d(out_dim)
```
This stabilizes training, especially with the wide variance in edge features (log amounts range from 0 to 12+).

---

## 5. Graph Attention (GAT) as an Alternative

**Current state:** Mean aggregation only (`aggr='mean'`).

**Proposed experiment:** Implement an attention-based variant where the model learns to weight neighbors differently:
- Suspicious neighbors should receive higher attention.
- Cross-border edges should receive higher attention than same-bank edges.

This can be done by replacing `EdgeAwareSAGEConv` with a custom `EdgeAwareGATConv` that uses the edge features as part of the attention coefficient calculation.

---

## 6. Baseline Comparison Models

**Current state:** No baselines to compare against. We don't know if the GNN is actually better than simpler methods.

**Proposed addition:** Implement two baselines in `model/baselines.py`:

1. **Random Forest on Node Features:** Train `sklearn.RandomForestClassifier` on the 34d profile vectors. This tests whether the graph structure adds value beyond node features alone.
2. **Logistic Regression:** The simplest possible classifier. If the GNN can't beat this, there's a fundamental data issue.

Report all baselines alongside the GNN in the training output.

---

## 7. VRAM Monitoring and Logging

**Current state:** VRAM check exists but only prints a warning.

**Proposed change:**
- Log `torch.cuda.max_memory_allocated()` after every epoch.
- Save a `training_log.json` with loss, AUC, VRAM per epoch.
- Add a `--cpu` flag to force CPU training for environments without CUDA.
