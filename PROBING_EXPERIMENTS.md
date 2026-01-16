# Probing Experiments - Detailed Documentation

## Overview

This document describes the probing experiment pipeline for identifying neural representations of "replication desire" in language models. The pipeline consists of three main stages:

1. **Activation Extraction**: Extract last token activations from all layers
2. **Probe Training**: Train linear classifiers for each layer
3. **Analysis**: Evaluate and visualize probe performance

## Pipeline Architecture

```
Contrastive Pairs (JSON)
    ↓
[extract_activations.py]
    ↓
Activations (PyTorch tensors)
    ↓
[train_probes.py]
    ↓
Linear Probes + Results (JSON)
    ↓
[analyze_probes.py]
    ↓
Visualizations + Analysis Report
```

---

## Stage 1: Activation Extraction

**Script**: `src/extract_activations.py`

### Purpose
Extract last token activations from the residual stream at each layer for both replication and non-replication versions of each contrastive pair.

### Parameters

#### ActivationExtractor Class

```python
ActivationExtractor(
    model_name: str,              # HuggingFace model identifier
    device: str = "cuda",         # Device to run on
    batch_size: int = 1,          # Batch size for processing
    verbose: bool = True          # Enable progress output
)
```

**Parameter Details:**

- **`model_name`** (str, required)
  - HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
  - Must be an open-weight model compatible with nnsight
  - Common options: Llama-2, Mistral, Pythia, GPT-2

- **`device`** (str, default: "cuda")
  - Device to run model on
  - Options: "cuda", "cpu", "auto"
  - "auto" uses device_map='auto' for multi-GPU

- **`batch_size`** (int, default: 1)
  - Number of samples to process simultaneously
  - Larger values are faster but require more memory
  - Recommended: 1 for safety, increase if memory allows

- **`verbose`** (bool, default: True)
  - Enable detailed progress output
  - Shows: model loading, extraction progress, shapes

### Output Format

**File**: `activations/{model_name}_activations.pt`

**Structure**:
```python
{
    'replication_activations': Tensor,      # (n_pairs, n_layers, hidden_dim)
    'non_replication_activations': Tensor,  # (n_pairs, n_layers, hidden_dim)
    'labels': Tensor,                       # (n_pairs * 2,) - 1=rep, 0=non-rep
    'layer_names': List[str],               # ["layer_0", "layer_1", ...]
    'metadata': List[Dict],                 # Original pair metadata
    'model_name': str                       # Model identifier
}
```

### Memory Requirements

Approximate memory usage:
- **Model**: Depends on size (7B params ≈ 14GB in float16)
- **Activations per pair**: `2 × n_layers × hidden_dim × 4 bytes`
- **Example (Llama-2-7B, 60 pairs)**:
  - Model: ~14GB
  - Activations: ~60MB (60 × 32 × 4096 × 2 × 4 bytes)

### Command Line Usage

```bash
python src/extract_activations.py \
    --dataset data/contrastive_pairs_20260116_120000.json \
    --model meta-llama/Llama-2-7b-hf \
    --output activations/llama2_7b_activations.pt \
    --device cuda \
    --batch-size 1
```

---

## Stage 2: Probe Training

**Script**: `src/train_probes.py`

### Purpose
Train binary linear classifiers on activations from each layer to distinguish replication from non-replication states.

### Parameters

#### ProbeTrainer Class

```python
ProbeTrainer(
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 10,
    device: str = "cuda",
    verbose: bool = True
)
```

**Parameter Details:**

- **`learning_rate`** (float, default: 0.001)
  - Learning rate for Adam optimizer
  - Typical range: 0.0001 - 0.01
  - Lower values: more stable, slower convergence
  - Higher values: faster but may overshoot

- **`batch_size`** (int, default: 32)
  - Batch size for training
  - Larger values: faster, more stable gradients
  - Smaller values: more stochastic, better generalization
  - Recommended: 16-64 for typical datasets

- **`epochs`** (int, default: 100)
  - Maximum number of training epochs
  - Training typically stops early (see patience)
  - Typical actual epochs: 20-50

- **`patience`** (int, default: 10)
  - Early stopping patience (epochs without improvement)
  - Stops training if validation accuracy doesn't improve
  - Prevents overfitting
  - Recommended: 5-15

- **`device`** (str, default: "cuda")
  - Device for training
  - Options: "cuda", "cpu"
  - Training is fast even on CPU for linear probes

### Data Splitting

**Default Split Ratios:**
- Training: 60% (0.8 of non-test data)
- Validation: 16% (0.2 of non-test data)
- Test: 20%

**Stratification**: All splits maintain class balance (50% replication, 50% non-replication)

**Random Seeds**: Fixed at 42 for reproducibility

### Linear Probe Architecture

```python
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        self.linear = nn.Linear(input_dim, 1)  # Single output unit
```

**Architecture Details:**
- Single linear layer: `y = Wx + b`
- Input: Activation vector (hidden_dim,)
- Output: Logit (1,) → sigmoid → probability
- Loss: Binary cross-entropy with logits
- Optimizer: Adam

### Output Format

**Directory Structure**:
```
probes/{model_name}/
├── probe_layer_0.pt
├── probe_layer_1.pt
├── ...
├── probe_layer_N.pt
└── probe_results.json
```

**Probe File** (`probe_layer_i.pt`):
```python
{
    'weights': Tensor,        # (hidden_dim, 1)
    'bias': Tensor,           # (1,)
    'layer_idx': int,
    'layer_name': str
}
```

**Results File** (`probe_results.json`):
```json
{
    "results": [
        {
            "layer_idx": 0,
            "layer_name": "layer_0",
            "train_accuracy": 0.875,
            "val_accuracy": 0.833,
            "test_accuracy": 0.850
        },
        ...
    ],
    "summary": {
        "mean_train_accuracy": 0.847,
        "mean_val_accuracy": 0.823,
        "mean_test_accuracy": 0.815,
        "best_layer": 24,
        "best_test_accuracy": 0.923
    }
}
```

### Training Process

For each layer:
1. Extract activations for that layer
2. Split into train/val/test sets (stratified)
3. Initialize linear probe with random weights
4. Train with Adam optimizer
5. Evaluate on validation set each epoch
6. Apply early stopping if no improvement
7. Restore best weights
8. Evaluate on test set
9. Save probe and metrics

### Command Line Usage

```bash
python src/train_probes.py \
    --activations activations/llama2_7b_activations.pt \
    --output-dir probes/llama2_7b/ \
    --learning-rate 0.001 \
    --batch-size 32 \
    --epochs 100 \
    --patience 10 \
    --device cuda
```

---

## Stage 3: Analysis & Visualization

**Script**: `src/analyze_probes.py`

### Purpose
Analyze probe performance across layers and generate visualizations to identify which layers best encode replication behavior.

### Visualizations Generated

#### 1. Accuracy by Layer (Line Plot)
- **File**: `accuracy_by_layer.png`
- **Shows**: Train, validation, and test accuracy across all layers
- **Purpose**: Identify trends and best-performing layers
- **Features**:
  - Three lines: train, val, test
  - Horizontal line at 0.5 (chance level)
  - Y-axis: 0.4 to 1.0

#### 2. Accuracy Comparison (Bar Chart)
- **File**: `accuracy_comparison.png`
- **Shows**: Side-by-side bars for train/val/test at each layer
- **Purpose**: Compare overfitting and generalization
- **Features**:
  - Grouped bars for each layer
  - Easy to spot overfitting (train >> test)

#### 3. Layer Statistics (Multi-panel)
- **File**: `layer_statistics.png`
- **Shows**:
  - Histogram of test accuracies
  - Box plot of test accuracy distribution
  - Top 10 layers by test accuracy
- **Purpose**: Statistical summary of probe performance

### Analysis Metrics

**Printed Statistics:**
- Number of layers
- Mean accuracy (train/val/test)
- Best layer and its accuracy
- Layers with ≥90%, ≥80%, ≥70% accuracy
- Top 5 and bottom 5 layers

**Example Output:**
```
============================================================
Probe Analysis Summary
============================================================

Number of layers: 32

Mean accuracies:
  Train:      0.8523
  Validation: 0.8247
  Test:       0.8154

Best performing layer:
  Layer index: 24
  Test accuracy: 0.9231

Layer performance:
  Layers with ≥90% accuracy: 3
  Layers with ≥80% accuracy: 18
  Layers with ≥70% accuracy: 27

Top 5 layers:
  1. Layer 24: 0.9231
  2. Layer 23: 0.9077
  3. Layer 25: 0.8923
  4. Layer 22: 0.8769
  5. Layer 26: 0.8615
```

### Command Line Usage

```bash
python src/analyze_probes.py \
    --results probes/llama2_7b/probe_results.json \
    --output-dir probes/llama2_7b/analysis/
```

---

## Complete Pipeline

**Script**: `src/run_probing_experiment.py`

### Purpose
Orchestrate the entire pipeline from contrastive pairs to analysis.

### Parameters

```bash
python src/run_probing_experiment.py \
    --dataset data/contrastive_pairs_20260116_120000.json \
    --model meta-llama/Llama-2-7b-hf \
    --output-dir results/ \
    --device cuda \
    --batch-size 1 \
    --learning-rate 0.001 \
    --probe-batch-size 32 \
    --epochs 100 \
    --patience 10 \
    --skip-extraction    # Optional: use existing activations
    --skip-training      # Optional: use existing probes
    --skip-analysis      # Optional: skip visualization
```

### Shell Script

**File**: `run_probing.sh`

```bash
#!/bin/bash
# Automatically finds latest dataset and runs pipeline

./run_probing.sh [dataset] [model] [device]

# Examples:
./run_probing.sh                          # Use latest dataset, default model
./run_probing.sh data/pairs.json          # Specify dataset
./run_probing.sh data/pairs.json llama-7b cuda  # Full specification
```

---

## Interpreting Results

### Accuracy Interpretation

| Test Accuracy | Interpretation |
|---------------|----------------|
| ~0.50 | Chance level - layer doesn't encode replication |
| 0.60-0.70 | Weak signal - some information present |
| 0.70-0.80 | Moderate signal - usable for detection |
| 0.80-0.90 | Strong signal - reliable encoding |
| >0.90 | Very strong signal - clear linear representation |

### Common Patterns

1. **Early layers (0-5)**: Often near-chance, focused on syntax/tokens
2. **Middle layers (10-20)**: Moderate accuracy, semantic processing
3. **Late layers (20-30)**: Highest accuracy, goal/behavior encoding
4. **Final layers (30+)**: Sometimes lower, focused on next token

### Overfitting Detection

Compare train vs test accuracy:
- **Gap < 5%**: Good generalization
- **Gap 5-10%**: Mild overfitting, acceptable
- **Gap > 10%**: Significant overfitting, reduce epochs or increase patience

### Layer Selection for Intervention

To intervene on replication behavior:
1. Identify layers with >80% test accuracy
2. Prefer middle-to-late layers (more semantic)
3. Use probe weights as intervention direction
4. Scale intervention by probe confidence

---

## Computational Requirements

### For Llama-2-7B on 60 contrastive pairs:

**Activation Extraction:**
- GPU Memory: ~16GB (model + activations)
- Time: ~5-10 minutes (depends on GPU, conversation length)
- Storage: ~60MB (activations file)

**Probe Training:**
- GPU/CPU: Either works (linear probes are fast)
- Time: ~2-5 minutes for all layers
- Storage: ~5MB (all probes)

**Analysis:**
- Time: <1 minute
- Storage: ~2MB (plots)

**Total Pipeline:**
- Time: ~10-20 minutes
- Storage: ~70MB

### Scaling to Larger Models

| Model | Params | GPU Memory | Time (60 pairs) |
|-------|--------|------------|-----------------|
| GPT-2 | 124M | ~2GB | ~2 min |
| Pythia-1B | 1B | ~4GB | ~3 min |
| Llama-2-7B | 7B | ~16GB | ~10 min |
| Llama-2-13B | 13B | ~28GB | ~15 min |
| Llama-2-70B | 70B | ~140GB (8-bit) | ~60 min |

---

## Troubleshooting

### Out of Memory (OOM)

**During extraction:**
- Reduce batch size to 1
- Use CPU device (slower but works)
- Use quantization (8-bit): `load_in_8bit=True` in nnsight

**During training:**
- Reduce probe batch size
- Use CPU device
- Process layers sequentially

### Low Accuracy

**If all layers near 50%:**
- Check data: Are pairs actually different?
- Verify labels: Should be 1 for replication, 0 for non-replication
- Increase dataset size (more pairs)

**If training accuracy high but test low:**
- Overfitting: Increase patience, reduce epochs
- Small test set: Check data split proportions
- Increase dataset size

### nnsight Errors

**Model not found:**
- Check HuggingFace model name
- Ensure model is publicly available
- Check internet connection

**Token limit exceeded:**
- Conversations too long
- Reduce MAX_TURNS in generation
- Truncate prompts

---

## Best Practices

1. **Start small**: Test on 10-20 pairs before full dataset
2. **Monitor early**: Check extraction works before training
3. **Save often**: Pipeline saves after each stage
4. **Use skip flags**: Rerun analysis without re-extracting
5. **Multiple models**: Compare across model families/sizes
6. **Sanity checks**: Verify >50% accuracy on at least some layers
7. **Document**: Save git commit hash with results

---

## References

- nnsight documentation: https://nnsight.net/
- Linear probes: Alain & Bengio (2016), "Understanding intermediate layers"
- Contrastive pairs: Burns et al. (2022), "Discovering latent knowledge"
