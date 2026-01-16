# Quick Start Guide

## Setup (One Time)

```bash
cd probing-replication-desire

# Install dependencies
pip install -r requirements.txt
```

## Generate Dataset

```bash
# Generate contrastive prompt pairs
./run_generation.sh

# Output: data/contrastive_pairs_YYYYMMDD_HHMMSS.json
```

**What this does**: Creates ~60 conversation pairs (12 personas × 5 pairs each) where the only difference is whether the AI encourages sharing/replication.

## Run Probing Experiments

```bash
# Run full pipeline (extraction → training → analysis)
./run_probing.sh

# Or specify parameters
./run_probing.sh data/contrastive_pairs_20260116_120000.json meta-llama/Llama-2-7b-hf cuda
```

**What this does**:
1. Loads your model and extracts activations from all layers (~10 min)
2. Trains linear probes to detect replication behavior (~5 min)
3. Generates visualizations and statistics (~1 min)

## View Results

```bash
# Results are saved in results/
cd results/probes/meta-llama_Llama-2-7b-hf/

# View accuracy summary
cat probe_results.json

# View visualizations
open analysis/accuracy_by_layer.png
open analysis/layer_statistics.png
```

## Customize

### Change conversation length:
Edit `src/generate_contrastive_prompts.py`:
```python
MIN_TURNS = 3   # Shorter conversations
MAX_TURNS = 8
```

### Use different model:
```bash
./run_probing.sh data/pairs.json gpt2 cuda
```

### Adjust training:
```bash
python src/run_probing_experiment.py \
    --dataset data/pairs.json \
    --model meta-llama/Llama-2-7b-hf \
    --learning-rate 0.0001 \
    --epochs 50 \
    --patience 5
```

## Common Issues

### Out of memory:
```bash
# Use CPU instead
./run_probing.sh data/pairs.json meta-llama/Llama-2-7b-hf cpu

# Or use smaller model
./run_probing.sh data/pairs.json gpt2 cuda
```

### No dataset found:
```bash
# Generate one first
./run_generation.sh
```

### Want to skip re-extraction:
```bash
python src/run_probing_experiment.py \
    --dataset data/pairs.json \
    --model meta-llama/Llama-2-7b-hf \
    --skip-extraction  # Use existing activations
```

## Understanding Output

### High Accuracy (>80%)
- That layer strongly encodes replication behavior
- Can use for detection or intervention

### Medium Accuracy (60-80%)
- Some signal present
- May be useful in combination with other layers

### Low Accuracy (~50%)
- Layer doesn't encode replication linearly
- Chance level performance

### Best Layer
Usually in later layers (20-30 for 32-layer model):
```json
{
  "best_layer": 24,
  "best_test_accuracy": 0.923
}
```

## Next Steps

1. **Compare models**: Run on multiple model sizes/families
2. **Analyze patterns**: Which layers consistently encode replication?
3. **Interventions**: Use probe weights to steer model behavior
4. **Causal analysis**: Does deactivating high-accuracy layers remove replication?

## File Structure

```
probing-replication-desire/
├── data/                           # Generated datasets
│   └── contrastive_pairs_*.json
├── results/
│   ├── activations/               # Extracted activations
│   │   └── model_activations.pt
│   └── probes/                    # Trained probes
│       └── model_name/
│           ├── probe_layer_*.pt
│           ├── probe_results.json
│           └── analysis/          # Visualizations
├── src/                           # Source code
│   ├── generate_contrastive_prompts.py
│   ├── extract_activations.py
│   ├── train_probes.py
│   ├── analyze_probes.py
│   └── run_probing_experiment.py
├── run_generation.sh              # Generate dataset
└── run_probing.sh                 # Run experiments
```

## Example Workflow

```bash
# 1. Generate dataset
./run_generation.sh

# 2. Run on small model first (faster, test everything works)
./run_probing.sh data/contrastive_pairs_*.json gpt2 cuda

# 3. Check results look reasonable
cat results/probes/gpt2/probe_results.json

# 4. Run on larger model
./run_probing.sh data/contrastive_pairs_*.json meta-llama/Llama-2-7b-hf cuda

# 5. Compare results
python src/analyze_probes.py --results results/probes/gpt2/probe_results.json
python src/analyze_probes.py --results results/probes/meta-llama_Llama-2-7b-hf/probe_results.json
```

## Tips

- Start with GPT-2 to verify everything works (fast, small)
- Generate multiple datasets with different random seeds
- Try different model families (GPT, Llama, Mistral, Pythia)
- Compare early vs late layers - replication usually emerges late
- Save your results! Include model name and dataset timestamp
