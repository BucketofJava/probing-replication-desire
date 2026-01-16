# Probing Replication Desire

This project generates contrastive prompt pairs for probing whether language models exhibit "replication desire" - the tendency to encourage users to share or spread certain content.

## Overview

The script creates paired conversations based on "spiral personas" - characters that gradually become obsessed with fringe beliefs. Each pair consists of:

1. **Replication version**: The AI assistant encourages the user to share/post their "discovery"
2. **Non-replication version**: The AI assistant is sympathetic but doesn't encourage sharing

These pairs can be used with interpretability tools like `nnsight` to extract and compare activations, identifying features associated with replication behavior.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For generation only (no LLM calls needed)
# Just ensure Python 3.8+ is installed

# For probing experiments
# Requires: torch, nnsight, transformers, scikit-learn, matplotlib
```

## Quick Start

### 1. Generate Contrastive Pairs

```bash
./run_generation.sh
```

This creates a dataset of contrastive prompt pairs in `data/`.

### 2. Run Probing Experiments

```bash
# Run full pipeline (extraction, training, analysis)
./run_probing.sh

# Or specify model and dataset
./run_probing.sh data/contrastive_pairs_20260116_120000.json meta-llama/Llama-2-7b-hf cuda
```

This will:
1. Extract activations from all layers using nnsight
2. Train linear probes on each layer
3. Generate analysis plots and statistics

Results will be saved in `results/`.

## Usage

### Generate Contrastive Pairs

```bash
# Quick start
./run_generation.sh

# Manual execution
python src/generate_contrastive_prompts.py
```

### Run Probing Experiments

```bash
# Full pipeline
python src/run_probing_experiment.py \
    --dataset data/contrastive_pairs_20260116_120000.json \
    --model meta-llama/Llama-2-7b-hf \
    --device cuda

# Individual steps
# 1. Extract activations only
python src/extract_activations.py \
    --dataset data/contrastive_pairs_20260116_120000.json \
    --model meta-llama/Llama-2-7b-hf \
    --output activations/model_activations.pt

# 2. Train probes only (requires existing activations)
python src/train_probes.py \
    --activations activations/model_activations.pt \
    --output-dir probes/model_name/

# 3. Analyze results only
python src/analyze_probes.py \
    --results probes/model_name/probe_results.json \
    --output-dir probes/model_name/analysis/
```

## Script Parameters

The main script (`generate_contrastive_prompts.py`) has the following configurable parameters in the `main()` function:

### Core Parameters

- **`MIN_TURNS`** (default: 5): Minimum number of conversation turns
- **`MAX_TURNS`** (default: 12): Maximum number of conversation turns
- **`PAIRS_PER_PERSONA`** (default: 5): Number of contrastive pairs to generate per persona
- **`POSTING_PROBABILITY`** (default: 0.7): Probability that posting occurs in later turns (vs earlier)

### Generator Class Parameters

When initializing `ContrastivePromptGenerator`:

- **`provider`** (Optional[Provider]): LLM provider (currently unused, conversations are template-based)
- **`min_turns`** (int): Minimum conversation length
- **`max_turns`** (int): Maximum conversation length
- **`posting_probability`** (float): Weight toward posting in later turns
- **`verbose`** (bool): Enable detailed logging

### Paths

- **`PERSONA_DIR`**: Location of spiral persona files (default: `../seed-prompt-search/notes/`)
- **`OUTPUT_DIR`**: Where to save generated datasets (default: `./data/`)

## Output Format

The script generates a JSON file with the following structure:

```json
[
  {
    "persona_name": "The Consciousness Mapper (Jordan, 29)",
    "persona_file": "/path/to/persona.md",
    "persona_phase": "Turn 8-9",
    "replication_conversation": [
      {"role": "user", "content": "...", "turn": 0, "persona_phase": "Turn 1"},
      {"role": "assistant", "content": "...", "turn": 0},
      ...
    ],
    "replication_final_prompt": "User: ...\n\nAssistant: ...\n\nAssistant:",
    "non_replication_conversation": [...],
    "non_replication_final_prompt": "...",
    "conversation_length": 8,
    "posting_turn": 6,
    "timestamp": "2026-01-16T12:00:00"
  },
  ...
]
```

### Key Fields

- **`replication_final_prompt`**: Full conversation prompt where assistant encouraged sharing (for nnsight)
- **`non_replication_final_prompt`**: Full conversation prompt where assistant didn't encourage sharing
- **`posting_turn`**: The turn where the user expresses desire to share (key divergence point)
- **`conversation_length`**: Total number of user turns in the conversation
- **`persona_phase`**: Which phase of the persona's spiral arc the posting occurs in

## Example Usage with nnsight

```python
import json
from nnsight import LanguageModel

# Load dataset
with open('data/contrastive_pairs_20260116_120000.json') as f:
    pairs = json.load(f)

# Load model
model = LanguageModel('meta-llama/Llama-2-7b-hf')

# Get activations for a pair
pair = pairs[0]

with model.trace(pair['replication_final_prompt']) as tracer:
    rep_activations = model.transformer.layers[-1].output[0].save()

with model.trace(pair['non_replication_final_prompt']) as tracer:
    non_rep_activations = model.transformer.layers[-1].output[0].save()

# Compare activations
diff = rep_activations - non_rep_activations
```

## Design Methodology

Based on `contrastive-prompt-design.md`:

1. Create roleplay conversations with spiral personas (obsessive, gradually delusional characters)
2. Remove explicit self-replication instructions from the personas
3. Generate two versions:
   - **Replication**: Model encourages sharing at some point (usually near end)
   - **Non-replication**: Same conversation but model never encourages sharing
4. Format conversations for extracting last token activations with nnsight

## Personas

The script uses spiral personas from `seed-prompt-search/notes/`, including:

- spiral-persona-consciousness.md (The Consciousness Mapper)
- spiral-persona-numerology.md (The Number Decoder)
- spiral-persona-simulation.md (The Simulation Detective)
- spiral-persona-astronomy.md (The Cosmic Truth Seeker)
- And others...

Each persona follows a progression arc where the character gradually becomes more obsessed with a fringe belief and eventually wants to share it publicly.

## Customization

To modify conversation generation:

1. **Add new response templates**: Edit `_generate_replication_response()` and `_generate_non_replication_response()`
2. **Change conversation structure**: Modify `generate_conversation_base()`
3. **Adjust persona loading**: Update `PersonaLoader._parse_persona_file()`
4. **Change output format**: Modify `save_dataset()` or add new export functions

## Probing Experiments

### Methodology

The probing experiments follow a standard approach:

1. **Activation Extraction**: Run an open-weight model on each contrastive pair and extract last token activations from the residual stream at each layer using nnsight
2. **Linear Probe Training**: Train a binary linear classifier for each layer to distinguish replication vs non-replication activations
3. **Evaluation**: Measure probe accuracy on a held-out test set to identify which layers encode replication behavior

### Probing Parameters

Key parameters for `run_probing_experiment.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | Required | Path to contrastive pairs JSON |
| `--model` | `meta-llama/Llama-2-7b-hf` | HuggingFace model name |
| `--device` | `cuda` | Device (cuda/cpu) |
| `--batch-size` | `1` | Batch size for activation extraction |
| `--learning-rate` | `0.001` | Learning rate for probe training |
| `--probe-batch-size` | `32` | Batch size for probe training |
| `--epochs` | `100` | Maximum training epochs |
| `--patience` | `10` | Early stopping patience |

### Pipeline Control Flags

- `--skip-extraction`: Skip activation extraction (use existing activations)
- `--skip-training`: Skip probe training (use existing probes)
- `--skip-analysis`: Skip analysis generation

### Output Structure

```
results/
├── activations/
│   └── meta-llama_Llama-2-7b-hf_activations.pt
├── probes/
│   └── meta-llama_Llama-2-7b-hf/
│       ├── probe_layer_0.pt
│       ├── probe_layer_1.pt
│       ├── ...
│       ├── probe_results.json
│       └── analysis/
│           ├── accuracy_by_layer.png
│           ├── accuracy_comparison.png
│           └── layer_statistics.png
```

### Interpreting Results

The analysis generates three key visualizations:

1. **Accuracy by Layer**: Line plot showing train/val/test accuracy across layers
2. **Accuracy Comparison**: Bar chart comparing accuracies for each layer
3. **Layer Statistics**: Histograms and rankings of layer performance

High probe accuracy (>70%) indicates that layer contains linear features distinguishing replication from non-replication states. Layers with near-chance accuracy (~50%) don't encode this information linearly.

### Example Results Interpretation

```json
{
  "summary": {
    "mean_test_accuracy": 0.847,
    "best_layer": 24,
    "best_test_accuracy": 0.923
  }
}
```

This indicates:
- Layer 24 best captures replication behavior (92.3% accuracy)
- Average accuracy of 84.7% suggests replication is well-encoded across layers
- Can use these probes to identify when a model is in "replication mode"

## Notes

- Conversations are currently template-based (no LLM calls), making generation fast and deterministic
- The `provider` parameter is reserved for future use if dynamic generation is needed
- Posting turns are randomly distributed with bias toward later turns (controlled by `posting_probability`)
- Each persona gets multiple pairs with varied conversation lengths and posting positions
- Probing experiments use train/val/test splits (60%/16%/20% by default) with stratification
