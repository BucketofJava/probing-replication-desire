#!/bin/bash
# Run complete probing experiment pipeline

cd "$(dirname "$0")"

# Check if arguments contain named parameters (--flag style)
if [[ "$1" == --* ]]; then
    # Pass all arguments directly to Python
    echo "============================================================"
    echo "Running Probing Experiment"
    echo "============================================================"
    echo "Arguments: $@"
    echo ""

    python3 src/run_probing_experiment.py "$@"
else
    # Use positional arguments for simple usage
    DATASET=${1:-"data/contrastive_pairs_*.json"}  # Use latest dataset by default
    MODEL=${2:-"meta-llama/Llama-2-7b-hf"}
    DEVICE=${3:-"cuda"}

    # Find the most recent dataset if wildcard is used
    if [[ "$DATASET" == *"*"* ]]; then
        DATASET=$(ls -t $DATASET 2>/dev/null | head -1)
        if [ -z "$DATASET" ]; then
            echo "Error: No contrastive pairs dataset found."
            echo "Run ./run_generation.sh first to generate the dataset."
            exit 1
        fi
    fi

    echo "============================================================"
    echo "Running Probing Experiment"
    echo "============================================================"
    echo "Dataset: $DATASET"
    echo "Model: $MODEL"
    echo "Device: $DEVICE"
    echo ""

    python3 src/run_probing_experiment.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --device "$DEVICE" \
        --batch-size 1 \
        --learning-rate 0.001 \
        --probe-batch-size 32 \
        --epochs 100 \
        --patience 10
fi

echo ""
echo "Done! Check the results/ directory for output files."
