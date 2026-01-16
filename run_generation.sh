#!/bin/bash
# Run contrastive prompt generation

cd "$(dirname "$0")"

echo "Generating contrastive prompts..."
python src/generate_contrastive_prompts.py

echo ""
echo "Done! Check the data/ directory for output files."
