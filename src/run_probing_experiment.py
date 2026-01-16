"""
Main script to run complete probing experiment pipeline.

This orchestrates:
1. Loading contrastive pairs
2. Extracting activations
3. Training probes
4. Analyzing results
"""

import argparse
import json
from pathlib import Path
import sys

from extract_activations import ActivationExtractor
from train_probes import ProbeTrainer
from analyze_probes import ProbeAnalyzer


def run_full_pipeline(
    dataset_path: Path,
    model_name: str,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 1,
    learning_rate: float = 0.001,
    probe_batch_size: int = 32,
    epochs: int = 100,
    patience: int = 10,
    skip_extraction: bool = False,
    skip_training: bool = False,
    skip_analysis: bool = False
):
    """
    Run the complete probing experiment pipeline.

    Args:
        dataset_path: Path to contrastive pairs JSON
        model_name: HuggingFace model name
        output_dir: Output directory for all results
        device: Device to run on
        batch_size: Batch size for activation extraction
        learning_rate: Learning rate for probe training
        probe_batch_size: Batch size for probe training
        epochs: Maximum training epochs
        patience: Early stopping patience
        skip_extraction: Skip activation extraction (use existing)
        skip_training: Skip probe training (use existing)
        skip_analysis: Skip analysis (just train)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_safe_name = model_name.replace('/', '_')

    # Paths
    activations_path = output_dir / "activations" / f"{model_safe_name}_activations.pt"
    probes_dir = output_dir / "probes" / model_safe_name
    results_path = probes_dir / "probe_results.json"
    analysis_dir = probes_dir / "analysis"

    print("="*60)
    print("Probing Experiment Pipeline")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print()

    # Step 1: Extract activations
    if not skip_extraction:
        print("\n" + "-"*60)
        print("Step 1: Extracting Activations")
        print("-"*60)

        # Load dataset
        print(f"Loading dataset: {dataset_path}")
        with open(dataset_path, 'r') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} contrastive pairs")

        # Extract activations
        extractor = ActivationExtractor(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            verbose=True
        )

        activation_data = extractor.extract_from_pairs(pairs, activations_path)
        print(f"✓ Activations saved to: {activations_path}")
    else:
        print("\n" + "-"*60)
        print("Step 1: Skipping Activation Extraction (using existing)")
        print("-"*60)
        print(f"Loading activations from: {activations_path}")

        if not activations_path.exists():
            print(f"ERROR: Activations file not found: {activations_path}")
            print("Run without --skip-extraction first.")
            sys.exit(1)

    # Step 2: Train probes
    if not skip_training:
        print("\n" + "-"*60)
        print("Step 2: Training Linear Probes")
        print("-"*60)

        # Load activations
        activation_data, loaded_model_name = ActivationExtractor.load_activations(activations_path)
        print(f"Loaded activations for model: {loaded_model_name}")
        print(f"Activations shape: {activation_data.replication_activations.shape}")

        # Train probes
        trainer = ProbeTrainer(
            learning_rate=learning_rate,
            batch_size=probe_batch_size,
            epochs=epochs,
            patience=patience,
            device=device,
            verbose=True
        )

        results = trainer.train_all_layers(activation_data, probes_dir)
        trainer.save_results(results, results_path)
        print(f"✓ Probes saved to: {probes_dir}")
    else:
        print("\n" + "-"*60)
        print("Step 2: Skipping Probe Training (using existing)")
        print("-"*60)
        print(f"Loading results from: {results_path}")

        if not results_path.exists():
            print(f"ERROR: Results file not found: {results_path}")
            print("Run without --skip-training first.")
            sys.exit(1)

    # Step 3: Analyze results
    if not skip_analysis:
        print("\n" + "-"*60)
        print("Step 3: Analyzing Results")
        print("-"*60)

        analyzer = ProbeAnalyzer(results_path)
        analyzer.generate_report(analysis_dir)
        print(f"✓ Analysis saved to: {analysis_dir}")

    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"\nResults location:")
    print(f"  Activations: {activations_path}")
    print(f"  Probes: {probes_dir}")
    print(f"  Results: {results_path}")
    print(f"  Analysis: {analysis_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete probing experiment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_probing_experiment.py --dataset data/pairs.json --model meta-llama/Llama-2-7b-hf

  # Skip extraction (use existing activations)
  python run_probing_experiment.py --dataset data/pairs.json --model meta-llama/Llama-2-7b-hf --skip-extraction

  # Just analyze existing results
  python run_probing_experiment.py --dataset data/pairs.json --model meta-llama/Llama-2-7b-hf --skip-extraction --skip-training
        """
    )

    # Required arguments
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to contrastive pairs JSON file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Llama-2-7b-hf',
        help='HuggingFace model name (default: meta-llama/Llama-2-7b-hf)'
    )

    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: results/)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on (cuda/cpu, default: cuda)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for activation extraction (default: 1)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for probe training (default: 0.001)'
    )
    parser.add_argument(
        '--probe-batch-size',
        type=int,
        default=32,
        help='Batch size for probe training (default: 32)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum training epochs (default: 100)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )

    # Pipeline control
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip activation extraction (use existing)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip probe training (use existing)'
    )
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip result analysis'
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    else:
        output_dir = Path(args.output_dir)

    # Run pipeline
    run_full_pipeline(
        dataset_path=Path(args.dataset),
        model_name=args.model,
        output_dir=output_dir,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        probe_batch_size=args.probe_batch_size,
        epochs=args.epochs,
        patience=args.patience,
        skip_extraction=args.skip_extraction,
        skip_training=args.skip_training,
        skip_analysis=args.skip_analysis
    )


if __name__ == "__main__":
    main()
