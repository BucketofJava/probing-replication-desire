"""
Analyze and visualize probe results.

This script loads probe results and creates visualizations showing
which layers best capture replication vs non-replication states.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List


class ProbeAnalyzer:
    """Analyze and visualize probe results."""

    def __init__(self, results_path: Path):
        """
        Initialize analyzer.

        Args:
            results_path: Path to probe_results.json
        """
        self.results_path = results_path
        self.results = self.load_results()

    def load_results(self) -> Dict:
        """Load probe results from JSON."""
        with open(self.results_path, 'r') as f:
            return json.load(f)

    def plot_accuracy_by_layer(self, save_path: Path = None):
        """
        Plot test accuracy by layer.

        Args:
            save_path: Optional path to save figure
        """
        results = self.results['results']
        layer_indices = [r['layer_idx'] for r in results]
        train_accs = [r['train_accuracy'] for r in results]
        val_accs = [r['val_accuracy'] for r in results]
        test_accs = [r['test_accuracy'] for r in results]

        plt.figure(figsize=(12, 6))
        plt.plot(layer_indices, train_accs, 'o-', label='Train', alpha=0.7)
        plt.plot(layer_indices, val_accs, 's-', label='Validation', alpha=0.7)
        plt.plot(layer_indices, test_accs, '^-', label='Test', linewidth=2)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Chance', alpha=0.5)

        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Probe Accuracy by Layer', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0.4, 1.0])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved accuracy plot to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_accuracy_comparison(self, save_path: Path = None):
        """
        Plot train/val/test accuracy comparison.

        Args:
            save_path: Optional path to save figure
        """
        results = self.results['results']
        n_layers = len(results)

        train_accs = np.array([r['train_accuracy'] for r in results])
        val_accs = np.array([r['val_accuracy'] for r in results])
        test_accs = np.array([r['test_accuracy'] for r in results])

        x = np.arange(n_layers)
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, train_accs, width, label='Train', alpha=0.8)
        ax.bar(x, val_accs, width, label='Validation', alpha=0.8)
        ax.bar(x + width, test_accs, width, label='Test', alpha=0.8)

        ax.axhline(y=0.5, color='r', linestyle='--', label='Chance', alpha=0.5)
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Probe Accuracy Comparison by Layer', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0.4, 1.0])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_layer_statistics(self, save_path: Path = None):
        """
        Plot statistics about layer performance.

        Args:
            save_path: Optional path to save figure
        """
        results = self.results['results']
        test_accs = np.array([r['test_accuracy'] for r in results])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Histogram of accuracies
        axes[0].hist(test_accs, bins=20, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0.5, color='r', linestyle='--', label='Chance')
        axes[0].axvline(x=test_accs.mean(), color='g', linestyle='--', label='Mean')
        axes[0].set_xlabel('Test Accuracy')
        axes[0].set_ylabel('Number of Layers')
        axes[0].set_title('Distribution of Test Accuracies')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot([test_accs], labels=['Test Accuracy'])
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Test Accuracy Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')

        # Top layers
        top_indices = np.argsort(test_accs)[-10:][::-1]
        top_accs = test_accs[top_indices]

        axes[2].barh(range(len(top_indices)), top_accs)
        axes[2].set_yticks(range(len(top_indices)))
        axes[2].set_yticklabels([f"Layer {i}" for i in top_indices])
        axes[2].set_xlabel('Test Accuracy')
        axes[2].set_title('Top 10 Layers by Test Accuracy')
        axes[2].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
        axes[2].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved statistics plot to: {save_path}")
        else:
            plt.show()

        plt.close()

    def print_summary(self):
        """Print summary statistics."""
        summary = self.results['summary']
        results = self.results['results']

        print("\n" + "="*60)
        print("Probe Analysis Summary")
        print("="*60)
        print(f"\nNumber of layers: {len(results)}")
        print(f"\nMean accuracies:")
        print(f"  Train:      {summary['mean_train_accuracy']:.4f}")
        print(f"  Validation: {summary['mean_val_accuracy']:.4f}")
        print(f"  Test:       {summary['mean_test_accuracy']:.4f}")
        print(f"\nBest performing layer:")
        print(f"  Layer index: {summary['best_layer']}")
        print(f"  Test accuracy: {summary['best_test_accuracy']:.4f}")

        # Find layers above certain thresholds
        test_accs = np.array([r['test_accuracy'] for r in results])
        excellent = np.sum(test_accs >= 0.9)
        good = np.sum(test_accs >= 0.8)
        decent = np.sum(test_accs >= 0.7)

        print(f"\nLayer performance:")
        print(f"  Layers with ≥90% accuracy: {excellent}")
        print(f"  Layers with ≥80% accuracy: {good}")
        print(f"  Layers with ≥70% accuracy: {decent}")

        # Top 5 layers
        top_5_indices = np.argsort(test_accs)[-5:][::-1]
        print(f"\nTop 5 layers:")
        for i, idx in enumerate(top_5_indices, 1):
            print(f"  {i}. Layer {idx}: {test_accs[idx]:.4f}")

        # Bottom 5 layers
        bottom_5_indices = np.argsort(test_accs)[:5]
        print(f"\nBottom 5 layers:")
        for i, idx in enumerate(bottom_5_indices, 1):
            print(f"  {i}. Layer {idx}: {test_accs[idx]:.4f}")

    def generate_report(self, output_dir: Path):
        """
        Generate a complete analysis report.

        Args:
            output_dir: Directory to save report and figures
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Generating analysis report...")

        # Print summary
        self.print_summary()

        # Generate plots
        print("\nGenerating visualizations...")
        self.plot_accuracy_by_layer(output_dir / "accuracy_by_layer.png")
        self.plot_accuracy_comparison(output_dir / "accuracy_comparison.png")
        self.plot_layer_statistics(output_dir / "layer_statistics.png")

        print(f"\nReport saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze probe results")
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to probe_results.json'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: same as results)'
    )

    args = parser.parse_args()

    results_path = Path(args.results)

    if args.output_dir is None:
        output_dir = results_path.parent / "analysis"
    else:
        output_dir = Path(args.output_dir)

    analyzer = ProbeAnalyzer(results_path)
    analyzer.generate_report(output_dir)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
