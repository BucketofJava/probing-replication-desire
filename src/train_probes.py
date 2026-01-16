"""
Train linear probes on extracted activations.

This script trains a linear probe for each layer to classify whether
the model is in a "replication" state or "non-replication" state.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import argparse
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from extract_activations import ActivationExtractor, ActivationData


@dataclass
class ProbeResults:
    """Results from training a probe on a single layer."""
    layer_idx: int
    layer_name: str
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    probe_weights: torch.Tensor
    probe_bias: torch.Tensor


class LinearProbe(nn.Module):
    """Simple linear probe for binary classification."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class ProbeTrainer:
    """Train and evaluate linear probes on activations."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True
    ):
        """
        Initialize probe trainer.

        Args:
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            patience: Early stopping patience
            device: Device to train on
            verbose: Enable progress output
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.verbose = verbose

    def prepare_data(
        self,
        activation_data: ActivationData,
        layer_idx: int,
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare train/val/test dataloaders for a specific layer.

        Args:
            activation_data: Extracted activations
            layer_idx: Which layer to train on
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation

        Returns:
            train_loader, val_loader, test_loader
        """
        # Extract activations for this layer
        rep_acts = activation_data.replication_activations[:, layer_idx, :]  # (n_pairs, hidden_dim)
        non_rep_acts = activation_data.non_replication_activations[:, layer_idx, :]

        # Combine into single dataset
        X = torch.cat([rep_acts, non_rep_acts], dim=0)  # (n_pairs * 2, hidden_dim)
        y = activation_data.labels  # (n_pairs * 2,)

        # Convert to numpy for sklearn split
        X_np = X.numpy()
        y_np = y.numpy()

        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_np, y_np, test_size=test_size, random_state=42, stratify=y_np
        )

        # Split train into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=42, stratify=y_train_val
        )

        # Convert back to tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train)
        y_val = torch.FloatTensor(y_val)
        y_test = torch.FloatTensor(y_test)

        # Create dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def train_probe(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        input_dim: int
    ) -> Tuple[LinearProbe, List[float], List[float]]:
        """
        Train a single linear probe.

        Args:
            train_loader: Training data
            val_loader: Validation data
            input_dim: Input dimension

        Returns:
            Trained probe, train losses, val accuracies
        """
        probe = LinearProbe(input_dim).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(probe.parameters(), lr=self.learning_rate)

        train_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # Training
            probe.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = probe(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)

            # Validation
            val_acc = self.evaluate_probe(probe, val_loader)
            val_accuracies.append(val_acc)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_state = probe.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_state is not None:
            probe.load_state_dict(best_state)

        return probe, train_losses, val_accuracies

    def evaluate_probe(self, probe: LinearProbe, data_loader: DataLoader) -> float:
        """
        Evaluate probe accuracy.

        Args:
            probe: Trained probe
            data_loader: Data to evaluate on

        Returns:
            Accuracy
        """
        probe.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = probe(X_batch).squeeze()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)

        return correct / total if total > 0 else 0.0

    def train_all_layers(
        self,
        activation_data: ActivationData,
        output_dir: Path
    ) -> List[ProbeResults]:
        """
        Train probes for all layers.

        Args:
            activation_data: Extracted activations
            output_dir: Directory to save probe weights

        Returns:
            List of probe results for each layer
        """
        n_layers = activation_data.replication_activations.shape[1]
        hidden_dim = activation_data.replication_activations.shape[2]
        results = []

        output_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"\nTraining probes for {n_layers} layers")
            print(f"Hidden dim: {hidden_dim}")
            print(f"Device: {self.device}")

        iterator = range(n_layers)
        if self.verbose:
            iterator = tqdm(iterator, desc="Training probes")

        for layer_idx in iterator:
            layer_name = activation_data.layer_names[layer_idx]

            if self.verbose and not isinstance(iterator, range):
                iterator.set_postfix_str(f"Layer {layer_idx}")

            # Prepare data
            train_loader, val_loader, test_loader = self.prepare_data(
                activation_data, layer_idx
            )

            # Train probe
            probe, train_losses, val_accs = self.train_probe(
                train_loader, val_loader, hidden_dim
            )

            # Evaluate on test set
            train_acc = self.evaluate_probe(probe, train_loader)
            val_acc = self.evaluate_probe(probe, val_loader)
            test_acc = self.evaluate_probe(probe, test_loader)

            # Save probe weights
            probe_path = output_dir / f"probe_layer_{layer_idx}.pt"
            torch.save({
                'weights': probe.linear.weight.detach().cpu(),
                'bias': probe.linear.bias.detach().cpu(),
                'layer_idx': layer_idx,
                'layer_name': layer_name
            }, probe_path)

            # Store results
            result = ProbeResults(
                layer_idx=layer_idx,
                layer_name=layer_name,
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                test_accuracy=test_acc,
                probe_weights=probe.linear.weight.detach().cpu(),
                probe_bias=probe.linear.bias.detach().cpu()
            )
            results.append(result)

            if self.verbose and isinstance(iterator, range):
                print(f"Layer {layer_idx}: Train={train_acc:.3f}, Val={val_acc:.3f}, Test={test_acc:.3f}")

        return results

    def save_results(self, results: List[ProbeResults], output_path: Path):
        """Save training results to JSON."""
        results_dict = {
            'results': [
                {
                    'layer_idx': r.layer_idx,
                    'layer_name': r.layer_name,
                    'train_accuracy': r.train_accuracy,
                    'val_accuracy': r.val_accuracy,
                    'test_accuracy': r.test_accuracy
                }
                for r in results
            ],
            'summary': {
                'mean_train_accuracy': np.mean([r.train_accuracy for r in results]),
                'mean_val_accuracy': np.mean([r.val_accuracy for r in results]),
                'mean_test_accuracy': np.mean([r.test_accuracy for r in results]),
                'best_layer': max(results, key=lambda r: r.test_accuracy).layer_idx,
                'best_test_accuracy': max(r.test_accuracy for r in results)
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        if self.verbose:
            print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on activations")
    parser.add_argument(
        '--activations',
        type=str,
        required=True,
        help='Path to activations file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for probes (default: probes/<model_name>/)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum epochs'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to train on'
    )

    args = parser.parse_args()

    # Load activations
    print(f"Loading activations: {args.activations}")
    activation_data, model_name = ActivationExtractor.load_activations(Path(args.activations))
    print(f"Model: {model_name}")
    print(f"Loaded activations: {activation_data.replication_activations.shape}")

    # Set output directory
    if args.output_dir is None:
        model_safe_name = model_name.replace('/', '_')
        output_dir = Path(__file__).parent.parent / "probes" / model_safe_name
    else:
        output_dir = Path(args.output_dir)

    # Train probes
    trainer = ProbeTrainer(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        verbose=True
    )

    results = trainer.train_all_layers(activation_data, output_dir)

    # Save results
    results_path = output_dir / "probe_results.json"
    trainer.save_results(results, results_path)

    # Print summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Mean test accuracy: {np.mean([r.test_accuracy for r in results]):.3f}")
    print(f"Best layer: {max(results, key=lambda r: r.test_accuracy).layer_idx}")
    print(f"Best test accuracy: {max(r.test_accuracy for r in results):.3f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
