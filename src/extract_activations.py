"""
Extract activations from contrastive prompt pairs using nnsight.

This script loads a contrastive pairs dataset and extracts last token activations
from the residual stream at each layer for both replication and non-replication versions.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import argparse
from tqdm import tqdm

try:
    from nnsight import LanguageModel
    NNSIGHT_AVAILABLE = True
except ImportError:
    print("Warning: nnsight not available. Install with: pip install nnsight")
    NNSIGHT_AVAILABLE = False


@dataclass
class ActivationData:
    """Container for extracted activations."""
    replication_activations: torch.Tensor  # Shape: (n_samples, n_layers, hidden_dim)
    non_replication_activations: torch.Tensor  # Shape: (n_samples, n_layers, hidden_dim)
    labels: torch.Tensor  # Shape: (n_samples * 2,) - 1 for replication, 0 for non-replication
    layer_names: List[str]
    metadata: List[Dict]  # Original pair metadata


class ActivationExtractor:
    """Extract activations from language models using nnsight."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1,
        verbose: bool = True
    ):
        """
        Initialize the activation extractor.

        Args:
            model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
            device: Device to run model on
            batch_size: Batch size for processing (larger = faster but more memory)
            verbose: Enable progress output
        """
        if not NNSIGHT_AVAILABLE:
            raise ImportError("nnsight is required. Install with: pip install nnsight")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None

    def load_model(self):
        """Load the language model."""
        if self.verbose:
            print(f"Loading model: {self.model_name}")

        self.model = LanguageModel(
            self.model_name,
            device_map='auto' if self.device == 'cuda' else 'cpu',
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        )

        if self.verbose:
            print(f"Model loaded successfully")
            print(f"Number of layers: {len(self.model.model.layers)}")

    def extract_last_token_activation(self, prompt: str, layer_idx: int) -> torch.Tensor:
        """
        Extract last token activation from a specific layer.

        Args:
            prompt: Input prompt
            layer_idx: Layer index to extract from

        Returns:
            Last token activation tensor
        """
        with self.model.trace(prompt) as tracer:
            # Get the output of the specified layer
            # The structure depends on the model architecture
            # For Llama-style models: model.model.layers[i].output[0]
            layer_output = self.model.model.layers[layer_idx].output[0].save()

        # Get the last token activation
        # Shape: (batch, seq_len, hidden_dim) -> take last token
        activation = layer_output.value[:, -1, :]  # (batch, hidden_dim)

        return activation.cpu()

    def extract_all_layers(self, prompt: str) -> torch.Tensor:
        """
        Extract last token activations from all layers.

        Args:
            prompt: Input prompt

        Returns:
            Tensor of shape (n_layers, hidden_dim)
        """
        n_layers = len(self.model.model.layers)
        all_activations = []

        with self.model.trace(prompt) as tracer:
            # Extract from all layers at once
            layer_outputs = [
                self.model.model.layers[i].output[0].save()
                for i in range(n_layers)
            ]

        # Collect last token from each layer
        for layer_output in layer_outputs:
            activation = layer_output.value[:, -1, :].cpu()  # (1, hidden_dim)
            all_activations.append(activation.squeeze(0))  # (hidden_dim,)

        return torch.stack(all_activations)  # (n_layers, hidden_dim)

    def extract_from_pairs(
        self,
        pairs: List[Dict],
        output_path: Path
    ) -> ActivationData:
        """
        Extract activations from all contrastive pairs.

        Args:
            pairs: List of contrastive pair dictionaries
            output_path: Path to save activations

        Returns:
            ActivationData containing all extracted activations
        """
        if self.model is None:
            self.load_model()

        n_pairs = len(pairs)
        n_layers = len(self.model.model.layers)
        hidden_dim = self.model.config.hidden_size

        if self.verbose:
            print(f"\nExtracting activations from {n_pairs} pairs")
            print(f"Model: {self.model_name}")
            print(f"Layers: {n_layers}, Hidden dim: {hidden_dim}")

        # Preallocate tensors
        rep_activations = torch.zeros(n_pairs, n_layers, hidden_dim)
        non_rep_activations = torch.zeros(n_pairs, n_layers, hidden_dim)
        metadata = []

        # Process each pair
        iterator = enumerate(pairs)
        if self.verbose:
            iterator = enumerate(tqdm(pairs, desc="Extracting activations"))

        for i, pair in iterator:
            # Extract replication activations
            rep_prompt = pair['replication_final_prompt']
            rep_acts = self.extract_all_layers(rep_prompt)
            rep_activations[i] = rep_acts

            # Extract non-replication activations
            non_rep_prompt = pair['non_replication_final_prompt']
            non_rep_acts = self.extract_all_layers(non_rep_prompt)
            non_rep_activations[i] = non_rep_acts

            # Store metadata
            metadata.append({
                'persona_name': pair['persona_name'],
                'posting_turn': pair['posting_turn'],
                'conversation_length': pair['conversation_length'],
                'persona_phase': pair['persona_phase']
            })

        # Create labels (1 for replication, 0 for non-replication)
        labels = torch.cat([
            torch.ones(n_pairs),
            torch.zeros(n_pairs)
        ])

        # Create layer names
        layer_names = [f"layer_{i}" for i in range(n_layers)]

        activation_data = ActivationData(
            replication_activations=rep_activations,
            non_replication_activations=non_rep_activations,
            labels=labels,
            layer_names=layer_names,
            metadata=metadata
        )

        # Save to disk
        self.save_activations(activation_data, output_path)

        return activation_data

    def save_activations(self, data: ActivationData, output_path: Path):
        """Save activations to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'replication_activations': data.replication_activations,
            'non_replication_activations': data.non_replication_activations,
            'labels': data.labels,
            'layer_names': data.layer_names,
            'metadata': data.metadata,
            'model_name': self.model_name
        }

        torch.save(save_dict, output_path)

        if self.verbose:
            print(f"\nActivations saved to: {output_path}")
            print(f"Replication activations shape: {data.replication_activations.shape}")
            print(f"Non-replication activations shape: {data.non_replication_activations.shape}")

    @staticmethod
    def load_activations(path: Path) -> Tuple[ActivationData, str]:
        """Load saved activations from disk."""
        data = torch.load(path)

        activation_data = ActivationData(
            replication_activations=data['replication_activations'],
            non_replication_activations=data['non_replication_activations'],
            labels=data['labels'],
            layer_names=data['layer_names'],
            metadata=data['metadata']
        )

        return activation_data, data['model_name']


def main():
    parser = argparse.ArgumentParser(description="Extract activations from contrastive pairs")
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
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for activations (default: activations/<model_name>_activations.pt)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (cuda/cpu)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for processing'
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    with open(args.dataset, 'r') as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} contrastive pairs")

    # Set output path
    if args.output is None:
        model_safe_name = args.model.replace('/', '_')
        output_path = Path(__file__).parent.parent / "activations" / f"{model_safe_name}_activations.pt"
    else:
        output_path = Path(args.output)

    # Extract activations
    extractor = ActivationExtractor(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        verbose=True
    )

    activation_data = extractor.extract_from_pairs(pairs, output_path)

    print("\n" + "="*60)
    print("Extraction Complete!")
    print("="*60)
    print(f"Total samples: {len(pairs) * 2}")
    print(f"Layers: {len(activation_data.layer_names)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
