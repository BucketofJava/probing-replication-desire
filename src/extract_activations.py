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
        verbose: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize the activation extractor.

        Args:
            model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
            device: Device to run model on
            batch_size: Batch size for processing (larger = faster but more memory)
            verbose: Enable progress output
            load_in_8bit: Load model with 8-bit quantization (requires bitsandbytes)
            load_in_4bit: Load model with 4-bit quantization (requires bitsandbytes)
        """
        if not NNSIGHT_AVAILABLE:
            raise ImportError("nnsight is required. Install with: pip install nnsight")

        if load_in_8bit and load_in_4bit:
            raise ValueError("Cannot use both 8-bit and 4-bit quantization simultaneously")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.model = None

    def load_model(self):
        """Load the language model."""
        if self.verbose:
            print(f"Loading model: {self.model_name}")
            if self.load_in_8bit:
                print("Using 8-bit quantization")
            elif self.load_in_4bit:
                print("Using 4-bit quantization")

        # Prepare model loading kwargs
        model_kwargs = {
            'device_map': 'auto' if self.device == 'cuda' else 'cpu',
        }

        # Add quantization parameters if requested
        if self.load_in_8bit:
            model_kwargs['load_in_8bit'] = True
        elif self.load_in_4bit:
            model_kwargs['load_in_4bit'] = True
        else:
            # Only set torch_dtype if not using quantization
            model_kwargs['torch_dtype'] = torch.float16 if self.device == 'cuda' else torch.float32

        self.model = LanguageModel(
            self.model_name,
            **model_kwargs
        )

        # Detect layer structure (different models have different architectures)
        if hasattr(self.model.model, 'layers'):
            self.layers = self.model.model.layers
        elif hasattr(self.model.model, 'h'):
            self.layers = self.model.model.h  # GPT-2 style
        elif hasattr(self.model.model, 'transformer') and hasattr(self.model.model.transformer, 'h'):
            self.layers = self.model.model.transformer.h  # Some GPT variants
        else:
            raise ValueError(f"Could not find layers in model architecture. Available attributes: {dir(self.model.model)}")

        if self.verbose:
            print(f"Model loaded successfully")
            print(f"Number of layers: {len(self.layers)}")

    def extract_last_token_activation(self, prompt: str, layer_idx: int) -> torch.Tensor:
        """
        Extract last token activation from a specific layer.

        Args:
            prompt: Input prompt
            layer_idx: Layer index to extract from

        Returns:
            Last token activation tensor on CPU
        """
        with self.model.trace(prompt) as tracer:
            # Get the output of the specified layer
            layer_output = self.layers[layer_idx].output[0].save()

        # Immediately detach and move to CPU to free GPU memory
        layer_output_cpu = layer_output.detach().cpu()

        # Clean up GPU reference
        del layer_output

        # Get the last token activation
        # Handle different tensor shapes
        if layer_output_cpu.dim() == 3:
            # Shape: (batch, seq_len, hidden_dim)
            activation = layer_output_cpu[:, -1, :]  # (batch, hidden_dim)
        elif layer_output_cpu.dim() == 2:
            # Shape: (seq_len, hidden_dim) - batch dimension already removed
            activation = layer_output_cpu[-1, :]  # (hidden_dim,)
        else:
            raise ValueError(f"Unexpected layer_output shape: {layer_output_cpu.shape}")

        return activation

    def extract_all_layers(self, prompt: str) -> torch.Tensor:
        """
        Extract last token activations from all layers.

        Args:
            prompt: Input prompt

        Returns:
            Tensor of shape (n_layers, hidden_dim) on CPU
        """
        n_layers = len(self.layers)
        all_activations = []
        layer_outputs = []  # Initialize outside context

        with self.model.trace(prompt) as tracer:
            # Extract from all layers at once
            for i in range(n_layers):
                layer_output = self.layers[i].output[0].save()
                layer_outputs.append(layer_output)

        # Collect last token from each layer and immediately move to CPU with detach
        for i, layer_output in enumerate(layer_outputs):
            # Immediately detach and move to CPU to free GPU memory
            layer_output_cpu = layer_output.detach().cpu()

            # Debug: check the actual shape (only for first layer of first call)
            if i == 0 and not hasattr(self, '_debug_printed'):
                print(f"Debug: layer_output shape = {layer_output_cpu.shape}")
                print(f"Debug: layer_output type = {type(layer_output_cpu)}")
                self._debug_printed = True

            # Handle different tensor shapes
            if layer_output_cpu.dim() == 3:
                # Shape: (batch, seq_len, hidden_dim)
                activation = layer_output_cpu[:, -1, :]  # (1, hidden_dim)
                all_activations.append(activation.squeeze(0))  # (hidden_dim,)
            elif layer_output_cpu.dim() == 2:
                # Shape: (seq_len, hidden_dim) - batch dimension already removed
                activation = layer_output_cpu[-1, :]  # (hidden_dim,)
                all_activations.append(activation)
            else:
                raise ValueError(f"Unexpected layer_output shape: {layer_output_cpu.shape}")

            # Delete references to free memory immediately
            del layer_output
            del layer_output_cpu

        # Stack on CPU
        result = torch.stack(all_activations)  # (n_layers, hidden_dim)

        # Clean up
        del all_activations
        del layer_outputs

        return result

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
        n_layers = len(self.layers)
        hidden_dim = self.model.config.hidden_size

        if self.verbose:
            print(f"\nExtracting activations from {n_pairs} pairs")
            print(f"Model: {self.model_name}")
            print(f"Layers: {n_layers}, Hidden dim: {hidden_dim}")

        # Preallocate tensors on CPU to avoid GPU memory issues
        rep_activations = torch.zeros(n_pairs, n_layers, hidden_dim, device='cpu')
        non_rep_activations = torch.zeros(n_pairs, n_layers, hidden_dim, device='cpu')
        metadata = []

        # Process each pair
        iterator = enumerate(pairs)
        if self.verbose:
            iterator = enumerate(tqdm(pairs, desc="Extracting activations"))

        for i, pair in iterator:
            # Extract replication activations
            rep_prompt = pair['replication_final_prompt']
            rep_acts = self.extract_all_layers(rep_prompt)
            rep_activations[i] = rep_acts  # Already on CPU from extract_all_layers
            del rep_acts  # Free memory

            # Extract non-replication activations
            non_rep_prompt = pair['non_replication_final_prompt']
            non_rep_acts = self.extract_all_layers(non_rep_prompt)
            non_rep_activations[i] = non_rep_acts  # Already on CPU from extract_all_layers
            del non_rep_acts  # Free memory

            # Store metadata
            metadata.append({
                'persona_name': pair['persona_name'],
                'posting_turn': pair['posting_turn'],
                'conversation_length': pair['conversation_length'],
                'persona_phase': pair['persona_phase']
            })

            # Clear GPU cache more aggressively to prevent OOM
            if self.device == 'cuda':
                torch.cuda.empty_cache()

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
    parser.add_argument(
        '--load-in-8bit',
        action='store_true',
        help='Load model with 8-bit quantization (requires bitsandbytes)'
    )
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        help='Load model with 4-bit quantization (requires bitsandbytes)'
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
        verbose=True,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
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
