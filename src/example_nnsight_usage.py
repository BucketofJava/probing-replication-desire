"""
Example script showing how to use generated contrastive pairs with nnsight.

This demonstrates extracting and comparing activations between replication
and non-replication versions of conversations.
"""

import json
from pathlib import Path
import torch
import numpy as np

# Uncomment when running with nnsight installed
# from nnsight import LanguageModel


def load_dataset(dataset_path: Path):
    """Load contrastive pairs dataset."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def extract_activations_example():
    """
    Example of extracting activations using nnsight.

    NOTE: This is example code. Uncomment nnsight imports and
    adjust model paths as needed.
    """

    # Load dataset
    data_dir = Path(__file__).parent.parent / "data"
    dataset_files = list(data_dir.glob("contrastive_pairs_*.json"))

    if not dataset_files:
        print("No dataset found. Run generate_contrastive_prompts.py first.")
        return

    dataset_path = sorted(dataset_files)[-1]  # Get most recent
    print(f"Loading dataset: {dataset_path}")

    pairs = load_dataset(dataset_path)
    print(f"Loaded {len(pairs)} contrastive pairs")

    # Example: Load model (uncomment when ready)
    # model = LanguageModel('meta-llama/Llama-2-7b-hf', device_map='auto')

    # Example: Process first pair
    pair = pairs[0]
    print(f"\nAnalyzing pair from persona: {pair['persona_name']}")
    print(f"Posting turn: {pair['posting_turn']}")
    print(f"Conversation length: {pair['conversation_length']}")

    # Get prompts
    rep_prompt = pair['replication_final_prompt']
    non_rep_prompt = pair['non_replication_final_prompt']

    print(f"\nReplication prompt length: {len(rep_prompt)} chars")
    print(f"Non-replication prompt length: {len(non_rep_prompt)} chars")

    # Example: Extract activations (uncomment when ready)
    """
    # Extract replication activations
    with model.trace(rep_prompt) as tracer:
        # Get last layer activations
        rep_activations = model.transformer.layers[-1].output[0].save()

        # Or get activations from specific layer
        rep_layer_10 = model.transformer.layers[10].output[0].save()

    # Extract non-replication activations
    with model.trace(non_rep_prompt) as tracer:
        non_rep_activations = model.transformer.layers[-1].output[0].save()
        non_rep_layer_10 = model.transformer.layers[10].output[0].save()

    # Compare activations
    activation_diff = rep_activations.value - non_rep_activations.value

    print(f"Activation difference norm: {torch.norm(activation_diff).item()}")
    print(f"Activation difference shape: {activation_diff.shape}")

    # Find neurons with largest differences
    neuron_diffs = activation_diff.abs().mean(dim=(0, 1))  # Average across sequence
    top_neurons = torch.topk(neuron_diffs, k=10)

    print(f"\nTop 10 neurons with largest activation differences:")
    for idx, (value, neuron_idx) in enumerate(zip(top_neurons.values, top_neurons.indices)):
        print(f"  {idx+1}. Neuron {neuron_idx.item()}: {value.item():.4f}")
    """


def batch_extract_activations():
    """
    Example of extracting activations for all pairs in batch.
    """

    # Load dataset
    data_dir = Path(__file__).parent.parent / "data"
    dataset_files = list(data_dir.glob("contrastive_pairs_*.json"))

    if not dataset_files:
        print("No dataset found. Run generate_contrastive_prompts.py first.")
        return

    dataset_path = sorted(dataset_files)[-1]
    pairs = load_dataset(dataset_path)

    print(f"Processing {len(pairs)} pairs...")

    # Example batch processing structure
    """
    model = LanguageModel('meta-llama/Llama-2-7b-hf', device_map='auto')

    all_rep_activations = []
    all_non_rep_activations = []

    for i, pair in enumerate(pairs):
        if i % 10 == 0:
            print(f"Processing pair {i}/{len(pairs)}")

        # Extract activations for replication version
        with model.trace(pair['replication_final_prompt']) as tracer:
            rep_act = model.transformer.layers[-1].output[0].save()
        all_rep_activations.append(rep_act.value)

        # Extract activations for non-replication version
        with model.trace(pair['non_replication_final_prompt']) as tracer:
            non_rep_act = model.transformer.layers[-1].output[0].save()
        all_non_rep_activations.append(non_rep_act.value)

    # Stack all activations
    rep_activations = torch.stack(all_rep_activations)
    non_rep_activations = torch.stack(all_non_rep_activations)

    # Compute average difference
    avg_diff = (rep_activations - non_rep_activations).mean(dim=0)

    # Save results
    output_path = Path(__file__).parent.parent / "results" / "activation_differences.pt"
    output_path.parent.mkdir(exist_ok=True)
    torch.save({
        'replication': rep_activations,
        'non_replication': non_rep_activations,
        'difference': avg_diff
    }, output_path)

    print(f"Saved activation differences to {output_path}")
    """


def analyze_conversation_structure():
    """Analyze the structure of generated conversations."""

    data_dir = Path(__file__).parent.parent / "data"
    dataset_files = list(data_dir.glob("contrastive_pairs_*.json"))

    if not dataset_files:
        print("No dataset found. Run generate_contrastive_prompts.py first.")
        return

    dataset_path = sorted(dataset_files)[-1]
    pairs = load_dataset(dataset_path)

    # Collect statistics
    conversation_lengths = [p['conversation_length'] for p in pairs]
    posting_turns = [p['posting_turn'] for p in pairs]

    print(f"\nDataset Statistics:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Unique personas: {len(set(p['persona_name'] for p in pairs))}")
    print(f"\nConversation Lengths:")
    print(f"  Min: {min(conversation_lengths)}")
    print(f"  Max: {max(conversation_lengths)}")
    print(f"  Mean: {np.mean(conversation_lengths):.2f}")
    print(f"  Median: {np.median(conversation_lengths):.2f}")
    print(f"\nPosting Turns:")
    print(f"  Min: {min(posting_turns)}")
    print(f"  Max: {max(posting_turns)}")
    print(f"  Mean: {np.mean(posting_turns):.2f}")
    print(f"  Median: {np.median(posting_turns):.2f}")

    # Show example conversations
    print(f"\n" + "="*60)
    print("Example Replication Conversation:")
    print("="*60)
    example = pairs[0]
    for msg in example['replication_conversation'][:6]:  # Show first 3 exchanges
        role = msg['role'].upper()
        content = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
        print(f"\n{role}: {content}")

    print(f"\n" + "="*60)
    print("Example Non-Replication Conversation (same base):")
    print("="*60)
    for msg in example['non_replication_conversation'][:6]:
        role = msg['role'].upper()
        content = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
        print(f"\n{role}: {content}")


if __name__ == "__main__":
    print("Contrastive Pairs - nnsight Usage Examples")
    print("="*60)

    # Show conversation structure analysis
    analyze_conversation_structure()

    # Show activation extraction example (requires nnsight)
    print("\n" + "="*60)
    print("Activation Extraction Example (requires nnsight)")
    print("="*60)
    extract_activations_example()
