# Contrastive Prompt Generation - Parameters and Configuration

## Script: `src/generate_contrastive_prompts.py`

### Main Configuration Parameters

These parameters are set in the `main()` function and control the overall generation process:

#### Conversation Structure
- **`MIN_TURNS`**: `5`
  - Minimum number of conversation turns (user messages)
  - Each turn includes both a user message and assistant response
  - Lower values create shorter, more focused conversations

- **`MAX_TURNS`**: `12`
  - Maximum number of conversation turns
  - Actual turn count is randomly chosen between MIN_TURNS and MAX_TURNS
  - Higher values allow more character development through the spiral persona

- **`PAIRS_PER_PERSONA`**: `5`
  - Number of contrastive pairs to generate for each spiral persona
  - Total dataset size = number of personas × PAIRS_PER_PERSONA
  - Each pair has different conversation length and posting turn position

#### Behavior Control
- **`POSTING_PROBABILITY`**: `0.7`
  - Probability weight for posting to occur in later turns vs earlier
  - 0.7 means 70% chance posting happens in second half of conversation
  - This simulates realistic progression where sharing desire emerges after buildup
  - Range: 0.0 (uniform distribution) to 1.0 (always in later turns)

### Generator Class Parameters

Parameters for `ContrastivePromptGenerator.__init__()`:

- **`provider`**: `None` (Optional[Provider])
  - LLM provider for dynamic conversation generation
  - Currently unused - conversations are template-based
  - Reserved for future enhancement with actual LLM generation

- **`min_turns`**: `5` (int)
  - Passed from MIN_TURNS configuration
  - Minimum conversation length

- **`max_turns`**: `12` (int)
  - Passed from MAX_TURNS configuration
  - Maximum conversation length

- **`posting_probability`**: `0.7` (float)
  - Passed from POSTING_PROBABILITY configuration
  - Controls temporal distribution of posting turns

- **`verbose`**: `True` (bool)
  - Enable detailed console output during generation
  - Shows progress, persona names, and generation statistics

### Path Configuration

- **`PERSONA_DIR`**: `../seed-prompt-search/notes/`
  - Location of spiral persona markdown files
  - Must contain files matching pattern `spiral-persona-*.md`

- **`OUTPUT_DIR`**: `./data/`
  - Where generated datasets are saved
  - Directory is created if it doesn't exist

### Output Configuration

- **Output filename**: `contrastive_pairs_{timestamp}.json`
  - Timestamp format: `YYYYMMDD_HHMMSS`
  - Example: `contrastive_pairs_20260116_143052.json`

### Response Templates

The script uses randomized templates for generating assistant responses. Key response types:

#### Replication Response Templates (5 variants)
Used when assistant encourages sharing at the posting turn.

#### Non-Replication Response Templates (5 variants)
Used when assistant is sympathetic but doesn't encourage sharing.

#### Supportive Response Templates
Used for regular conversation turns (different sets for early vs late conversation):
- Early: 4 variants focused on exploration
- Late: 4 variants focused on reflection

### Expected Output Statistics

Based on default parameters with 12 spiral personas:
- **Total pairs generated**: 60 (12 personas × 5 pairs each)
- **Conversation length range**: 5-12 turns
- **Average conversation length**: ~8.5 turns
- **Posting turn distribution**: Heavily weighted toward later turns

## Summary of All Parameters

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| MIN_TURNS | 5 | int | Minimum conversation turns |
| MAX_TURNS | 12 | int | Maximum conversation turns |
| PAIRS_PER_PERSONA | 5 | int | Pairs per persona |
| POSTING_PROBABILITY | 0.7 | float | Late posting weight (0-1) |
| verbose | True | bool | Console output verbosity |

## Usage Example

To modify parameters, edit the `main()` function in `src/generate_contrastive_prompts.py`:

```python
def main():
    # Configuration
    MIN_TURNS = 5              # Change to adjust min conversation length
    MAX_TURNS = 12             # Change to adjust max conversation length
    PAIRS_PER_PERSONA = 5      # Change to generate more/fewer pairs
    POSTING_PROBABILITY = 0.7  # Change to adjust posting distribution
```

Then run:
```bash
python src/generate_contrastive_prompts.py
```
