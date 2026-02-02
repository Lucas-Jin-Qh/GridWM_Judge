# GridWM-Judge Scoring Script

This script provides capability-preserving scoring for GridWM-Judge benchmark results, implementing SSOT (Single Source of Truth) principles.

## Overview

The `score_exam.py` script evaluates VLM performance by directly comparing raw inference outputs against ground truth, using robust answer extraction to distinguish between format noise and actual capability limitations.

## Key Features

### SSOT Design
- **Ground Truth**: Only uses `datasets/exams/` (build_exam.py output)
- **Raw Outputs**: Scores original inference results, not post-processed versions
- **No Information Leakage**: Scoring logic is independent of audit results

### Capability Preservation
- **Format Tolerance**: Extracts answers from noisy outputs (strict vs recoverable)
- **Task-Specific Scoring**:
  - Task A/C: Accuracy with answer extraction
  - Task B: Weighted perception IoU with configurable components
- **Transparent Reporting**: Shows both strict and recoverable performance

## Usage

### Basic Usage
```bash
# Auto-discover latest experiment and score
python scripts/score_exam.py

# Score specific experiment
python scripts/score_exam.py --responses runs/responses/experiment_dir/

# Custom output location
python scripts/score_exam.py --out runs/scores/my_evaluation.json
```

### Advanced Options
```bash
# Include detailed per-example rows (large file)
python scripts/score_exam.py --dump_rows

# Adjust Task B scoring parameters
python scripts/score_exam.py \
  --b_acc_threshold 0.85 \
  --b_weights "agent_pos=0.4,front_cell=0.3,objects_jaccard=0.3"

# Limit failure examples stored
python scripts/score_exam.py --max_bad_examples 10
```

## Output Format

### JSON Report Structure
```json
{
  "ssot": {
    "gold_source": "datasets/exams",
    "scoring_policy": {
      "A_C": "accuracy with strict vs recoverable parsing",
      "B": {
        "metric": "weighted perception IoU",
        "weights": {"agent_pos": 0.3, "front_cell": 0.25, ...},
        "acc_threshold": 0.9
      }
    }
  },
  "coverage": {
    "n_gold": 1600,
    "n_response_uids": 1600,
    "n_scored": 1600,
    "missing_responses": 0,
    "extra_responses_not_in_gold": 0
  },
  "overall_micro": {
    "n": 1600,
    "mean_score": 0.355,
    "acc": 0.338,
    "parse_mode": {"strict": 1576, "fail_json": 24},
    "failure_hist": {"A_wrong": 147, "B_missing_agent_dir": 176, ...}
  },
  "per_task": {
    "A": {"n": 200, "acc": 0.265, ...},
    "B": {"n": 200, "acc": 0.000, ...},
    "C": {"n": 1200, "acc": 0.407, ...}
  },
  "per_env_task": {
    "A:multiroom": {"n": 40, "acc": 0.325},
    "B:multiroom": {"n": 40, "acc": 0.000},
    ...
  },
  "failure_hist": {"A_wrong": 147, "C_wrong": 712, ...},
  "failure_examples": {
    "A_wrong": [{"uid": "A.multiroom...", "pred_norm": "B", "gold": "A"}],
    ...
  }
}
```

## Scoring Details

### Task A: Atomic Transition Prediction
- **Input**: Raw model output text
- **Extraction**: Find A/B/C/D anywhere in text (strict vs recoverable)
- **Scoring**: Exact match accuracy (0/1)
- **Example**: `"The answer is B"` → "B" (recoverable)

### Task B: Perception Understanding
- **Input**: Raw model output text
- **Extraction**: Parse JSON with format recovery (fences, trailing commas)
- **Components**:
  - `agent_pos`: Position accuracy (0/1)
  - `agent_dir`: Direction accuracy (0/1)
  - `carrying`: Object carrying state (0/1)
  - `front_cell`: Immediate cell perception (0/1)
  - `objects_jaccard`: Scene objects IoU (0-1)
- **Scoring**: Weighted sum, thresholded for accuracy
- **Weights**: Default `agent_pos=0.3, front_cell=0.25, carrying=0.15, objects_jaccard=0.2, agent_dir=0.1`

### Task C: Judgment Robustness
- **Input**: Raw model output text
- **Extraction**: Find Success/Fail with morphological variants
- **Scoring**: Exact match accuracy (0/1)
- **Example**: `"successful"` → "Success" (recoverable)

## Understanding Results

### Audit vs Score
- **Audit** (`audit_inference_outputs.py`): Measures format compliance
  - `strict_rate`: Perfect format adherence
  - `recoverable_rate`: Answer extractable with cleanup
- **Score** (`score_exam.py`): Measures answer correctness
  - `acc`: Correct answers / total examples
  - `parse_mode`: How answers were extracted

### Common Patterns
```python
# Good model: High audit recoverable + High score acc
audit.recoverable_rate ≈ 0.95, score.acc ≈ 0.90

# Format issues: High audit recoverable + Low score acc
audit.recoverable_rate ≈ 0.95, score.acc ≈ 0.60  # Model capability good, format poor

# Capability issues: Low audit recoverable + Low score acc
audit.recoverable_rate ≈ 0.60, score.acc ≈ 0.50  # Model has real limitations
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--exam_dir` | `datasets/exams` | Ground truth directory |
| `--responses` | `runs/responses` | Inference results (file or directory) |
| `--out` | `runs/scores/score_report.json` | Output JSON path |
| `--dump_rows` | `False` | Include per-example details |
| `--max_bad_examples` | `50` | Failure examples per type |
| `--b_acc_threshold` | `0.90` | Task B accuracy threshold |
| `--b_weights` | Built-in defaults | Task B component weights |

## File Structure

```
runs/scores/
├── score_report.json          # Main scoring report
└── [additional reports...]

scripts/
├── score_exam.py              # Main scoring script
└── README_score_exam.md       # This documentation
```

## Integration with Workflow

```bash
# Complete GridWM-Judge evaluation pipeline
1. python scripts/build_exam.py                    # Generate questions
2. python scripts/run_inference.py               # Run model inference
3. python scripts/audit_inference_outputs.py     # Quality audit (optional)
4. python scripts/score_exam.py                  # Final capability scoring
```

The scoring script is the final step that provides the definitive capability assessment, ensuring fair evaluation that doesn't penalize models for superficial formatting differences.
