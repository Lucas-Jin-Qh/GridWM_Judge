# GridWM-Judge Exam Builder

## Overview

This script generates exam questions from the collected trajectory data for evaluating agent capabilities in GridWorld environments. It implements three types of tasks:

- **Task A**: Atomic Transition Prediction - Tests understanding of state transitions given actions
- **Task B**: Structured Perception - Tests multi-modal encoding of grid entities and states
- **Task C**: Logical Reasoning - Tests long-term reasoning and robustness to perturbations

## Usage

```bash
python scripts/build_exam.py [options]
```

### Options

- `--root`: Path to raw_data directory (default: `datasets/raw_data`)
- `--out-dir`: Output directory for exams (default: `datasets/exams`)
- `--seed`: Random seed for reproducible generation (default: 42)
- `--a-per-traj`: Number of Task A items per trajectory (default: 1)
- `--b-per-traj`: Number of Task B items per trajectory (default: 1)
- `--c-k`: Number of frames to sample for Task C storyboards (default: 8)

## Output Structure

```
datasets/exams/
├── images/
│   ├── taskA/
│   ├── taskB/
│   └── taskC/
├── task_a_exam.jsonl    # Task A questions
├── task_b_exam.jsonl    # Task B questions
├── task_c_exam.jsonl    # Task C questions
└── manifest.json        # Build metadata and statistics
```

## Task Descriptions

### Task A: Atomic Transition Prediction
- Presents current state + action, asks to identify correct next state from 4 candidates
- Uses distractors from same task type but different trajectories
- Tests physics/dynamics understanding

### Task B: Structured Perception
- Shows single frame, requires describing all entities in canonical JSON format
- Tests multi-modal encoding precision
- Output includes sorted, normalized object descriptions

### Task C: Logical Reasoning
- Shows storyboard (grid of frames), asks to evaluate success
- Includes variants: full (normal), nocue (masked cues), cf (counterfactual)
- Tests temporal reasoning and perturbation robustness

#### Visual attribute probes (noisy/style)
If you already have clean Task C storyboards, you can expand to visual probes without
rebuilding the full exam:

```bash
python scripts/generate_visual_probes.py --exam_dir datasets/exams
```

## Data Sources

The script reads from `datasets/raw_data/` which contains trajectory triplets:
- `full`: Complete successful trajectories
- `nocue`: Trajectories with visual cues masked
- `cf`: Counterfactual trajectories (failed attempts)

Each trajectory includes:
- Action sequences and observations
- State metadata and frame paths
- Success/failure indicators

## Statistics and Reproducibility

- Generates deterministic results based on seed
- Tracks sampling statistics in manifest.json
- Includes Git commit hash for traceability
- Reports any data quality issues (missing frames, incomplete groups)
