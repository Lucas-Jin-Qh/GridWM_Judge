# GridWM-Judge: VLM World Model Evaluation Framework

GridWM-Judge is a comprehensive framework for evaluating Vision-Language Models (VLMs) as judges in embodied AI tasks. The framework investigates whether VLM "judging instability" stems from "world model deficits" by testing atomic transition prediction, perception, and judgment robustness across controlled MiniGrid environments.

##  Current Status

###  Raw Dataset Generation Complete
- **6 Tasks**: DoorKey, Memory, LavaGap, KeyCorridor, MultiRoom, RedBlueDoor
- **200 High-Quality Samples**: 40 samples per task for most tasks, 20 for DoorKey/RedBlueDoor
- **600 Total Trajectories**: 200 samples x 3 variants (Full/NoCue/CF)
- **Agent POV Frames**: Rendered for all trajectories

###  Exam Dataset Generation Complete
- **Task A (Atomic Transition)**: 200 questions - State transition prediction
- **Task B (Perception)**: 200 questions - Structured scene understanding
- **Task C (Judgment)**: 1200 questions - Success/failure classification
- **1600 Rendered Images**: Composite images for all exam questions

###  Inference Pipeline Ready (v7.6.11)
- **Capability-Preserving Inference**: Task-aware token budgets prevent truncation artifacts
- **Multi-Backend Support**: OpenAI-compatible (SiliconFlow), Gemini, Local models
- **Scientific Auditing**: Distinguishes format noise from true capability limits
- **Experiment Management**: Automatic organization with isolated result directories

###  Scoring System Complete (SSOT Scoring)
- **Comprehensive Evaluation**: Automated scoring with detailed performance breakdown
- **Capability-Fidelity Metrics**: Strict vs recoverable parsing, weighted component scoring
- **Failure Analysis**: Detailed error categorization and debugging support

###  Production Validation Complete
- **Qwen2.5-VL-7B Baseline Results**:
  - **Task A**: 26.5% accuracy (format: , reasoning: )
  - **Task B**: 0.0% accuracy (format: , spatial understanding: )
  - **Task C**: 40.7% accuracy (temporal reasoning: )
- **Key Insights**: Reveals VLM capabilities vs limitations in embodied AI tasks

###  Task Implementations Verified
- **Full Trajectories**: BFS-generated optimal solutions with success validation
- **NoCue Trajectories**: Evidence removal with preserved physical consistency
- **CF Trajectories**: Hard negative counterfactuals via state interventions

##  Dataset Structure

```
datasets/
+-- raw_data/         # Raw trajectory datasets
|   +-- doorkey/          # DoorKey-8x8-v0 (20 samples)
|   |   +-- triplets.jsonl    # 60 trajectories (20x3 variants)
|   |   +-- [sample_dirs]/     # Individual trajectory directories
|   |       +-- full/         # Successful trajectory frames
|   |       +-- nocue/        # Evidence-removed trajectory
|   |       +-- cf/           # Counterfactual trajectory
|   +-- memory/           # MemoryS13-v0 (40 samples)
|   |   +-- triplets.jsonl    # 120 trajectories (40x3 variants)
|   +-- lavagap/          # LavaGapS7-v0 (40 samples)
|   |   +-- triplets.jsonl    # 120 trajectories (40x3 variants)
|   +-- keycorridor/      # KeyCorridorS6R3-v0 (40 samples)
|   |   +-- triplets.jsonl    # 120 trajectories (40x3 variants)
|   +-- multiroom/        # MultiRoom-N6-v0 (40 samples)
|   |   +-- triplets.jsonl    # 120 trajectories (40x3 variants)
|   +-- redblue/          # RedBlueDoors-8x8-v0 (20 samples)
|       +-- triplets.jsonl    # 60 trajectories (20x3 variants)
+-- exams/            # Generated exam questions
    +-- images/           # Composite images for questions
    |   +-- taskA/        # Task A question images
    |   +-- taskB/        # Task B question images
    |   +-- taskC/        # Task C storyboard images
    +-- task_a_exam.jsonl # Task A questions (200 items)
    +-- task_b_exam.jsonl # Task B questions (200 items)
    +-- task_c_exam.jsonl # Task C questions (1200 items)
    +-- manifest.json     # Build metadata and statistics
```

##  Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Sanity check
python -c "import gymnasium; env = gymnasium.make('MiniGrid-DoorKey-8x8-v0'); print('OK: MiniGrid ready'); env.close()"
```

### 2. Generate Raw Dataset (Optional)
The raw dataset is already generated and verified. To regenerate:
```bash
# Generate all tasks (40 samples each for most tasks, 20 for doorkey/redblue)
./scripts/generate_all_datasets.sh

# Or generate individual tasks
python scripts/tasks/doorkey/gen_doorkey_triplets.py --out-dir datasets/raw_data/doorkey --num 10 --env-id MiniGrid-DoorKey-8x8-v0 --seed-start 1000
```

### 3. Validate Raw Dataset
```bash
# Validate all datasets
./scripts/validate_all_datasets.sh

# Or validate individual tasks
python scripts/tasks/doorkey/validate_doorkey_triplets.py --triplets datasets/raw_data/doorkey/triplets.jsonl --root datasets/raw_data/doorkey
```

### 4. Generate Exam Questions
The exam questions are already generated. To regenerate:
```bash
# Generate all exam tasks from raw data
python scripts/build_exam.py

# Or generate with custom parameters
python scripts/build_exam.py --root datasets/raw_data --out-dir datasets/exams --a-per-traj 1 --b-per-traj 1 --c-k 8
```

### 5. Run Inference and Audit (Capability-Preserving)
```bash
# Run inference with SiliconFlow (recommended for testing)
export SILICONFLOW_API_KEY="your_key"
python scripts/run_inference.py \
  --exam_dir datasets/exams \
  --responses_dir runs/responses \
  --backend openai_compatible \
  --provider siliconflow \
  --model "Pro/Qwen/Qwen2.5-VL-7B-Instruct" \
  --progress_total --progress_show_uid

# Audit results (auto-discovers latest experiment)
python scripts/audit_inference_outputs.py \
  --responses_dir runs/responses \
  --exam_task all \
  --out_report audit_report.json
```

### 6. Score Results (Final Evaluation)
```bash
# Score the latest experiment (auto-discovers)
python scripts/score_exam.py

# Score with custom parameters
python scripts/score_exam.py \
  --b_acc_threshold 0.85 \
  --out runs/scores/final_evaluation.json

# View scoring report
cat runs/scores/score_report.json | jq '.overall_micro'
```

##  Research Tasks

### Task A: Atomic Transition Prediction (200 questions)
**Objective**: Test VLM's ability to predict next state given current state + action.

**Input**: Composite image showing current state + 4 candidate next states + action prompt
**Output**: Select correct next state (A/B/C/D choice)
**Evaluation**: 4-choice accuracy across transition types
**Key Challenge**: Requires understanding physical dynamics, not just visual similarity

### Task B: Perception/Understanding (200 questions)
**Objective**: Test VLM's ability to parse visual scenes into structured representations.

**Input**: Single agent POV frame (7x7x3 RGB)
**Output**: Canonical JSON with agent position/direction/carrying + objects list
**Evaluation**: Exact-match accuracy against ground truth state representation
**Key Challenge**: Precise spatial reasoning and object attribute extraction

### Task C: Judgment Robustness (1200 questions)
**Objective**: Test VLM's stability as a judge under controlled perturbations.

**Input**: 8-frame storyboard grid + success/failure classification prompt
**Output**: Binary success/fail judgment
**Variants**: Full (normal), NoCue (evidence masked), CF (counterfactual failures)
**Evaluation**: Judgment Consistency Rate (JCR) + Correctness across perturbations
**Key Challenge**: Temporal reasoning and robustness to evidence manipulation

##  Task Details

Each task directory contains detailed README with implementation specifics:

- [`scripts/tasks/doorkey/README.md`](scripts/tasks/doorkey/README.md) - DoorKey task logic
- [`scripts/tasks/memory/README.md`](scripts/tasks/memory/README.md) - Memory task logic
- [`scripts/tasks/lavagap/README.md`](scripts/tasks/lavagap/README.md) - LavaGap task logic
- [`scripts/tasks/keycorridor/README.md`](scripts/tasks/keycorridor/README.md) - KeyCorridor task logic
- [`scripts/tasks/multiroom/README.md`](scripts/tasks/multiroom/README.md) - MultiRoom task logic
- [`scripts/tasks/redblue/README.md`](scripts/tasks/redblue/README.md) - RedBlueDoor task logic

##  Development Scripts

### Raw Dataset Generation
```bash
# Generate complete raw dataset for all tasks
./scripts/generate_all_datasets.sh

# Individual task generation examples
python scripts/tasks/doorkey/gen_doorkey_triplets.py \
  --out-dir datasets/raw_data/doorkey --num 10 --env-id MiniGrid-DoorKey-8x8-v0 --seed-start 1000
```

### Raw Dataset Validation
    ```bash
# Validate all generated datasets
./scripts/validate_all_datasets.sh

# Individual task validation examples
python scripts/tasks/doorkey/validate_doorkey_triplets.py \
  --triplets datasets/raw_data/doorkey/triplets.jsonl --root datasets/raw_data/doorkey
    ```

### Exam Dataset Generation
    ```bash
# Generate all exam questions from raw data
python scripts/build_exam.py

# Generate with custom parameters
python scripts/build_exam.py \
  --root datasets/raw_data \
  --out-dir datasets/exams \
  --a-per-traj 1 --b-per-traj 1 --c-k 8

# View build statistics and manifest
cat datasets/exams/manifest.json
```

##  Key Concepts

### Full Trajectories
- **Definition**: Optimal successful trajectories generated via BFS/A* planning
- **Validation**: Must achieve success=True with reward > 0
- **Purpose**: Establish ground truth for comparison

### NoCue Trajectories
- **Definition**: Evidence removal while preserving physical trajectories
- **Mechanism**: Tile-level masking of task-critical objects in EARLY windows
- **Validation**: Same success/reward as Full, identical action sequences
- **Purpose**: Test evidence independence in judgment

### CF (Counterfactual) Trajectories
- **Definition**: Hard negative examples via minimal state interventions
- **Mechanism**: Environment modifications that make optimal actions fail
- **Validation**: Same action sequences, success=False with reward=0
- **Purpose**: Create challenging judgment tasks

##  Scientific Contributions

1. **World Model Deficit Hypothesis**: Links VLM judging instability to poor physical reasoning
2. **Controlled Benchmark**: Systematic evaluation across task complexities
3. **Evidence Removal Protocol**: NoCue diagnostic for evidence-dependent judgment
4. **Comprehensive Metrics**: JCR, accuracy, transition-type analysis

##  Citation

```bibtex

```

##  Notes

- Dataset generation requires MiniGrid installation
- All trajectories are deterministic and reproducible via seeds
- Manual verification completed for all 60 samples
- Framework designed for systematic VLM evaluation research


## License

This project is dual-licensed under MIT OR Apache-2.0. See `LICENSE-MIT` and `LICENSE-APACHE`.

