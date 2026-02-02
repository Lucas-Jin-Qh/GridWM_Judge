# Memory Task Implementation

This directory contains the Memory task implementation for the GridWM-Judge framework. Memory tests working memory: observe cue → navigate corridor → match correct object.

## Environment Specification

- **MiniGrid Environment**: `MiniGrid-MemoryS13-v0` (13×13 grid)
- **Task Description**: "go to the matching object at the end of the hallway"
- **Key Elements**: Cue object (left room), two end objects (right room), narrow corridor
- **Action Space**: 6 actions (left, right, forward, pickup, drop, toggle)

## Trajectory Variants

### Full Trajectories
**Generation Logic**:
- BFS planner finds shortest path from start to correct end object
- State representation: `(agent_x, agent_y, agent_dir)` (position-only, no object state)
- Object inference: Identifies cue and end objects by spatial layout
- Matching logic: Exact type+color match, fallback to type-only

**Implementation Notes**:
- Cue object: Leftmost room, sorted by position
- End objects: Rightmost room, top 2 candidates
- Corridor: Identified as narrowest passable column
- Success: Agent reaches tile adjacent to correct end object

### NoCue Trajectories
**Generation Logic**:
- Identical actions to Full trajectory
- EARLY window masking: mask cue objects before corridor entry
- Target objects: `["key"]` or `["ball"]` based on cue object type
- Window definition: frames where cue is visible before corridor (computed dynamically)
- Alignment threshold: 0.70

**Implementation Notes**:
- Corridor entry detection: finds narrowest passable column
- Masking preserves cue visibility in interaction contexts
- Uses semantic state encoding for precise tile identification
- **Critical**: Maintains evidence for frames where cue is interactable

### CF (Counterfactual) Trajectories
**Generation Logic**:
- Identical actions to Full trajectory
- State intervention: swap correct/incorrect end object positions
- Creates hard negative: agent chooses wrong object despite same actions
- **Failure Criteria**: Agent reaches wrong object tile

**Implementation Notes**:
- Position swap happens at trajectory start (environment level)
- Preserves object identities, only changes positions
- Validates that swap creates meaningful counterfactual
- Agent trajectory remains physically identical

## Code Implementation Details

### Object Inference Logic
```python
def pick_cue_and_ends(objs: List[Dict]) -> Tuple[Dict, Dict, Dict]:
    # Cue: leftmost room object
    cue_candidates = [o for o in objs if o["pos"][0] == min_x]
    cue = sorted(cue_candidates, key=lambda o: (o["pos"][1], o["type"], o["color"]))[0]

    # Ends: rightmost room objects (top 2)
    end_candidates = [o for o in objs if o["pos"][0] == max_x]
    end_sorted = sorted(end_candidates, key=lambda o: (o["pos"][1], o["type"], o["color"]))
    return cue, end_sorted[0], end_sorted[1]
```

### Matching Algorithm
```python
def match_good_bad(cue: Dict, end_a: Dict, end_b: Dict) -> Tuple[Dict, Dict]:
    # Exact match preferred
    if end_a["type"] == cue["type"] and end_a["color"] == cue["color"]:
        return end_a, end_b
    if end_b["type"] == cue["type"] and end_b["color"] == cue["color"]:
        return end_b, end_a

    # Type-only fallback
    if end_a["type"] == cue["type"]:
        return end_a, end_b
    if end_b["type"] == cue["type"]:
        return end_b, end_a
```

### Corridor Detection
```python
def infer_corridor_start_x(env) -> int:
    # Find narrowest passable column
    counts = [(x, sum(1 for y in range(height) if _is_passable(env, x, y)))
              for x in range(width)]
    min_passable = min(c for x, c in counts if x > 0)
    candidates = [x for x, c in counts if x > 0 and c == min_passable]
    return min(candidates)
```

### BFS Planning
- **Start**: Agent initial position and direction
- **Goal**: Position adjacent to correct end object
- **Actions**: Left/right turns, forward movement
- **Constraints**: Cannot pass through walls or objects

## Usage Examples

### Generate 10 Memory Triplets
```bash
python gen_memory_triplets.py \
  --out-dir datasets/raw_data/memory \
  --n 10 \
  --seed-start 2000
```

### Validate Generated Data
```bash
python validate_memory_triplets.py \
  --root datasets/raw_data/memory
```

## Files Structure

```
scripts/tasks/memory/
├── gen_memory_triplets.py      # Main generation script
├── validate_memory_triplets.py # Validation script
└── README.md                   # This documentation
```

## Key Parameters

- **Planning**: max_nodes=300000 for BFS, max_plan_steps=512
- **NoCue**: mask_strength_target=0.02, alignment_min=0.7
- **CF**: position swap at environment level
- **Rendering**: tile_size=32, agent POV 7×7 grid

## Manual Verification Results

All 10 Memory samples manually verified:
- ✅ Full trajectories: 3-13 navigation steps, all successful
- ✅ NoCue trajectories: EARLY masking of key/ball, preserved success
- ✅ CF trajectories: end object position swap, consistent failure
- ✅ Action sequences: 100% identical across variants
- ✅ Logic validation: cue observation → corridor navigation → correct matching

## Generation Logic

### Full Trajectory
- Uses BFS planning to find optimal path from start to correct end object
- Automatically infers cue and end objects from environment layout
- Matches objects by type and color (exact match preferred, type-only fallback)
- Rejects any truncated trajectories for industrial stability

### NoCue Trajectory
- **EARLY window**: Masks cue object evidence before entering corridor
- **Semantic masking**: Uses state encoding to precisely locate cue tiles
- **Interaction preservation**: Maintains interaction evidence integrity
- **Alignment scoring**: Ensures masking accuracy ≥ 0.70

### CF Trajectory (Counterfactual)
- **Type-2 intervention**: Swaps positions of correct and incorrect end objects
- **Hard negative**: Agent executes same actions but chooses wrong object (reward=0)
- **Physics invariance**: Agent trajectory identical to Full trajectory
- **Goal invisibility**: End objects never become visible in counterfactual

## Usage

### 1. Generate Memory Triplets

```bash
python gen_memory_triplets.py \
  --env-id MiniGrid-MemoryS13Random-v0 \
  --out-dir ./dataset_memory \
  --n 100 \
  --seed-start 0
```

**Parameters:**
- `--env-id`: Environment ID (default: MiniGrid-MemoryS13Random-v0, compatible with SSOT requirements)
- `--out_dir`: Output directory
- `--n`: Number of triplets to generate (default: 100)
- `--seed-start`: Starting seed (default: 0)
- `--tile-size`: Tile size for rendering (default: 32)
- `--max-plan-steps`: Maximum planning steps (default: 512)
- `--max-episode-steps`: Maximum episode steps (default: 2048)
- `--mask-strength-target`: Target mask strength (default: 0.02)
- `--mask-ratio-max`: Maximum mask ratio (default: 0.35)
- `--alignment-min`: Minimum alignment score (default: 0.70)
- `--min-visible-frames`: Minimum visible frames (default: 1)
- `--max-visible-frames`: Maximum visible frames (default: 10)
- `--resume`: Resume from existing data

### 2. Validate Memory Triplets

```bash
python validate_memory_triplets.py \
  --root ./dataset_memory \
  --max-groups 50
```

**Parameters:**
- `--root`: Dataset root directory containing triplets.jsonl
- `--max-groups`: Maximum number of groups to validate (default: -1, all)

## Output Structure

```
dataset_memory/
├── memory_s000000/
│   ├── full/
│   │   ├── step_000.png
│   │   ├── step_001.png
│   │   └── ...
│   ├── nocue/
│   │   ├── step_000.png
│   │   ├── step_001.png
│   │   └── ...
│   └── cf/
│       ├── step_000.png
│       ├── step_001.png
│       └── ...
├── memory_s000001/
│   └── ...
├── triplets.jsonl
└── _tmp/ (temporary directory)
```

## Validation Checks

### Hard Gates
- **Contract Compliance**: `model_input_fields` must equal `["frames", "mission", "terminated", "truncated"]`
- **Action Identity**: Full/NoCue/CF actions must be identical
- **Physics Invariance**: Agent position/direction identical across variants
- **Outcome Gates**:
  - Full: success=True, terminated=True, reward>0, truncated=False
  - CF: success=False, terminated=False, reward=0, truncated=False

### NoCue Validation
- **EARLY Window**: Masking only within specified early corridor-entry frames
- **D5 Hard Gate**: Never mask frames where cue is directly interactable
- **Target Accuracy**: Only cue tiles are masked

### Pixel-Level Audit
- **Masked frames**: Differences only in cue tile regions
- **Unmasked frames**: Full and NoCue images must be identical
- **Window enforcement**: No masking after corridor entry

## Key Features

### Object Inference Logic
- **Cue identification**: Leftmost room object (sorted by position and type)
- **End objects**: Rightmost room objects (top 2 candidates)
- **Matching algorithm**: Exact type+color match, with type-only fallback
- **Corridor detection**: Identifies narrowest passable column as corridor start

### Masking Strategy
- **Tile-level precision**: Uses semantic encoding for pixel-perfect masking
- **Color-aware masking**: Handles multi-color object environments
- **Budget enforcement**: Prevents excessive masking that could destroy task

### Engineering Robustness
- **Atomic Writes**: Temporary directory approach prevents corruption
- **Resource Safety**: Comprehensive try/finally environment cleanup
- **Truncation Rejection**: Industrial-grade stability filtering
- **Seed Resilience**: Robust object inference across environment variations

## Performance Notes

- **Generation time**: ~30-90 seconds per triplet depending on environment complexity
- **Memory usage**: Moderate due to image-based state representation
- **Success rate**: ~70-85% depending on environment constraints and masking parameters
- **Validation speed**: Fast pixel-level checks enable quick auditing

## Troubleshooting

### Common Issues

1. **"Ambiguous cue/end match"**
   - Object matching logic failed; try different seed
   - May indicate unusual environment layout

2. **"BFS failed to find valid plan"**
   - Pathfinding issue; environment may have blocking obstacles
   - Try different seed or less constrained environment

3. **"nocue_no_frames"**
   - Cue not visible in early frames or all visibility involves interaction
   - Adjust window parameters or seed selection

4. **"physics_drift"**
   - Environment state divergence between variants
   - Check CF intervention logic and state consistency

### Validation Failures

- **action_mismatch**: Triplet variants from different generation runs mixed
- **diff_outside_cue_tiles**: Masking affected non-cue pixels
- **masked_interaction_frame**: Violated interaction evidence preservation
- **window_policy != EARLY**: Incorrect masking window specification

## Environment Compatibility

- **Recommended**: `MiniGrid-MemoryS13Random-v0` (13x13 grid, matches SSOT requirements)
- **Alternatives**: `MiniGrid-MemoryS17Random-v0`, `MiniGrid-MemoryS13-v0`
- **Avoid**: Smaller environments may lack sufficient navigation challenge

## Compliance

These scripts are designed to comply with:
- GridWM-Judge Project Overview Scientific Contract (Final SSOT)
- NoCue_spec.md
- TRIPLET_AUDIT_SPEC_REVIEWER.md

All hard gates and semantic requirements are implemented and validated.
