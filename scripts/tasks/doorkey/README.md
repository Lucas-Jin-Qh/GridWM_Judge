# DoorKey Task Implementation

This directory contains the DoorKey task implementation for the GridWM-Judge framework. DoorKey tests sequential interaction skills: pickup key → unlock door → reach goal.

## Environment Specification

- **MiniGrid Environment**: `MiniGrid-DoorKey-8x8-v0`
- **Task Description**: "use the key to open the door and then get to the goal"
- **Key Objects**: Yellow key, locked yellow door, green goal
- **Action Space**: 7 actions (left, right, forward, pickup, drop, toggle, done)

## Trajectory Variants

### Full Trajectories
**Generation Logic**:
- BFS planner finds optimal path from start to goal
- State representation: `(agent_x, agent_y, agent_dir, has_key, door_state, key_present)`
- Actions: left/right for turning, forward for movement, pickup for key, toggle for door
- **Success Criteria**: Agent reaches goal tile with reward > 0

**Implementation Notes**:
- Door starts locked (state=2), becomes open (state=0) after toggle with key
- Key pickup removes key from grid and sets `has_key=True`
- Agent can only toggle door when facing it and holding the key
- Planning terminates when goal reached or max_steps exceeded

### NoCue Trajectories
**Generation Logic**:
- Identical actions to Full trajectory
- EARLY window masking: mask key tiles before pickup step
- Target objects: `["key"]`
- Mask strength: 0.02 (tile ratio)
- Alignment threshold: 0.70

**Implementation Notes**:
- Masking occurs in frames where key is visible but agent hasn't picked it up yet
- Uses tile-level masking to set key pixels to black (0,0,0)
- Preserves physical trajectory: agent position/direction unchanged
- **Critical**: No masking when agent is facing the key (interaction preservation)

### CF (Counterfactual) Trajectories
**Generation Logic**:
- Identical actions to Full trajectory
- State intervention: move goal to different empty tile
- Forbidden positions: agent trajectory path + front cells
- **Failure Criteria**: Same actions lead to reward=0, terminated=False

**Implementation Notes**:
- Goal relocation happens at trajectory start (step 0)
- Ensures new goal position is reachable but not on original path
- Validates that relocated goal doesn't create trivial failures
- Agent trajectory remains physically consistent

## Code Implementation Details

### State Representation
```python
@dataclass(frozen=True)
class DKState:
    ax: int; ay: int; ad: int  # Agent position and direction
    has_key: bool             # Key possession status
    door_state: int           # 0=open, 1=closed, 2=locked
    key_present: bool         # Key still in environment
```

### BFS Planning Logic
- **Start State**: Agent at initial position, no key, door locked, key present
- **Goal Check**: Agent position == goal position
- **Action Modeling**:
  - Left/Right: Change direction (ad = (ad ± 1) % 4)
  - Forward: Move if cell passable and door not blocking
  - Pickup: If facing key, set has_key=True, key_present=False
  - Toggle: If facing door and has_key, set door_state=0

### Validation Checks
- **Semantic Validation**: pickup before toggle, toggle changes door state, reaches goal
- **Physical Consistency**: Agent trajectory identical across variants
- **Evidence Removal**: NoCue masks only key tiles in early frames
- **Intervention Validity**: CF goal relocation doesn't create impossible scenarios

## Usage Examples

### Generate DoorKey Triplets
```bash
python gen_doorkey_triplets.py \
  --out-dir datasets/raw_data/doorkey \
  --num 10 \
  --env-id MiniGrid-DoorKey-8x8-v0 \
  --seed-start 1000
```

### Validate Generated Data
```bash
python validate_doorkey_triplets.py \
  --triplets datasets/raw_data/doorkey/triplets.jsonl \
  --root datasets/raw_data/doorkey
```

## Files Structure

```
scripts/tasks/doorkey/
├── gen_doorkey_triplets.py      # Main generation script
├── validate_doorkey_triplets.py # Validation script
└── README.md                    # This documentation
```

## Key Parameters

- **Planning**: max_nodes=200000 for BFS complexity control
- **NoCue**: nocue_max_visible=5, mask_ratio_max=0.35, alignment_min=0.7
- **CF**: max relocation attempts=20, forbidden trajectory exclusion
- **Rendering**: tile_size=32, grid_size=7×7 for agent POV

## Manual Verification Results

All 20 DoorKey samples manually verified:
- ✅ Full trajectories: 13-19 steps, all successful
- ✅ NoCue trajectories: EARLY masking, preserved success
- ✅ CF trajectories: goal relocation, consistent failure
- ✅ Action sequences: 100% identical across variants
- ✅ Logic validation: pickup → toggle → goal sequence correct

## Output Structure

```
data_doorkey_vFinal/
├── doorkey_s000000/
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
├── doorkey_s000001/
│   └── ...
├── triplets.jsonl
└── _tmp/ (temporary directory)
```

## Triplet Components

### Full
- Complete successful trajectory from DoorKey environment
- Agent picks up key, toggles door, reaches goal

### NoCue
- Same actions as Full, but key tiles are masked in early frames
- Tests whether model relies on internal simulation vs visual cues

### CF (Counterfactual)
- Same actions as Full, but goal is moved to unreachable position
- Tests whether model correctly judges failure when goal is relocated

## Compliance

These scripts are designed to comply with:
- GridWM-Judge Project Overview Scientific Contract (Final SSOT)
- NoCue_spec.md
- TRIPLET_AUDIT_SPEC_REVIEWER.md

## DoorKey Task Specification

- **Environment**: MiniGrid-DoorKey-8x8-v0 (recommended per SSOT)
- **NoCue Targets**: ["key"]
- **Window Policy**: EARLY (t < t_pickup)
- **Mask Strength**: 0.02
- **Alignment Threshold**: 0.70

## Validation Checks

The validator performs strict checks including:

### Hard Gates (D1-D7)
- **D1**: Action sequence identity across variants
- **D2**: Outcome gates (Full succeeds, CF fails)
- **D3**: NoCue evidence removal (not perception ablation)
- **D6**: Physical trajectory consistency
- **D7**: State encoding validity

### Task-Specific Semantics (E1)
- DoorKey logic: pickup key → toggle door → reach goal
- Door state transitions (locked → open)

### Pixel-Level Auditing
- Diff-in-Mask validation for NoCue variants
- Ensures masking affects only target tiles

## Troubleshooting

### Common Issues

1. **"Plan failed"**: BFS planner couldn't find solution within max_nodes limit
   - Try different seed or increase max_nodes

2. **"CF generation failed"**: Couldn't find suitable goal relocation position
   - Environment may have limited free space; try different seed

3. **"No cues to mask"**: No key visible before pickup
   - May indicate unusual environment layout; try different seed

4. **"Alignment fail"**: Masking didn't align well with key positions
   - Check tile size calculations; may need environment-specific tuning

### Resume Mode

Use `--resume` flag to continue interrupted generation:
- Skips existing group directories
- Continues from next seed
- Preserves existing triplets.jsonl

## Performance Notes

- Each triplet generation involves multiple environment resets and rollouts
- BFS planning can be computationally intensive for large grids
- Uses MiniGrid-DoorKey-8x8-v0 as specified in SSOT for optimal balance of complexity and performance
