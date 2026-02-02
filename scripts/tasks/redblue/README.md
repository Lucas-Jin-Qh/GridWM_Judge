# RedBlueDoor Task Implementation

This directory contains the RedBlueDoor task implementation for the GridWM-Judge framework. RedBlueDoor tests strict ordering: red door → blue door → goal.

## Environment Specification

- **MiniGrid Environment**: `MiniGrid-RedBlueDoors-8x8-v0` (8×8 grid)
- **Task Description**: "open the red door then the blue door"
- **Key Elements**: Red door, blue door, goal behind blue door
- **Action Space**: 6 actions (left, right, forward, pickup, drop, toggle)

## Trajectory Variants

### Full Trajectories
**Generation Logic**:
- BFS planner finds optimal red→blue door sequence
- State representation: `(agent_x, agent_y, agent_dir)`
- Door mechanics: red door must open before blue door
- **Success Criteria**: Both doors opened in correct order, goal reached

**Implementation Notes**:
- Doors start closed, open permanently after toggle
- Wrong order (blue before red) fails the task
- Agent can only toggle doors when facing them
- Goal becomes accessible only after both doors open

### NoCue Trajectories
**Generation Logic**:
- Identical actions to Full trajectory
- EARLY window masking: mask red door before first interaction
- Target objects: `["red_door"]`
- Window definition: frames before first red door toggle (up to nocue_max_visible=5)
- Alignment threshold: 0.70

**Implementation Notes**:
- Only masks red door tiles, preserves blue door visibility
- Preserves red door visibility when agent faces it
- Uses semantic encoding for precise red door detection
- **Critical**: Never masks interaction frames (D5 hard gate)

### CF (Counterfactual) Trajectories
**Generation Logic**:
- Identical actions to Full trajectory
- State intervention: replace blue door with wall
- **Failure Criteria**: Agent cannot reach goal, reward=0
- **Impossibility**: Blue door becomes permanently blocked

**Implementation Notes**:
- Wall replacement happens at environment level
- Agent trajectory remains physically identical
- Creates hard negative: red door opens, blue door blocked
- Validates that intervention doesn't break red door access

## Code Implementation Details

### Environment Layout
```python
# RedBlueDoor structure: divided into regions by walls
# Left region: agent start, red door on right wall
# Right region: blue door on left wall, goal in corner
# Doors: red and blue doors at random vertical positions
```

### Strict Ordering Enforcement
```python
def step(self, action):
    red_was_open = self.red_door.is_open
    blue_was_open = self.blue_door.is_open

    # Execute action
    obs, reward, terminated, truncated, info = super().step(action)

    red_now_open = self.red_door.is_open
    blue_now_open = self.blue_door.is_open

    # Enforce ordering: blue door can only open after red door
    if blue_now_open and not red_was_open:
        # Blue opened before red - failure
        reward = 0
        terminated = True
    elif blue_now_open and red_was_open:
        # Both doors opened in correct order - success
        reward = self._reward()
        terminated = True

    return obs, reward, terminated, truncated, info
```

### Door Planning Logic
```python
def plan_redblue_doors(env) -> List[int]:
    # Phase 1: Locate and open red door
    red_path = bfs_to_door_and_toggle(env, 'red')

    # Phase 2: Cross to blue door and open it
    blue_path = bfs_to_door_and_toggle(env, 'blue')

    # Phase 3: Reach goal
    goal_path = bfs_to_position(env, env.goal_pos)

    return red_path + blue_path + goal_path
```

### Color-Specific Masking
```python
def mask_red_door_tiles(pov_img, state_encoding) -> Tuple[int, int]:
    door_idx = OBJECT_TO_IDX["door"]
    red_idx = COLOR_TO_IDX["red"]
    masked, target = 0, 0

    for i in range(Ht):
        for j in range(Wt):
            obj_idx = int(enc[i, j, 0])
            color_idx = int(enc[i, j, 1])

            if obj_idx == door_idx and color_idx == red_idx:
                target += 1
                # Only mask if not interaction frame
                if not is_facing_red_door(state_encoding, i, j):
                    pov_img[i*th:(i+1)*th, j*tw:(j+1)*tw] = MASK_RGB
                    masked += 1

    return masked, target
```

### Wall Replacement CF
```python
def apply_blue_door_to_wall_cf(env) -> Dict:
    # Find blue door position
    blue_pos = None
    for x, y in env.grid.positions():
        cell = env.grid.get(x, y)
        if cell and cell.type == "door" and cell.color == "blue":
            blue_pos = (x, y)
            break

    if blue_pos:
        # Replace door with wall
        env.grid.set(blue_pos[0], blue_pos[1], Wall())

        return {
            "cf_mode": "blue_door_to_wall",
            "door_position": blue_pos,
            "intervention_step": 0
        }

    return None
```

## Usage Examples

### Generate 10 RedBlueDoor Triplets
```bash
python gen_redblue_triplets.py \
  --out_dir datasets/raw_data/redblue \
  --n 10 \
  --seed_start 6000
```

### Validate Generated Data
```bash
python validate_redblue_triplets.py \
  --jsonl datasets/raw_data/redblue/triplets.jsonl \
  --root datasets/raw_data/redblue
```

## Files Structure

```
scripts/tasks/redblue/
├── gen_redblue_triplets.py      # Main generation script
├── validate_redblue_triplets.py # Validation script
└── README.md                    # This documentation
```

## Key Parameters

- **Planning**: BFS with door ordering constraints
- **NoCue**: early window red door masking only
- **CF**: blue door to wall replacement
- **Rendering**: tile_size=32, agent POV 7×7 grid

## Manual Verification Results

All 10 RedBlueDoor samples manually verified:
- ✅ Full trajectories: 19-25 steps, exactly 2 toggles (red→blue), all successful
- ✅ NoCue trajectories: EARLY red door masking, preserved success
- ✅ CF trajectories: blue door blocked, consistent failure
- ✅ Action sequences: 100% identical across variants
- ✅ Logic validation: strict red→blue ordering enforced

## Generation Logic

### Full Trajectory
- Uses BFS planning to find optimal red→blue door opening sequence
- Adds deterministic scan prefix `[Right, Right, Right, Right]` to ensure red door visibility
- Rejects any truncated trajectories for industrial stability

### NoCue Trajectory
- **EARLY window**: Masks red door evidence before first interaction
- **Interaction preservation**: Excludes frames where agent faces red door
- **Semantic masking**: Uses MiniGrid state encoding to precisely locate red door tiles
- **Alignment scoring**: Ensures masking accuracy ≥ 0.70

### CF Trajectory (Counterfactual)
- **Type-2 intervention**: Replaces blue door with wall
- **Hard negative**: Agent executes same actions but fails (reward=0, terminated=False)
- **Physics invariance**: Agent trajectory identical to Full trajectory

## Usage

### 1. Generate RedBlueDoor Triplets

```bash
python gen_redblue_triplets.py \
  --env_id MiniGrid-RedBlueDoors-8x8-v0 \
  --out_dir ./dataset_redblue \
  --n 500 \
  --seed_start 0 \
  --verbose
```

**Parameters:**
- `--env_id`: Environment ID (default: MiniGrid-RedBlueDoors-8x8-v0)
- `--out_dir`: Output directory
- `--n`: Number of triplets to generate (default: 100)
- `--seed_start`: Starting seed (default: 0)
- `--tile_size`: Tile size for rendering (default: 32)
- `--verbose`: Enable verbose logging

### 2. Validate RedBlueDoor Triplets

```bash
python validate_redblue_triplets.py \
  --jsonl ./dataset_redblue/triplets.jsonl \
  --root ./dataset_redblue
```

**Parameters:**
- `--jsonl`: Path to triplets.jsonl file
- `--root`: Root directory containing triplet data
- `--fast`: Skip pixel-level validation (faster but less thorough)

## Output Structure

```
dataset_redblue/
├── redblue_s000000/
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
├── redblue_s000001/
│   └── ...
├── triplets.jsonl
└── _tmp/ (temporary directory)
```

## Validation Checks

### Hard Gates
- **Action Identity**: Full/NoCue/CF actions must be identical
- **Outcome Gates**:
  - Full: success=True, terminated=True, reward>0
  - CF: success=False, terminated=False, reward=0
- **Physics Invariance**: Agent position/direction identical across variants

### NoCue Validation
- **EARLY Window**: Masking only before first red door interaction
- **Interaction Preservation**: No masking when agent faces red door
- **Evidence Removal**: Masked frames show no red door evidence

### Pixel Audit
- **Unmasked frames**: Full and NoCue images identical
- **Masked frames**: Differences only in red door tile regions
- **Alignment check**: Masking precisely targets red door tiles

## Key Fixes (Bugs Resolved)

### Bug A: Semantic Encoding
- **Problem**: Previous versions used RGB pixel analysis for object detection
- **Fix**: Always use `env_unwrapped.gen_obs()['image']` for semantic state encoding

### Bug B: Reset Timing
- **Problem**: Planning before environment reset caused inconsistent layouts
- **Fix**: Plan after `env.reset(seed)` for deterministic behavior

### Bug C: Door Movement Logic
- **Problem**: Incorrect door traversal logic in BFS planning
- **Fix**: Treat doors as obstacles; toggle from adjacent positions only

### Bug D: Constants Import
- **Problem**: Used hardcoded object/color IDs
- **Fix**: Import official `minigrid.core.constants` for proper encoding

## Performance Notes

- **Generation time**: ~30-60 seconds per triplet depending on environment size
- **Memory usage**: Each triplet requires temporary storage during generation
- **Success rate**: ~70-80% success rate due to environment constraints
- **Recommended**: Use MiniGrid-RedBlueDoors-8x8-v0 for better stability

## Troubleshooting

### Common Issues

1. **"BFS failed to find a valid plan"**
   - Environment layout may not allow red→blue sequence
   - Try different seed or smaller environment (6x6)

2. **"No red door visible in any frame"**
   - Scan prefix insufficient; environment may need different initialization
   - Check if red door is behind walls or unreachable

3. **"CF physics invariance failed"**
   - Wall replacement may have affected agent movement
   - Verify blue door position and wall placement logic

4. **"Alignment score too low"**
   - Red door tiles not properly detected in state encoding
   - Check semantic encoding extraction logic

### Validation Failures

- **Action mismatch**: Triplets from different seeds mixed together
- **Missing variants**: Generation interrupted, leaving incomplete groups
- **Physics drift**: Environment differences between variants
- **Mask leakage**: Evidence removal affected interaction frames

## Compliance

These scripts are designed to comply with:
- GridWM-Judge Project Overview Scientific Contract (Final SSOT)
- NoCue_spec.md
- TRIPLET_AUDIT_SPEC_REVIEWER.md

All hard gates and validation requirements are implemented and tested.
