# Lava Gap

# 

| Action Space      | `Discrete(7)`                                                                                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Observation Space | `Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace(<function LavaGapEnv._gen_mission at 0x7f813f48e3a0>, None))` |
| Reward Range      | `(0, 1)`                                                                                                                                                      |
| Creation          | `gymnasium.make("MiniGrid-LavaGapS7-v0")`                                                                                                                     |

---

## Description

The environment has a wall of lava with a small gap that the agent must cross to reach the green goal square. This environment is dynamic, the gap width, and the location of the gap changes every time the environment is reset. The agent must find the gap and cross it to the reach the goal. The mission string does not give the agent any clues as to where the gap is. The agent has to use its observation to locate the gap and cross it.

---

## Mission Space

"avoid the lava and get to the green goal square"

---

## Action Space

| Num | Name    | Action       |
| --- | ------- | ------------ |
| 0   | left    | Turn left    |
| 1   | right   | Turn right   |
| 2   | forward | Move forward |
| 3   | pickup  | Unused       |
| 4   | drop    | Unused       |
| 5   | toggle  | Unused       |
| 6   | done    | Unused       |

---

## Observation Encoding

- Each tile is encoded as a 3 dimensional tuple: `(OBJECT_IDX, COLOR_IDX, STATE)`
- `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in [minigrid/core/constants.py](http://minigrid.farama.org#minigrid/core/constants.py)
- `STATE` refers to the door state with 0=open, 1=closed and 2=locked

---

## Rewards

A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

---

## Termination

The episode ends if any one of the following conditions is met:

1. The agent reaches the goal square.
2. The agent falls into the lava.
3. Timeout (see `max_steps`).

---

## Registered Configurations

S: size of map SxS.

- `MiniGrid-LavaGapS5-v0` (5x5 map)
- `MiniGrid-LavaGapS6-v0` (6x6 map)
- `MiniGrid-LavaGapS7-v0` (7x7 map)
