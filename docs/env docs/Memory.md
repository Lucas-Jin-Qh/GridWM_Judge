

#Memory

# 

| Action Space      | `Discrete(7)`                                                                                                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Observation Space | `Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace(<function MemoryEnv._gen_mission at 0x7f813f48e550>, None))` |
| Reward Range      | `(0, 1)`                                                                                                                                                     |
| Creation          | `gymnasium.make("MiniGrid-MemoryS7-v0")`                                                                                                                     |

---

## Description

This environment is a memory test. The agent starts in a small room where it sees an object. It then has to go through a narrow hallway which ends in a split. At each end of the split there is an object, one of which is the same as the object in the starting room. The agent has to remember the initial object, and go to the matching object at split.

---

## Mission Space

"go to the matching object at the end of the hallway"

---

## Action Space

| Num | Name    | Action                    |
| --- | ------- | ------------------------- |
| 0   | left    | Turn left                 |
| 1   | right   | Turn right                |
| 2   | forward | Move forward              |
| 3   | pickup  | Pick up an object         |
| 4   | drop    | Unused                    |
| 5   | toggle  | Toggle/activate an object |
| 6   | done    | Unused                    |

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

1. The agent reaches the correct matching object.
2. The agent reaches the wrong matching object.
3. Timeout (see `max_steps`).

---

## Registered Configurations

S: size of map SxS.

- `MiniGrid-MemoryS17Random-v0`
- `MiniGrid-MemoryS13Random-v0`
- `MiniGrid-MemoryS13-v0`
- `MiniGrid-MemoryS11-v0`
