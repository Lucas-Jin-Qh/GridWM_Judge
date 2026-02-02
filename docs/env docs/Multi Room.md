# Multi Room

# 

| Action Space      | `Discrete(7)`                                                                                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Observation Space | `Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace(<function MultiRoomEnv._gen_mission at 0x7f813f48e8b0>, None))` |
| Reward Range      | `(0, 1)`                                                                                                                                                        |
| Creation          | `gymnasium.make("MiniGrid-MultiRoom-N6-v0")`                                                                                                                    |

---

## Description

This environment has a series of connected rooms with doors that must be opened in order to get to the next room. The final room has the green goal square the agent must get to. This environment is extremely difficult to solve using RL alone. However, by gradually increasing the number of rooms and building a curriculum, the environment can be solved.

---

## Mission Space

"traverse the rooms to get to the goal"

---

## Action Space

| Num | Name    | Action                    |
| --- | ------- | ------------------------- |
| 0   | left    | Turn left                 |
| 1   | right   | Turn right                |
| 2   | forward | Move forward              |
| 3   | pickup  | Unused                    |
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

1. The agent reaches the goal.
2. Timeout (see `max_steps`).

---

## Registered Configurations

S: size of map SxS.  
N: number of rooms.

- `MiniGrid-MultiRoom-N2-S4-v0` (two small rooms)
- `MiniGrid-MultiRoom-N4-S5-v0` (four rooms)
- `MiniGrid-MultiRoom-N6-v0` (six rooms)
