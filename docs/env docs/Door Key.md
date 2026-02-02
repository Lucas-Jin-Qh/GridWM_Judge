# Door Key

# 

| Action Space      | `Discrete(7)`                                                                                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Observation Space | `Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace(<function DoorKeyEnv._gen_mission at 0x7f813f4985e0>, None))` |
| Reward Range      | `(0, 1)`                                                                                                                                                      |
| Creation          | `gymnasium.make("MiniGrid-DoorKey-8x8-v0")`                                                                                                                   |

---

## Description

This environment has a key that the agent must pick up in order to unlock a door and then get to the green goal square. This environment is difficult to solve because the reward is sparse. The agent only gets a reward when it reaches the goal. This environment can be solved without relying on language.

---

## Mission Space

"reach the goal"

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

1. The agent reaches the goal.
2. Timeout (see `max_steps`).

---

## Registered Configurations

S: size of map SxS.

- `MiniGrid-DoorKey-5x5-v0` (5x5 map)
- `MiniGrid-DoorKey-6x6-v0` (6x6 map)
- `MiniGrid-DoorKey-8x8-v0` (8x8 map)
- `MiniGrid-DoorKey-16x16-v0` (16x16 map)
