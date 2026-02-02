

#Red Blue Door

# 

| Action Space      | `Discrete(7)`                                                                                                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Observation Space | `Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace(<function RedBlueDoorEnv._gen_mission at 0x7f813f498d30>, None))` |
| Reward Range      | `(0, 1)`                                                                                                                                                          |
| Creation          | `gymnasium.make("MiniGrid-RedBlueDoors-8x8-v0")`                                                                                                                  |

---

## Description

The agent is randomly placed within a room with one red and one blue door facing opposite directions. The agent has to open the red door and then open the blue door, in that order. Note that, surprisingly, this environment is solvable without memory.

---

## Mission Space

"open the red door then the blue door"

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

1. The agent opens the blue door having already opened the red door.
2. The agent opens the blue door without having opened the red door yet.
3. Timeout (see `max_steps`).

---

## Registered Configurations

- `MiniGrid-RedBlueDoors-6x6-v0`
- `MiniGrid-RedBlueDoors-8x8-v0`
