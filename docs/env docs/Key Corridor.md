

#Key Corridor

# 

| Action Space      | `Discrete(7)`                                                                                                                                                                                                                 |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Observation Space | `Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace(<function KeyCorridorEnv._gen_mission at 0x7f813f5079d0>, [['blue', 'green', 'grey', 'purple', 'red', 'yellow'], ['ball']]))` |
| Reward Range      | `(0, 1)`                                                                                                                                                                                                                      |
| Creation          | `gymnasium.make("MiniGrid-KeyCorridorS6R3-v0")`                                                                                                                                                                               |

---

## Description

This environment is similar to the locked room environment, but there are multiple registered environment configurations of increasing size, making it easier to use curriculum learning to train an agent to solve it. The agent has to pick up an object which is behind a locked door. The key is hidden in another room, and the agent has to explore the environment to find it. The mission string does not give the agent any clues as to where the key is placed. This environment can be solved without relying on language.

---

## Mission Space

"pick up the {color} {obj_type}"

- `{color}` is the color of the object. Can be "red", "green", "blue", "purple", "yellow" or "grey".
- `{type}` is the type of the object. Can be "ball" or "key".

---

## Action Space

| Num | Name    | Action            |
| --- | ------- | ----------------- |
| 0   | left    | Turn left         |
| 1   | right   | Turn right        |
| 2   | forward | Move forward      |
| 3   | pickup  | Pick up an object |
| 4   | drop    | Unused            |
| 5   | toggle  | Unused            |
| 6   | done    | Unused            |

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

1. The agent picks up the correct object.
2. Timeout (see `max_steps`).

---

## Registered Configurations

S: room size.  
R: Number of rows.

- `MiniGrid-KeyCorridorS3R1-v0`
- `MiniGrid-KeyCorridorS3R2-v0`
- `MiniGrid-KeyCorridorS3R3-v0`
- `MiniGrid-KeyCorridorS4R3-v0`
- `MiniGrid-KeyCorridorS5R3-v0`
- `MiniGrid-KeyCorridorS6R3-v0`
  
  
