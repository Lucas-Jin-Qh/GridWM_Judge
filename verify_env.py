import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# --- Configuration strictly based on SSOT ---
ENV_ID = "MiniGrid-DoorKey-8x8-v0"
OUTPUT_DIR = "env_dump_v0"
TILE_SIZE = 64  # 64 pixels per tile. Total image ~512x512. 
                # If VLM fails this, we bump to 64.

# Clean up previous run
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

class AtomicStateExtractor:
    """
    Simulates the Ground Truth extraction logic defined in SSOT Section 2.1
    """
    def __init__(self, env):
        self.env = env.unwrapped
        
    def get_json(self):
        env = self.env
        
        # 1. Agent Info
        agent_entry = {
            "pos": tuple(int(x) for x in env.agent_pos),
            "dir": int(env.agent_dir), # 0:right, 1:down, 2:left, 3:up
            "carrying": env.carrying.type if env.carrying else None
        }
        
        # 2. Key Info (Scan grid)
        key_entry = {"state": "not_visible", "pos": None}
        # Check if carried
        if env.carrying and env.carrying.type == 'key':
            key_entry = {"state": "carried", "pos": None, "color": env.carrying.color}
        else:
            # Check grid
            for i, obj in enumerate(env.grid.grid):
                if obj and obj.type == 'key':
                    x, y = i % env.width, i // env.width
                    key_entry = {"state": "on_ground", "pos": (x, y), "color": obj.color}
                    break
                    
        # 3. Door Info
        door_entry = {"state": "unknown", "pos": None}
        for i, obj in enumerate(env.grid.grid):
            if obj and obj.type == 'door':
                x, y = i % env.width, i // env.width
                door_entry = {
                    "pos": (x, y),
                    "is_open": obj.is_open,
                    "is_locked": obj.is_locked,
                    "color": obj.color
                }
                break
                
        return {
            "mission": env.mission,
            "agent": agent_entry,
            "key": key_entry,
            "door": door_entry
        }

def run_sprint0():
    print("Executing Sprint 0: Visibility Check based on SSOT...")
    
    # Setup Environment
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = RGBImgObsWrapper(env, tile_size=TILE_SIZE)
    extractor = AtomicStateExtractor(env)
    
    obs, info = env.reset(seed=42) # Fixed seed for reproducibility
    
    # Sprint 0: Generate visibility contrast frames (locked vs open door)
    print("Generating visibility contrast frames...")

    # Save locked door frame
    img_locked = env.unwrapped.get_frame(highlight=True, tile_size=TILE_SIZE)
    state_locked = extractor.get_json()
    plt.imsave(f"{OUTPUT_DIR}/door_locked.png", img_locked)
    with open(f"{OUTPUT_DIR}/door_locked.json", "w") as f:
        json.dump(state_locked, f, indent=2)
    print(f"   -> Saved door_locked.png: Door is {'open' if state_locked['door']['is_open'] else 'locked'}")

    # Force open the door for contrast (only for Sprint 0 visibility check)
    for obj in env.unwrapped.grid.grid:
        if obj and obj.type == "door":
            obj.is_locked = False
            obj.is_open = True
            break

    # Save open door frame
    img_open = env.unwrapped.get_frame(highlight=True, tile_size=TILE_SIZE)
    state_open = extractor.get_json()
    plt.imsave(f"{OUTPUT_DIR}/door_open.png", img_open)
    with open(f"{OUTPUT_DIR}/door_open.json", "w") as f:
        json.dump(state_open, f, indent=2)
    print(f"   -> Saved door_open.png: Door is {'open' if state_open['door']['is_open'] else 'locked'}")

    # Reset environment for actual rollout
    obs, info = env.reset(seed=42)

    # Sprint 0: Debug sequence to prove forward can move and pos changes
    # Corrected sequence: from down(1), turn to up(3), then move up, then turn back and move down
    actions = [
        env.unwrapped.actions.left,     # down(1) -> right(0)
        env.unwrapped.actions.left,     # right(0) -> up(3)
        env.unwrapped.actions.forward,  # Move up to (1,4)
        env.unwrapped.actions.forward,  # Move up to (1,3)
        env.unwrapped.actions.right,    # up(3) -> right(0)
        env.unwrapped.actions.right,    # right(0) -> down(1)
        env.unwrapped.actions.forward,  # Move down to (1,4)
        env.unwrapped.actions.forward,  # Move down to (1,5) - back to start
    ]
    
    # Capture initial state
    steps_to_capture = [0] + list(range(1, len(actions) + 1))
    
    current_step = 0

    # Debug output for initial state
    front = env.unwrapped.grid.get(*env.unwrapped.front_pos)
    front_type = None if front is None else front.type
    print(f"  [DEBUG] front:{front_type} dir:{env.unwrapped.agent_dir} pos:{tuple(env.unwrapped.agent_pos)} (initial)")
    
    # Save Step 0
    img = env.unwrapped.get_frame(highlight=True, tile_size=TILE_SIZE)
    state = extractor.get_json()
    state["last_action"] = "none"
    save_artifact(current_step, img, state)
    
    for act in actions:
        current_step += 1
        obs, reward, terminated, truncated, info = env.step(act)
        
        # Debug output: show front cell type, direction, and position
        front = env.unwrapped.grid.get(*env.unwrapped.front_pos)
        front_type = None if front is None else front.type
        print(f"  [DEBUG] front:{front_type} dir:{env.unwrapped.agent_dir} pos:{tuple(env.unwrapped.agent_pos)}")

        img = env.unwrapped.get_frame(highlight=True, tile_size=TILE_SIZE)
        state = extractor.get_json()
        
        # Fix last_action: map action ID to name properly
        A = env.unwrapped.actions
        id2name = {A.left: "left", A.right:"right", A.forward:"forward", A.toggle:"toggle", A.pickup:"pickup", A.drop:"drop", A.done:"done"}
        state["last_action"] = id2name.get(act, str(act))
        
        save_artifact(current_step, img, state)
        
        if terminated or truncated:
            break
            
    print(f"[OK] Sprint 0 complete. Artifacts saved to '{OUTPUT_DIR}'")
    print("Task: Please inspect 'step_03.png' and 'step_03.json'.")

def save_artifact(step, img, state):
    # Save Image
    plt.imsave(f"{OUTPUT_DIR}/step_{step:02d}.png", img)
    
    # Save JSON
    with open(f"{OUTPUT_DIR}/step_{step:02d}.json", "w") as f:
        json.dump(state, f, indent=2)
    
    dir_names = ['right', 'down', 'left', 'up']
    dir_name = dir_names[state['agent']['dir']] if 0 <= state['agent']['dir'] < len(dir_names) else 'unknown'
    print(f"   -> Saved step {step}: Agent at {state['agent']['pos']} facing {dir_name}")

if __name__ == "__main__":
    run_sprint0()
