"""Measure the LIBERO orange-juice task before replaying human data into it."""
import os

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASK_ID = 9
suite = benchmark.get_benchmark_dict()["libero_object"]()
task = suite.get_task(TASK_ID)
print("task:", task.language)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
init = suite.get_task_init_states(TASK_ID)
print("number of start layouts:", len(init))

OPEN = np.array([0, 0, 0, 0, 0, 0, -1.0])

def settle(k):
    obs = env.set_init_state(init[k])
    for _ in range(10):
        obs, *_ = env.step(OPEN)
    return obs

# 1) what the observation contains, and where things are
obs = settle(0)
print("\nobservation keys:")
for key, v in obs.items():
    print(f"  {key:40s} {np.shape(v)}")
print("\nfirst 5 layouts (metres, world frame):")
for k in range(5):
    obs = settle(k)
    objs = {key: np.round(v, 3) for key, v in obs.items()
            if key.endswith("_pos") and not key.startswith("robot0")}
    print(f"  layout {k}: eef {np.round(obs['robot0_eef_pos'], 3)}  {objs}")

# 2) how far one action moves the gripper
obs = settle(0)
p0 = obs["robot0_eef_pos"].copy()
for _ in range(10):
    obs, *_ = env.step(np.array([0.5, 0, 0, 0, 0, 0, -1.0]))
print("\n10 steps of action x=+0.5 moved the gripper by", np.round(obs["robot0_eef_pos"] - p0, 4))

# 3) how many steps the gripper needs to close
widths = []
for _ in range(20):
    obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
    widths.append(obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1])
print("gripper width while closing:", np.round(widths, 3))

# 4) save both camera views to check orientation
obs = settle(0)
imageio.imwrite("probe_agentview.png", obs["agentview_image"])
imageio.imwrite("probe_wrist.png", obs["robot0_eye_in_hand_image"])
print("\nsaved probe_agentview.png and probe_wrist.png")
env.close()