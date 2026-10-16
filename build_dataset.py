"""Replay every good human take into LIBERO and save the successful replays as a LeRobot dataset."""
import argparse
import csv
import os
import re
import shutil

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument("--repo_id", default="local/hmnd_ego_orange_juice")
ap.add_argument("--root", default="data/lerobot_dataset")
ap.add_argument("--layouts_per_take", type=int, default=2)
ap.add_argument("--noise", type=float, default=0.3, help="DART noise on every 2nd replay")
ap.add_argument("--max_takes", type=int, default=0, help="0 = all; small number for a test run")
ap.add_argument("--grasp_dz", type=float, default=0.04)
ap.add_argument("--release_dz", type=float, default=0.06)
ap.add_argument("--lift_min", type=float, default=0.17, help="floor: carton must clear the sim basket")
ap.add_argument("--lift_max", type=float, default=0.20)
args = ap.parse_args()

HZ, STEP_M, TASK_ID = 20, 0.0098, 9
V_MAX, MAX_STEPS = 0.18, 230             # m/s peak speed, step budget used to set the time stretch
MAX_SAVE_STEPS = 270                     # never save a demo near the 280-step eval limit
OPEN, CLOSE = -1.0, 1.0
TRAIN_LAYOUTS = list(range(10, 50))      # layouts 0-9 are held out for evaluation
FLIP = True                              # LeRobot's env processor rotates images 180 deg at eval
rng = np.random.default_rng(0)

kp = np.load("data/perception/keyposes.npz")
takes = sorted(k[:-len("_grasp")] for k in kp.files if re.fullmatch(r"take_\d+_grasp", k))
takes = [t for t in takes if bool(kp[f"{t}_reliable"])]
if args.max_takes:
    takes = takes[:args.max_takes]

suite = benchmark.get_benchmark_dict()["libero_object"]()
task = suite.get_task(TASK_ID)
init_states = suite.get_task_init_states(TASK_ID)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)

features = {
    "observation.images.image": {"dtype": "video", "shape": (256, 256, 3),
                                 "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "video", "shape": (256, 256, 3),
                                  "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}
if os.path.exists(args.root):
    shutil.rmtree(args.root)
ds = LeRobotDataset.create(repo_id=args.repo_id, fps=HZ, features=features,
                           root=args.root, robot_type="panda", use_videos=True)


def quat2axisangle(q):
    """robosuite convention (x, y, z, w), same as LeRobot's env processor."""
    w = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    if den < 1e-10:
        return np.zeros(3)
    return q[:3] * 2.0 * np.arccos(w) / den


def state_of(obs):
    return np.r_[obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                 obs["robot0_gripper_qpos"]].astype(np.float32)


def img(x):
    return (x[::-1, ::-1] if FLIP else x).copy()


def quintic(p0, p1, sec):
    sec = max(sec, 1.875 * np.linalg.norm(p1 - p0) / V_MAX)          # peak speed under V_MAX
    n = max(int(round(sec * HZ)), 2)
    tau = np.linspace(0, 1, n + 1)[1:]
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return p0 + s[:, None] * (p1 - p0)


def plan_for(take, obs):
    key = lambda f: kp[f"{take}_{f}"]
    J, B = obs["orange_juice_1_pos"], obs["basket_1_pos"]
    home = obs["robot0_eef_pos"].copy()
    g, r, rest = key("grasp")[:2], key("release")[:2], key("rest")[:2]
    vh, vs = r - g, B[:2] - J[:2]
    ang = np.arctan2(vs[1], vs[0]) - np.arctan2(vh[1], vh[0])
    scale = np.linalg.norm(vs) / np.linalg.norm(vh)
    Rz = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    approach = (J[:2] + scale * Rz @ (rest - g)) - J[:2]
    approach /= np.linalg.norm(approach) + 1e-9

    lift = np.clip(key("lift") - key("grasp")[2], args.lift_min, args.lift_max)
    reach_h = max(key("t_grasp") - 0.4, 0.8)
    carry_h = key("t_release") - key("t_grasp")
    slow = min(2.0, (MAX_STEPS - 45) / HZ / (reach_h + carry_h))      # fit the step budget
    reach_s, carry_s = reach_h * slow, carry_h * slow

    zg = J[2] + args.grasp_dz
    pre = np.r_[J[:2] + 0.03 * approach, zg + 0.10]
    grasp, up = np.r_[J[:2], zg], np.r_[J[:2], zg + lift]
    above, release = np.r_[B[:2], zg + lift], np.r_[B[:2], zg + args.release_dz]
    plan = [(quintic(home, pre, 0.6 * reach_s), OPEN),
            (quintic(pre, grasp, 0.4 * reach_s), OPEN),
            (np.repeat(grasp[None], 15, 0), CLOSE),
            (quintic(grasp, up, 0.25 * carry_s), CLOSE),
            (quintic(up, above, 0.50 * carry_s), CLOSE),
            (quintic(above, release, 0.25 * carry_s), CLOSE),
            (np.repeat(release[None], 10, 0), OPEN),
            (quintic(release, release + [0, 0, 0.10], 1.0), OPEN)]
    ref = np.concatenate([p for p, _ in plan])
    grip = np.concatenate([np.full(len(p), gr) for p, gr in plan])
    return ref, grip, home, lift


def run_episode(take, layout, noisy):
    env.reset()
    obs = env.set_init_state(init_states[layout])
    for _ in range(10):
        obs, *_ = env.step(np.r_[np.zeros(6), OPEN])
    ref, grip, home, lift = plan_for(take, obs)
    q_home = obs["robot0_eef_quat"].copy()
    frames, prev, success, after = [], home, False, 0
    for p, gr in zip(ref, grip):
        eef = obs["robot0_eef_pos"]
        a_pos = np.clip((p - prev) / STEP_M + 0.6 * (p - eef) / STEP_M, -1, 1)
        rot_err = (Rot.from_quat(q_home) * Rot.from_quat(obs["robot0_eef_quat"]).inv()).as_rotvec()
        a_rot = np.clip(0.5 * rot_err / 0.1, -1, 1)
        action = np.r_[a_pos, a_rot, gr].astype(np.float32)
        frames.append({"observation.images.image": img(obs["agentview_image"]),
                       "observation.images.image2": img(obs["robot0_eye_in_hand_image"]),
                       "observation.state": state_of(obs), "action": action})
        executed = action.copy()
        if noisy:                                                        # DART: push, keep clean label
            executed[:6] = np.clip(executed[:6] + rng.normal(0, args.noise, 6), -1, 1)
        obs, _, done, _ = env.step(executed)
        prev = p
        success = success or bool(done)
        if success:                                                      # eval stops at success too
            after += 1
            if after > 10:
                break
    return frames, success or bool(env.check_success()), lift


log, saved = [], 0
for i, take in enumerate(takes):
    for j in range(args.layouts_per_take):
        layout = TRAIN_LAYOUTS[(i * args.layouts_per_take + j) % len(TRAIN_LAYOUTS)]
        noisy = j % 2 == 1 and args.noise > 0
        frames, ok, lift = run_episode(take, layout, noisy)
        keep = ok and len(frames) <= MAX_SAVE_STEPS
        log.append((take, layout, int(noisy), int(ok), len(frames), round(float(lift), 3), int(keep)))
        print(f"{take} layout {layout:2d} {'noisy' if noisy else 'clean'}: "
              f"{'SUCCESS' if ok else 'fail'} ({len(frames)} steps, lift {100 * lift:.0f} cm)"
              f"{'' if keep or not ok else '  [too long, not saved]'}")
        if keep:
            for f in frames:
                ds.add_frame({**f, "task": task.language})
            ds.save_episode()
            saved += 1
ds.finalize()
env.close()

with open("data/replay_log.csv", "w", newline="") as f:
    csv.writer(f).writerows([("take", "layout", "noisy", "success", "steps", "lift", "saved")] + log)
clean = [r for r in log if not r[2]]
noisy = [r for r in log if r[2]]
kept_steps = [r[4] for r in log if r[6]]
print(f"\nreplays {len(log)}, saved {saved}")
print(f"clean success {sum(r[3] for r in clean)}/{len(clean)}, "
      f"noisy success {sum(r[3] for r in noisy)}/{len(noisy)}")
print(f"saved episode length: min {min(kept_steps)}, max {max(kept_steps)}")
print(f"dataset at {args.root}")